"""Guard the shim against Dynamo drift.

SGLang is pre-1.0 and Dynamo tracks it closely -- that is why Dynamo carries
its own ``_compat.py``. When Dynamo starts importing a module or reading a
``ServerArgs`` field the shim does not provide, the symptom at runtime is an
import error deep in a worker launch, or worse, a silently missing attribute.

This test re-derives the contract from Dynamo's source and fails by name
instead. It is skipped when the Dynamo checkout is not present.
"""

from __future__ import annotations

import ast
import importlib
import os
import re
from pathlib import Path
from typing import Iterator, Set

import pytest

DYNAMO_SGLANG = Path(
    os.environ.get(
        "DYNAMO_SGLANG_PATH",
        Path(__file__).resolve().parents[2] / "dynamo/components/src/dynamo/sglang",
    )
)

requires_dynamo_source = pytest.mark.skipif(
    not DYNAMO_SGLANG.is_dir(),
    reason=f"Dynamo SGLang backend not found at {DYNAMO_SGLANG}",
)


def _source_files() -> Iterator[Path]:
    for path in sorted(DYNAMO_SGLANG.rglob("*.py")):
        # Dynamo's own tests monkeypatch and stub freely; they do not define
        # the production contract.
        if "tests" not in path.parts:
            yield path


def _sglang_imports(node: ast.AST) -> Set[tuple]:
    """``(module, symbol_or_None)`` pairs for sglang imports under ``node``."""
    found: Set[tuple] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and (child.module or "").startswith(
            "sglang"
        ):
            for alias in child.names:
                found.add((child.module, alias.name))
        elif isinstance(child, ast.Import):
            for alias in child.names:
                if alias.name.startswith("sglang"):
                    found.add((alias.name, None))
    return found


def _guards_import_error(handler: ast.ExceptHandler) -> bool:
    names = []
    if isinstance(handler.type, ast.Name):
        names = [handler.type.id]
    elif isinstance(handler.type, ast.Tuple):
        names = [e.id for e in handler.type.elts if isinstance(e, ast.Name)]
    return any(n in ("ImportError", "ModuleNotFoundError") for n in names)


def _allows_absent_import(handler: ast.ExceptHandler) -> bool:
    for statement in handler.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            if isinstance(statement.value, ast.Constant) and statement.value.value is None:
                continue
        if isinstance(statement, ast.Return):
            if statement.value is None or (
                isinstance(statement.value, ast.Constant) and statement.value.value is None
            ):
                continue
        if isinstance(statement, ast.Pass):
            continue
        return False
    return bool(handler.body)


def _collect() -> tuple:
    """Split imports into required ones and N/N-1 alternative groups.

    Dynamo's ``_compat.py`` deliberately tries a new SGLang path and falls back
    to an old one. A shim pinned to a single SGLang version legitimately
    provides only one side, so an alternative group passes when ANY member
    resolves. Explicit None/no-op fallbacks also permit the import to be absent.
    """
    required: Set[tuple] = set()
    groups: list = []

    def visit(node: ast.AST) -> None:
        # Only the OUTERMOST guarded try is a group. _compat.py nests them
        # (new path, then 0.5.18, then None), and every nested branch is an
        # alternative within the same choice, not a separate requirement.
        if isinstance(node, ast.Try) and any(
            _guards_import_error(h) for h in node.handlers
        ):
            alternatives = _sglang_imports(node)
            guarded.update(alternatives)
            optional = any(
                _guards_import_error(h) and _allows_absent_import(h)
                for h in node.handlers
            )
            if alternatives and not optional:
                groups.append(alternatives)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for path in _source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        guarded: Set[tuple] = set()
        visit(tree)
        required |= _sglang_imports(tree) - guarded
    return required, groups


# Fallback groups the shim deliberately leaves unresolved. Each is one where
# Dynamo's ``except ImportError`` sets the name to ``None`` and skips a feature,
# and skipping is the honest answer for a simulator. A new group still fails
# the test until it is either provided or added here with a reason.
DELIBERATELY_ABSENT = {
    "sglang.srt.arg_groups.model_override_base.get_default_attn_backend":
        "predicts SGLang's MLA attention backend; the fake has no attention "
        "backends, and None makes Dynamo skip the DCP backend check",
    "sglang.srt.configs.model_config.is_deepseek_dsa":
        "same MLA backend prediction as above",
    "sglang.srt.observability.req_time_stats.APIServerReqTimeStats":
        "advertises aborts ordered after scheduler dispatch; the fake does not "
        "implement that, and None keeps Dynamo's cancel path fail-closed",
    "sglang.srt.arg_groups.overrides.declare_resolution":
        "records which source resolved each ServerArgs field; None skips the "
        "bookkeeping and nothing reads it back",
    "sglang.srt.multimodal.processors.qwen_vl":
        "Qwen-VL video sizing constants; multimodal is not simulated",
}


def _resolves(module_name: str, symbol) -> bool:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False
    return symbol is None or hasattr(module, symbol)


def _direct_attributes(receiver: str) -> Set[str]:
    """Attributes read straight off ``receiver``, excluding defensive access.

    ``getattr(x, "y", default)`` means Dynamo already tolerates absence, so
    those are not part of the contract; a bare ``x.y`` is.
    """
    direct = re.compile(rf"(?<![\w.]){receiver}\.([a-zA-Z_][a-zA-Z0-9_]*)")
    guarded = re.compile(
        rf"""(?:getattr|hasattr)\(\s*[\w.]*{receiver}\s*,\s*["']([a-zA-Z_][a-zA-Z0-9_]*)["']"""
    )
    found: Set[str] = set()
    optional: Set[str] = set()
    for path in _source_files():
        text = path.read_text()
        found.update(direct.findall(text))
        optional.update(guarded.findall(text))
    return found - optional


@requires_dynamo_source
def test_every_required_import_resolves():
    required, _ = _collect()
    missing = sorted(
        f"{module}.{symbol}" if symbol else module
        for module, symbol in required
        if not _resolves(module, symbol)
    )
    assert not missing, (
        "Dynamo imports sglang names the shim lacks:\n  " + "\n  ".join(missing)
    )


def _name(module: str, symbol) -> str:
    return f"{module}.{symbol}" if symbol else module


@requires_dynamo_source
def test_every_compat_fallback_group_has_a_live_option():
    _, groups = _collect()
    dead = [
        sorted(_name(m, s) for m, s in group)
        for group in groups
        if not any(_resolves(m, s) for m, s in group)
        and not all(_name(m, s) in DELIBERATELY_ABSENT for m, s in group)
    ]
    assert not dead, (
        "Dynamo has try/except import alternatives where the shim satisfies "
        "none:\n  " + "\n  ".join(" | ".join(g) for g in dead)
    )


@pytest.mark.parametrize(
    "fallback,expected_groups",
    [
        ("value = None", 0),
        ("pass", 0),
        ("return None", 0),
        ("from sglang.old import value", 1),
        ("raise", 1),
        ("value = unsupported()", 1),
    ],
)
def test_import_guard_distinguishes_optional_and_required_fallbacks(
    tmp_path, monkeypatch, fallback, expected_groups
):
    source = tmp_path / "compat.py"
    source.write_text(f"try:\n    from sglang.new import value\nexcept ImportError:\n    {fallback}\n")
    monkeypatch.setattr(__import__(__name__, fromlist=["_source_files"]),
                        "_source_files", lambda: iter([source]))
    required, groups = _collect()
    assert required == set()
    assert len(groups) == expected_groups


@requires_dynamo_source
def test_server_args_declares_every_field_dynamo_reads():
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(model_path="stub")
    # Methods and dunders are covered by the symbol test; this is about data.
    # ``parsed_args`` is the Namespace from ServerArgs.add_cli_args, read before
    # ServerArgs exists (args.py), so its fields are the same contract.
    missing = sorted(
        name
        for name in _direct_attributes("server_args") | _direct_attributes("parsed_args")
        if not name.startswith("_") and not hasattr(server_args, name)
    )
    assert not missing, (
        "Dynamo reads ServerArgs fields the shim does not declare "
        "(add them to _SPEC in sglang/srt/server_args.py):\n  " + "\n  ".join(missing)
    )


@requires_dynamo_source
def test_tokenizer_manager_exposes_everything_dynamo_calls():
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    engine = Engine(server_args=ServerArgs(model_path="stub"))
    missing = sorted(
        name
        for name in _direct_attributes("tokenizer_manager")
        if not name.startswith("_") and not hasattr(engine.tokenizer_manager, name)
    )
    assert not missing, (
        "Dynamo calls tokenizer_manager members the shim lacks:\n  " + "\n  ".join(missing)
    )


@requires_dynamo_source
def test_engine_exposes_everything_dynamo_calls():
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import ServerArgs

    engine = Engine(server_args=ServerArgs(model_path="stub"))
    # Dynamo reaches `engine.<attr>` and `self.engine.<attr>` alike.
    wanted = _direct_attributes("engine") | _direct_attributes(r"self\.engine")
    ignore = {"py", "calls", "value", "dangerous", "loop", "my_custom_method"}
    missing = sorted(
        name
        for name in wanted
        if name not in ignore
        and not name.startswith("_")
        and not hasattr(engine, name)
    )
    assert not missing, "Dynamo calls Engine members the shim lacks:\n  " + "\n  ".join(
        missing
    )


@requires_dynamo_source
def test_deliberately_absent_list_is_current():
    """An entry Dynamo no longer imports, or one the shim now provides, is a
    stale decision and should be deleted rather than left to mislead."""
    guarded = {
        _name(m, s)
        for path in _source_files()
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Try) and any(_guards_import_error(h) for h in node.handlers)
        for m, s in _sglang_imports(node)
    }
    stale = sorted(
        name
        for name in DELIBERATELY_ABSENT
        if name not in guarded or _resolves(*name.rsplit(".", 1))
    )
    assert not stale, "Stale DELIBERATELY_ABSENT entries:\n  " + "\n  ".join(stale)


def test_optional_imports_remain_in_absence_inventory(tmp_path, monkeypatch):
    source = tmp_path / "compat.py"
    source.write_text(
        "try:\n    from sglang.optional import value\n"
        "except ImportError:\n    value = None\n"
    )
    module = __import__(__name__, fromlist=["_source_files"])
    monkeypatch.setattr(module, "_source_files", lambda: iter([source]))
    monkeypatch.setattr(module, "DELIBERATELY_ABSENT", {"sglang.optional.value": "optional"})
    test_deliberately_absent_list_is_current()
    source.write_text("")
    with pytest.raises(AssertionError, match="Stale DELIBERATELY_ABSENT"):
        test_deliberately_absent_list_is_current()


@requires_dynamo_source
def test_engine_class_attributes_dynamo_reads_exist():
    """``sgl.Engine.<name>`` read off the class, private names included.

    Class-level reads run at import time (``gateway.GatewayEngine`` binds
    ``sgl.Engine._resolve_routed_dp_rank`` in its class body), so one missing
    attribute stops every worker before it starts -- and the instance check
    above skips private names, which is how that one got through.
    """
    from sglang import Engine

    pattern = re.compile(r"(?<![\w.])(?:sgl|sglang)\.Engine\.([a-zA-Z_][a-zA-Z0-9_]*)")
    wanted = set()
    for path in _source_files():
        wanted.update(pattern.findall(path.read_text()))
    missing = sorted(name for name in wanted if not hasattr(Engine, name))
    assert not missing, "Dynamo reads Engine class attributes the shim lacks:\n  " + (
        "\n  ".join(missing)
    )


@requires_dynamo_source
def test_env_fields_dynamo_reads_have_sglang_types():
    """``envs.X.get()`` must return SGLang's type, not the raw string: Dynamo
    compares them (``envs.SGLANG_MAX_THINK_TOKENS.get() >= 0``)."""
    from sglang.srt.environ import _TYPED

    pattern = re.compile(r"\benvs\.(SGLANG_[A-Z0-9_]+)")
    wanted = set()
    for path in _source_files():
        wanted.update(pattern.findall(path.read_text()))
    missing = sorted(wanted - set(_TYPED))
    assert not missing, (
        "Dynamo reads SGLang env fields the shim does not type "
        "(add them to _TYPED in sglang/srt/environ.py):\n  " + "\n  ".join(missing)
    )


@requires_dynamo_source
def test_dynamo_sees_cache_salt_support():
    """Dynamo only sends a salt after checking both the native request type
    and the async_generate signature; either missing makes it refuse salted
    requests outright."""
    from dynamo.sglang._compat import cache_salt_kwargs
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.server_args import ServerArgs

    assert "cache_salt" in GenerateReqInput.__dataclass_fields__
    engine = Engine(server_args=ServerArgs(model_path="stub"))
    assert cache_salt_kwargs(engine, "tenant-a") == {"cache_salt": "tenant-a"}
