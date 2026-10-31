#!/usr/bin/env bash
# Bootstrap a native macOS dev environment for Dynamo + the fake SGLang engine.
#
# Dynamo's Python bindings are a PyO3 extension with no macOS wheel, so they are
# built from source. Real sglang is never installed -- fake-sglang provides an
# importable `sglang` package instead.
#
# Python 3.10 is deliberate: ai-dynamo's `aisimulate` dependency is marked
# `python_version >= '3.11'` and ships only an sdist, so 3.10 skips a Rust build
# we do not need.
#
# SAFE TO RE-RUN. Every step checks before it acts; a second run reuses the
# existing toolchain and virtualenv and reinstalls nothing that is already
# satisfied. The only work a re-run repeats is the Rust binding build
# (~15s incremental against cargo's cache, vs ~2.5min cold), because that is
# the one step whose output can be stale after a change under dynamo/lib.
#
# Env overrides:
#   FORCE_RECREATE=1  delete and rebuild the virtualenv from scratch
#   SKIP_BINDINGS=1   skip the Rust binding build if dynamo._core already imports
#   PY_VERSION=3.12   use a different Python (see the aisimulate note above)
set -euo pipefail

FAKE_SGLANG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOS="${REPOS:-$(dirname "$FAKE_SGLANG")}"
DYNAMO="$REPOS/dynamo"
VENV="$REPOS/.venv"
PY_VERSION="${PY_VERSION:-3.10}"

step() { printf '\n=== %s\n' "$*"; }
skip() { printf '    (already done) %s\n' "$*"; }

step "1/7 system libraries"
missing_pkgs=()
command -v cmake  >/dev/null || missing_pkgs+=(cmake)
command -v protoc >/dev/null || missing_pkgs+=(protobuf)
if [ ${#missing_pkgs[@]} -gt 0 ]; then
  brew install "${missing_pkgs[@]}"
else
  skip "cmake and protoc are on PATH"
fi

step "2/7 rust toolchain"
if command -v cargo >/dev/null || [ -x "$HOME/.cargo/bin/cargo" ]; then
  skip "cargo is installed"
else
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
fi
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version

step "3/7 uv"
if command -v uv >/dev/null || [ -x "$HOME/.local/bin/uv" ]; then
  skip "uv is installed"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

step "4/7 virtualenv ($VENV, python $PY_VERSION)"
if [ "${FORCE_RECREATE:-}" = "1" ] && [ -d "$VENV" ]; then
  echo "FORCE_RECREATE=1: removing $VENV"
  rm -rf "$VENV"
fi
if [ -x "$VENV/bin/python" ]; then
  # `uv venv` refuses to touch an existing environment and exits non-zero, so
  # under `set -e` an unguarded call would abort the whole script on re-run.
  # Reuse it instead -- but only if it is the interpreter we expect, because a
  # venv left at this path by another project would fail much later and much
  # more confusingly.
  existing="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [ "$existing" != "$PY_VERSION" ]; then
    echo "error: $VENV is Python $existing, expected $PY_VERSION." >&2
    echo "Re-run with FORCE_RECREATE=1 to rebuild it, or PY_VERSION=$existing to accept it." >&2
    exit 1
  fi
  skip "virtualenv exists (Python $existing)"
else
  uv venv --python "$PY_VERSION" "$VENV"
fi
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
uv pip install pip maturin      # uv no-ops when already satisfied

step "5/7 build Dynamo python bindings (ai-dynamo-runtime)"
if [ "${SKIP_BINDINGS:-}" = "1" ] && python -c "import dynamo._core" 2>/dev/null; then
  skip "dynamo._core imports and SKIP_BINDINGS=1"
else
  # Rebuilt on every run by design: cargo's cache makes this ~15s once warm,
  # and it is the only way a change under dynamo/lib reaches the wheel.
  # The bindings crate pins its own toolchain via dynamo/rust-toolchain.toml.
  # If the default feature set fails to link, retry without it: the default
  # pulls Linux/CUDA-oriented NIXL, NUMA and O_DIRECT code.
  cd "$DYNAMO/lib/bindings/python"
  if ! maturin develop --uv; then
    echo "!! default-feature build failed; retrying with --no-default-features"
    maturin develop --uv --no-default-features
  fi
fi

step "6/7 install ai-dynamo (frontend + sglang backend) and fake-sglang"
cd "$DYNAMO"
uv pip install -e .            # NOT .[sglang] -- that extra pulls CUDA wheels
uv pip install -e "$FAKE_SGLANG[dev]"
# Imported at module scope by dynamo.sglang.request_handlers.llm.decode_handler.
uv pip install torch pillow
# Not declared by ai-dynamo itself, but imported by the SGLang backend.
uv pip install blake3

step "7/7 verify"
# `cmd && echo OK` is exempt from `set -e` -- a failing check would print its
# traceback and the script would still announce "Done". Count failures and
# exit non-zero instead.
verify_failed=0
verify() {
  local label="$1"; shift
  if "$@" >/dev/null; then
    echo "$label OK"
  else
    echo "!! $label FAILED (see the error above)" >&2
    verify_failed=1
  fi
}
python -c "import sglang; assert sglang.__is_fake_sglang__; print('sglang shim ->', sglang.__file__)"
verify pytest          python -m pytest --version
verify ruff            python -m ruff --version
verify dynamo.frontend python -m dynamo.frontend --help
verify dynamo.sglang   python -m dynamo.sglang --help
if [ "$verify_failed" -ne 0 ]; then
  cat >&2 <<'MSG'

Verification failed. The usual causes, after `git pull` in ../dynamo:
  * dynamo.frontend: the Rust bindings were rebuilt but still fail to load --
    re-run this script and check step 5's output.
  * dynamo.sglang: Dynamo started using SGLang APIs the shim lacks. Run
      pytest tests/test_shim_contract.py
    for the exact list of missing names.
MSG
  exit 1
fi

cat <<MSG

Done. Activate with:
  export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"
  source $VENV/bin/activate

Model metadata (config + tokenizer) is already in $FAKE_SGLANG/models.
Next: $FAKE_SGLANG/launch/agg.sh
MSG
