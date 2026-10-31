"""SGLang-shaped ``ServerArgs`` for the fake engine.

Only the flags and fields that Dynamo's SGLang backend
(``dynamo/components/src/dynamo/sglang``) actually reads are declared. The
authoritative list is derivable from the backend itself:

    grep -rhoE "server_args\\.[a-zA-Z_][a-zA-Z0-9_]*" components/src/dynamo/sglang
    grep -rhoE "parsed_args\\.[a-zA-Z_][a-zA-Z0-9_]*" components/src/dynamo/sglang/args.py

``tests/test_shim_contract.py`` re-runs that derivation and fails when Dynamo
starts reading a field we do not declare.

One table drives both the dataclass defaults and ``add_cli_args`` so the two
cannot drift.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Iterable, Optional

FAKE_ENGINE_MARKER = "fake-sglang"

# kind: "str" | "int" | "float" | "flag" (store_true) | "noflag" (store_false, default True)
# (dest, kind, default, choices)
_SPEC: tuple[tuple[str, str, Any, Optional[Iterable[str]]], ...] = (
    # --- model / tokenizer ---
    ("model_path", "str", None, None),
    ("served_model_name", "str", None, None),
    ("tokenizer_path", "str", None, None),
    ("tokenizer_mode", "str", "auto", None),
    ("chat_template", "str", None, None),
    ("revision", "str", None, None),
    ("dtype", "str", "auto", None),
    ("kv_cache_dtype", "str", "fp8_e4m3", ("auto", "fp8_e4m3", "fp8_e5m2")),
    ("context_length", "int", 1_048_576, None),
    ("load_format", "str", "auto", ("auto", "dummy", "safetensors", "pt", "remote_instance")),
    # Dynamo routes ModelExpress weight loading on these two; the fake loads no
    # weights, so they are accepted and recorded only.
    ("remote_instance_weight_loader_backend", "str", "nccl", ("transfer_engine", "nccl", "modelexpress")),
    ("trust_remote_code", "flag", False, None),
    ("skip_tokenizer_init", "flag", False, None),
    ("is_embedding", "flag", False, None),
    ("encoder_only", "flag", False, None),
    ("enable_multimodal", "flag", False, None),
    ("device", "str", "cpu", None),
    ("random_seed", "int", 0, None),
    # --- serving surface ---
    ("host", "str", "127.0.0.1", None),
    ("port", "int", 30000, None),
    ("log_level", "str", "info", None),
    ("log_level_http", "str", None, None),
    ("config", "str", None, None),
    # --- parallelism / topology ---
    ("tp_size", "int", 1, None),
    ("dp_size", "int", 1, None),
    # Dynamo raises this to its --gateway-workers count; gateway mode itself
    # is unsupported (see managers/multi_tokenizer_mixin.py).
    ("tokenizer_worker_num", "int", 1, None),
    ("pp_size", "int", 1, None),
    ("ep_size", "int", 1, None),
    ("dcp_size", "int", 1, None),
    ("nnodes", "int", 1, None),
    ("node_rank", "int", 0, None),
    ("dist_init_addr", "str", None, None),
    ("dist_timeout", "int", None, None),
    ("nccl_port", "int", None, None),
    ("base_gpu_id", "int", 0, None),
    ("gpu_id_step", "int", 1, None),
    ("enable_dp_attention", "flag", False, None),
    ("elastic_ep_backend", "str", None, None),
    # --- scheduling / memory ---
    ("max_running_requests", "int", None, None),
    ("max_total_tokens", "int", None, None),
    ("max_prefill_tokens", "int", 8192, None),
    ("chunked_prefill_size", "int", 8192, None),
    ("mem_fraction_static", "float", 0.9, None),
    ("page_size", "int", 256, None),
    ("schedule_policy", "str", "fcfs", None),
    ("schedule_conservativeness", "float", 1.0, None),
    ("stream_interval", "int", 1, None),
    ("schedule_low_priority_values_first", "flag", False, None),
    # --- attention / speculative / diffusion ---
    ("attention_backend", "str", None, None),
    ("prefill_attention_backend", "str", None, None),
    ("decode_attention_backend", "str", None, None),
    ("sampling_backend", "str", None, None),
    ("speculative_algorithm", "str", None, None),
    ("dllm_algorithm", "str", None, None),
    ("dllm_algorithm_config", "str", None, None),
    # --- disaggregation ---
    ("disaggregation_mode", "str", "null", ("null", "prefill", "decode", "encode")),
    (
        "disaggregation_transfer_backend",
        "str",
        "fake",
        ("mooncake", "nixl", "ascend", "mori", "fake"),
    ),
    ("disaggregation_bootstrap_port", "int", 8998, None),
    ("disaggregation_ib_device", "str", None, None),
    ("disaggregation_decode_polling_interval", "int", 1, None),
    ("enable_kv_return", "flag", False, None),
    # --- parsers ---
    ("reasoning_parser", "str", None, None),
    ("tool_call_parser", "str", None, None),
    # --- observability / misc ---
    ("enable_metrics", "flag", False, None),
    ("enable_trace", "flag", False, None),
    ("enable_memory_saver", "flag", False, None),
    ("enable_forward_pass_metrics", "flag", False, None),
    ("forward_pass_metrics_ipc_name", "str", None, None),
    ("forward_pass_metrics_worker_id", "str", None, None),
    ("kv_events_config", "str", None, None),
    # Honored: no prefix reuse and no BlockStored events, as in SGLang.
    ("disable_radix_cache", "flag", False, None),
    ("incremental_streaming_output", "flag", False, None),
    ("allow_auto_truncate", "flag", False, None),
    ("constrained_json_whitespace_pattern", "str", None, None),
    ("debug_tensor_dump_input_file", "str", None, None),
    ("hicache_storage_backend", "str", None, None),
    # --- fake-engine simulation knobs (not present in real SGLang) ---
    ("fake_prefill_ms_per_1k_tokens", "float", 55.0, None),
    ("fake_prefill_overhead_ms", "float", 4.0, None),
    ("fake_kv_generation_ms_per_1k_tokens", "float", 0.0, None),
    ("fake_itl_ms", "float", 9.0, None),
    ("fake_itl_ms_per_running_req", "float", 0.35, None),
    ("fake_kv_bandwidth_gb_s", "float", 64.0, None),
    ("fake_kv_transfer_overhead_ms", "float", 1.0, None),
    ("fake_kv_return_max_tokens", "int", 4096, None),
    ("fake_kv_return_deadline_ms", "float", None, None),
    ("fake_kv_return_bytes_per_token", "int", None, None),
    ("fake_kv_return_bandwidth_gb_s", "float", None, None),
    ("fake_kv_return_overhead_ms", "float", None, None),
    ("fake_kv_return_retention_max_leases", "int", 32, None),
    ("fake_kv_return_retention_max_blocks", "int", None, None),
    ("fake_kv_return_retention_max_bytes", "int", None, None),
    ("fake_kv_return_retention_timeout_ms", "float", None, None),
    ("fake_speedup_ratio", "float", 1.0, None),
    ("fake_num_kv_blocks", "int", 15_509, None),
    ("fake_bootstrap_poll_interval_ms", "float", 5.0, None),
    ("fake_bootstrap_timeout_s", "float", 30.0, None),
    # Natural output length: unset = every request runs to max_new_tokens.
    ("fake_output_len_median", "float", None, None),
    ("fake_output_len_sigma", "float", 1.0, None),
    # Network placement: this worker's location path and the deployment-wide link
    # table (JSON, see fakeengine/topology.py). Both unset = one global link.
    ("fake_location", "str", None, None),
    ("fake_topology", "str", None, None),
)

_DEFAULTS = {dest: default for dest, _kind, default, _choices in _SPEC}


def _flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


class ServerArgs:
    """A permissive stand-in for ``sglang.srt.server_args.ServerArgs``.

    Unknown keyword arguments are accepted and stored: Dynamo and its tests
    construct this with a partial field set, and SGLang's real ServerArgs
    tolerates far more fields than we model.
    """

    def __init__(self, **kwargs: Any) -> None:
        for dest, default in _DEFAULTS.items():
            setattr(self, dest, default)
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if not self.served_model_name:
            self.served_model_name = self.model_path
        self._model_config: Any = None

    # -- construction ------------------------------------------------------

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        for dest, kind, default, choices in _SPEC:
            name = _flag(dest)
            if kind == "flag":
                parser.add_argument(name, dest=dest, action="store_true", default=default)
                continue
            if kind == "noflag":
                parser.add_argument(
                    "--no-" + dest.replace("_", "-"),
                    dest=dest,
                    action="store_false",
                    default=default,
                )
                continue
            kwargs: dict[str, Any] = {
                "dest": dest,
                "default": default,
                "type": {"str": str, "int": int, "float": float}[kind],
            }
            if choices:
                kwargs["choices"] = list(choices)
            if dest == "model_path":
                kwargs["required"] = True
            parser.add_argument(name, **kwargs)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "ServerArgs":
        return cls(**{k: v for k, v in vars(args).items() if not k.startswith("_")})

    # -- SGLang accessors Dynamo calls -------------------------------------

    def get_model_config(self) -> "ModelConfig":
        """Resolve the HF config. SGLang exposes this; Dynamo's ``_compat``
        prefers it over the newer module-level ``model_config_of()``."""
        if self._model_config is None:
            self._model_config = ModelConfig(self)
        return self._model_config

    def use_mla_backend(self) -> bool:
        return False

    # -- helpers used by the fake engine -----------------------------------

    @property
    def kv_events_publisher(self) -> Optional[str]:
        if not self.kv_events_config:
            return None
        try:
            return json.loads(self.kv_events_config).get("publisher")
        except json.JSONDecodeError:
            return None

    def __repr__(self) -> str:
        return (
            f"ServerArgs(model_path={self.model_path!r}, "
            f"disaggregation_mode={self.disaggregation_mode!r}, "
            f"tp_size={self.tp_size}, dp_size={self.dp_size})"
        )


class ModelConfig:
    """The subset of SGLang's ModelConfig that Dynamo reads.

    ``handler_base.py`` reads ``hf_text_config.vocab_size``; ``args.py`` reads
    ``is_multimodal``; ``register.py`` reads ``model_path``. The fake engine
    additionally uses the layer/head geometry to size simulated KV transfers.
    """

    def __init__(self, server_args: ServerArgs) -> None:
        self.model_path = server_args.model_path
        raw = _load_hf_config(server_args.model_path)
        self.hf_config = _Namespace(raw)
        text_cfg = raw.get("text_config") or raw
        self.hf_text_config = _Namespace(text_cfg)

        self.vocab_size = int(text_cfg.get("vocab_size", 32000))
        self.num_hidden_layers = int(text_cfg.get("num_hidden_layers", 24))
        self.hidden_size = int(text_cfg.get("hidden_size", 2048))
        num_heads = int(text_cfg.get("num_attention_heads", 16))
        self.num_attention_heads = num_heads
        self.num_key_value_heads = int(text_cfg.get("num_key_value_heads", num_heads))
        self.head_dim = int(text_cfg.get("head_dim", self.hidden_size // max(num_heads, 1)))
        self.context_len = int(
            server_args.context_length
            or text_cfg.get("max_position_embeddings", 8192)
        )
        self.is_multimodal = bool(
            raw.get("vision_config") or raw.get("audio_config") or "text_config" in raw
        )
        self.is_generation = not server_args.is_embedding

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """KV bytes one token occupies across all layers.

        Same formula the Dynamo mocker uses
        (``lib/mocker/src/common/protocols.rs``): layers * 2 (K and V) *
        kv_heads * head_dim * dtype width.
        """
        return (
            self.num_hidden_layers
            * 2
            * self.num_key_value_heads
            * self.head_dim
            * dtype_bytes
        )


class _Namespace:
    """Attribute view over a dict, with ``getattr(..., default)`` semantics."""

    def __init__(self, data: dict) -> None:
        self.__dict__.update(data)

    def __repr__(self) -> str:
        return f"_Namespace({sorted(self.__dict__)})"


def _load_hf_config(model_path: Optional[str]) -> dict:
    """Read ``config.json`` for a local dir or a cached HF repo id.

    Dynamo calls ``fetch_model(model_path)`` before building ServerArgs, which
    places config.json and the tokenizer in the HF cache without weights, so a
    repo id resolves here even though nothing was ever downloaded in full.
    """
    if not model_path:
        return {}

    local = os.path.join(os.path.expanduser(model_path), "config.json")
    if os.path.isfile(local):
        with open(local) as handle:
            return json.load(handle)

    try:
        from transformers import AutoConfig

        # Cache only. Dynamo has already fetched config.json by now, so going
        # to the network can only find nothing -- and huggingface_hub retries
        # with backoff, turning an unknown repo id into a minutes-long stall
        # inside Engine() instead of an immediate empty config.
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True
        )
        return config.to_dict()
    except Exception:  # noqa: BLE001 - metadata is best-effort for a fake engine
        return {}


class PortArgs:
    """IPC endpoints SGLang's processes talk over.

    The fake engine has no subprocesses, but Dynamo's publisher binds a PULL
    socket on ``metrics_ipc_name`` regardless, so the name must be real and
    unique per worker -- two workers sharing one path would steal each other's
    metrics.
    """

    def __init__(self, server_args: Optional[ServerArgs] = None) -> None:
        token = f"{os.getpid()}-{os.urandom(4).hex()}"
        base = os.environ.get("SGLANG_IPC_DIR", "/tmp")
        self.tokenizer_ipc_name = f"ipc://{base}/fake-sglang-tokenizer-{token}"
        self.scheduler_input_ipc_name = f"ipc://{base}/fake-sglang-scheduler-{token}"
        self.detokenizer_ipc_name = f"ipc://{base}/fake-sglang-detokenizer-{token}"
        self.rpc_ipc_name = f"ipc://{base}/fake-sglang-rpc-{token}"
        self.metrics_ipc_name = f"ipc://{base}/fake-sglang-metrics-{token}"
        self.nccl_port = getattr(server_args, "nccl_port", None) if server_args else None

    @classmethod
    def init_new(cls, server_args: ServerArgs, *args: Any, **kwargs: Any) -> "PortArgs":
        return cls(server_args)


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """Mirror of SGLang's module-level helper; used by the fake launcher."""
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return ServerArgs.from_cli_args(parser.parse_args(argv))
