"""Simulation configuration, derived from SGLang ``ServerArgs``."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

from fakeengine.topology import Topology, load as load_topology

logger = logging.getLogger(__name__)

_BANNER_SHOWN = False


@dataclass
class SimConfig:
    model_path: str
    served_model_name: str
    disaggregation_mode: str        # null | prefill | decode
    vocab_size: int
    context_len: int
    kv_bytes_per_token: int

    block_size: int
    num_kv_blocks: int
    max_running_requests: int
    max_batched_tokens: int

    prefill_ms_per_1k: float
    prefill_overhead_ms: float
    itl_ms: float
    itl_ms_per_running_req: float
    kv_bandwidth_gb_s: float
    kv_transfer_overhead_ms: float
    speedup_ratio: float

    bootstrap_host: Optional[str]
    bootstrap_port: int
    bootstrap_poll_interval_s: float
    bootstrap_timeout_s: float

    prefix_caching: bool = True
    location: Optional[str] = None
    topology: Optional[Topology] = None
    kv_generation_ms_per_1k: float = 0.0

    @property
    def total_kv_tokens(self) -> int:
        return self.num_kv_blocks * self.block_size

    @classmethod
    def from_server_args(cls, server_args: Any) -> "SimConfig":
        model_config = server_args.get_model_config()
        # Must equal Dynamo's kv_event_block_size(), which is what it registers
        # as the model's kv_cache_block_size and what the frontend uses to
        # chunk prompts before hashing. A mismatch means every published block
        # is silently dropped on the Rust side.
        page_size = _positive_int(server_args.page_size, "page_size")
        dcp_size = _positive_int(server_args.dcp_size, "dcp_size")
        block_size = page_size * dcp_size

        # A KV pool large enough that a laptop-scale smoke test never evicts,
        # but still finite so pressure and preemption are reachable on purpose.
        num_blocks = _positive_int(server_args.fake_num_kv_blocks, "fake_num_kv_blocks")
        if server_args.max_total_tokens is not None:
            total_tokens = _positive_int(server_args.max_total_tokens, "max_total_tokens")
            if total_tokens % block_size:
                raise ValueError("max_total_tokens must be a multiple of page_size * dcp_size")
            num_blocks = total_tokens // block_size

        kv_generation_ms = float(server_args.fake_kv_generation_ms_per_1k_tokens)
        if not math.isfinite(kv_generation_ms) or kv_generation_ms < 0:
            raise ValueError("fake_kv_generation_ms_per_1k_tokens must be finite and nonnegative")

        max_batched = int(
            server_args.max_prefill_tokens
            or server_args.chunked_prefill_size
            or 8192
        )

        location = getattr(server_args, "fake_location", None) or None
        topology = load_topology(getattr(server_args, "fake_topology", None))
        if topology is not None and location is not None:
            topology.parse_location(location)  # fail at startup, not mid-transfer

        return cls(
            model_path=server_args.model_path,
            served_model_name=server_args.served_model_name or server_args.model_path,
            disaggregation_mode=server_args.disaggregation_mode or "null",
            vocab_size=model_config.vocab_size,
            context_len=model_config.context_len,
            kv_bytes_per_token=model_config.kv_bytes_per_token(),
            block_size=block_size,
            num_kv_blocks=num_blocks,
            max_running_requests=int(server_args.max_running_requests or 256),
            max_batched_tokens=max_batched,
            prefill_ms_per_1k=float(server_args.fake_prefill_ms_per_1k_tokens),
            prefill_overhead_ms=float(server_args.fake_prefill_overhead_ms),
            kv_generation_ms_per_1k=kv_generation_ms,
            itl_ms=float(server_args.fake_itl_ms),
            itl_ms_per_running_req=float(server_args.fake_itl_ms_per_running_req),
            kv_bandwidth_gb_s=float(server_args.fake_kv_bandwidth_gb_s),
            kv_transfer_overhead_ms=float(server_args.fake_kv_transfer_overhead_ms),
            speedup_ratio=max(float(server_args.fake_speedup_ratio), 1e-6),
            bootstrap_host=None,
            bootstrap_port=int(server_args.disaggregation_bootstrap_port or 8998),
            bootstrap_poll_interval_s=float(
                server_args.fake_bootstrap_poll_interval_ms
            )
            / 1000.0,
            bootstrap_timeout_s=float(server_args.fake_bootstrap_timeout_s),
            prefix_caching=not bool(getattr(server_args, "disable_radix_cache", False)),
            location=location,
            topology=topology,
        )


def announce(config: SimConfig) -> None:
    """Say loudly that this is not SGLang.

    A fake engine that is quiet about being fake is a benchmark waiting to be
    misread, so this prints once per process regardless of log level.
    """
    global _BANNER_SHOWN
    if _BANNER_SHOWN or os.environ.get("FAKE_SGLANG_QUIET") == "1":
        return
    _BANNER_SHOWN = True
    logger.warning(
        "fake-sglang: SIMULATED engine -- no model weights, no real inference. "
        "mode=%s model=%s vocab=%d kv=%d blocks x %d tok "
        "(%.1f KiB/token) prefill=%.1fms/1k kv-generation=%.1fms/1k "
        "itl=%.1fms xfer=%s speedup=%.2fx "
        "location=%s",
        config.disaggregation_mode,
        config.served_model_name,
        config.vocab_size,
        config.num_kv_blocks,
        config.block_size,
        config.kv_bytes_per_token / 1024,
        config.prefill_ms_per_1k,
        config.kv_generation_ms_per_1k,
        config.itl_ms,
        "per-link topology"
        if config.topology is not None
        else "%.0fGB/s" % config.kv_bandwidth_gb_s,
        config.speedup_ratio,
        config.location or "-",
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
