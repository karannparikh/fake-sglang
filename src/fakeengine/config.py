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
KV_RETURN_PROTOCOL_VERSION = 1


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
    stream_interval: int

    prefill_ms_per_1k: float
    prefill_overhead_ms: float
    itl_ms: float
    itl_ms_per_running_req: float
    kv_bandwidth_gb_s: float
    kv_transfer_overhead_ms: float
    enable_kv_return: bool  # Advertise reverse-KV return on eligible rooms.
    kv_return_protocol_version: int  # Version of the return-ticket schema.
    kv_return_max_tokens: int  # Maximum generated tokens a ticket accepts.
    kv_return_deadline_ms: float  # Return-ticket deadline in milliseconds.
    kv_return_bytes_per_token: int  # Reverse payload bytes per generated token.
    kv_return_bandwidth_gb_s: float  # Simulated reverse-transfer bandwidth.
    kv_return_overhead_ms: float  # Fixed reverse-transfer cost in milliseconds.
    kv_return_retention_max_leases: int  # Maximum active decode KV leases.
    kv_return_retention_max_blocks: int  # Maximum retained decode KV blocks.
    kv_return_retention_max_bytes: int  # Maximum retained decode KV bytes.
    kv_return_retention_timeout_ms: float  # Decode KV lease lifetime.
    speedup_ratio: float

    bootstrap_host: Optional[str]
    bootstrap_port: int
    bootstrap_poll_interval_s: float
    bootstrap_timeout_s: float

    prefix_caching: bool = True
    location: Optional[str] = None
    topology: Optional[Topology] = None
    kv_generation_ms_per_1k: float = 0.0
    output_len_median: Optional[float] = None
    output_len_sigma: float = 1.0

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
        dp_size = _positive_int(server_args.dp_size, "dp_size")
        max_running = server_args.max_running_requests
        if max_running is not None:
            max_running = _positive_int(max_running, "max_running_requests")
            if max_running % dp_size:
                raise ValueError("max_running_requests must be divisible by dp_size")

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

        output_len_median = getattr(server_args, "fake_output_len_median", None)
        if output_len_median is not None:
            output_len_median = float(output_len_median)
            if not math.isfinite(output_len_median) or output_len_median < 1:
                raise ValueError("fake_output_len_median must be finite and at least 1")
        output_len_sigma = float(getattr(server_args, "fake_output_len_sigma", 1.0))
        if not math.isfinite(output_len_sigma) or output_len_sigma < 0:
            raise ValueError("fake_output_len_sigma must be finite and nonnegative")

        max_batched = int(
            server_args.max_prefill_tokens
            or server_args.chunked_prefill_size
            or 8192
        )

        location = getattr(server_args, "fake_location", None) or None
        topology = load_topology(getattr(server_args, "fake_topology", None))
        if topology is not None and location is not None:
            topology.parse_location(location)  # fail at startup, not mid-transfer

        kv_bytes_per_token = model_config.kv_bytes_per_token(
            dtype_bytes=1 if server_args.kv_cache_dtype.startswith("fp8") else 2
        )
        kv_bandwidth_gb_s = _positive_finite_float(
            server_args.fake_kv_bandwidth_gb_s,
            "fake_kv_bandwidth_gb_s",
        )
        kv_transfer_overhead_ms = _nonnegative_finite_float(
            server_args.fake_kv_transfer_overhead_ms,
            "fake_kv_transfer_overhead_ms",
        )
        bootstrap_timeout_s = _nonnegative_finite_float(
            server_args.fake_bootstrap_timeout_s,
            "fake_bootstrap_timeout_s",
        )

        enable_kv_return = bool(getattr(server_args, "enable_kv_return", False))
        disaggregation_mode = server_args.disaggregation_mode or "null"
        prefix_caching = not bool(getattr(server_args, "disable_radix_cache", False))
        # Mirror ``KVSender._return_ticket``: the reverse path only produces a
        # ticket on prefill workers with radix caching enabled. Any other
        # combination omits the capability regardless of timeouts.
        return_eligible = (
            enable_kv_return
            and disaggregation_mode == "prefill"
            and prefix_caching
        )

        return_bytes = getattr(server_args, "fake_kv_return_bytes_per_token", None)
        if return_bytes is None:
            return_bytes = kv_bytes_per_token
        else:
            return_bytes = _positive_int(
                return_bytes,
                "fake_kv_return_bytes_per_token",
            )

        return_bandwidth = getattr(server_args, "fake_kv_return_bandwidth_gb_s", None)
        if return_bandwidth is None:
            return_bandwidth = kv_bandwidth_gb_s
        else:
            return_bandwidth = _positive_finite_float(
                return_bandwidth,
                "fake_kv_return_bandwidth_gb_s",
            )

        return_overhead = getattr(server_args, "fake_kv_return_overhead_ms", None)
        if return_overhead is None:
            return_overhead = kv_transfer_overhead_ms
        else:
            return_overhead = _nonnegative_finite_float(
                return_overhead,
                "fake_kv_return_overhead_ms",
            )

        return_deadline = getattr(server_args, "fake_kv_return_deadline_ms", None)
        if return_deadline is None:
            return_deadline = bootstrap_timeout_s * 1000.0
        else:
            return_deadline = _positive_finite_float(
                return_deadline,
                "fake_kv_return_deadline_ms",
            )
        # A zero bootstrap timeout is a legitimate forward-path configuration,
        # but it collapses the fallback reverse-return deadline to zero. Only
        # an eligible reverse path actually uses that deadline, so validate the
        # effective value there rather than forbidding a zero forward timeout.
        if return_eligible and not (
            math.isfinite(return_deadline) and return_deadline > 0
        ):
            raise ValueError(
                "enable_kv_return requires a positive reverse-return deadline; "
                "set fake_kv_return_deadline_ms or a positive "
                "fake_bootstrap_timeout_s"
            )

        retention_max_leases = _positive_int(
            getattr(server_args, "fake_kv_return_retention_max_leases", 32),
            "fake_kv_return_retention_max_leases",
        )
        retention_max_blocks = getattr(
            server_args, "fake_kv_return_retention_max_blocks", None,
        )
        if retention_max_blocks is None:
            retention_max_blocks = num_blocks
        else:
            retention_max_blocks = _positive_int(
                retention_max_blocks,
                "fake_kv_return_retention_max_blocks",
            )
        retention_max_bytes = getattr(
            server_args, "fake_kv_return_retention_max_bytes", None,
        )
        if retention_max_bytes is None:
            retention_max_bytes = (
                retention_max_blocks * block_size * kv_bytes_per_token
            )
        else:
            retention_max_bytes = _positive_int(
                retention_max_bytes,
                "fake_kv_return_retention_max_bytes",
            )
        retention_timeout = getattr(
            server_args, "fake_kv_return_retention_timeout_ms", None,
        )
        if retention_timeout is None:
            retention_timeout = return_deadline
        else:
            retention_timeout = _positive_finite_float(
                retention_timeout,
                "fake_kv_return_retention_timeout_ms",
            )

        return cls(
            model_path=server_args.model_path,
            served_model_name=server_args.served_model_name or server_args.model_path,
            disaggregation_mode=disaggregation_mode,
            vocab_size=model_config.vocab_size,
            context_len=model_config.context_len,
            kv_bytes_per_token=kv_bytes_per_token,
            block_size=block_size,
            num_kv_blocks=num_blocks,
            max_running_requests=max_running // dp_size if max_running is not None else 32,
            max_batched_tokens=max_batched,
            stream_interval=_positive_int(
                server_args.stream_interval,
                "stream_interval",
            ),
            prefill_ms_per_1k=float(server_args.fake_prefill_ms_per_1k_tokens),
            prefill_overhead_ms=float(server_args.fake_prefill_overhead_ms),
            kv_generation_ms_per_1k=kv_generation_ms,
            itl_ms=float(server_args.fake_itl_ms),
            itl_ms_per_running_req=float(server_args.fake_itl_ms_per_running_req),
            kv_bandwidth_gb_s=kv_bandwidth_gb_s,
            kv_transfer_overhead_ms=kv_transfer_overhead_ms,
            enable_kv_return=enable_kv_return,
            kv_return_protocol_version=KV_RETURN_PROTOCOL_VERSION,
            kv_return_max_tokens=_positive_int(
                getattr(server_args, "fake_kv_return_max_tokens", 4096),
                "fake_kv_return_max_tokens",
            ),
            kv_return_deadline_ms=return_deadline,
            kv_return_bytes_per_token=return_bytes,
            kv_return_bandwidth_gb_s=return_bandwidth,
            kv_return_overhead_ms=return_overhead,
            kv_return_retention_max_leases=retention_max_leases,
            kv_return_retention_max_blocks=retention_max_blocks,
            kv_return_retention_max_bytes=retention_max_bytes,
            kv_return_retention_timeout_ms=retention_timeout,
            speedup_ratio=max(float(server_args.fake_speedup_ratio), 1e-6),
            bootstrap_host=None,
            bootstrap_port=int(server_args.disaggregation_bootstrap_port or 8998),
            bootstrap_poll_interval_s=float(
                server_args.fake_bootstrap_poll_interval_ms
            )
            / 1000.0,
            bootstrap_timeout_s=bootstrap_timeout_s,
            prefix_caching=prefix_caching,
            location=location,
            topology=topology,
            output_len_median=output_len_median,
            output_len_sigma=output_len_sigma,
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
        "output-len=%s location=%s",
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
        "max_new_tokens"
        if config.output_len_median is None
        else "lognormal(median=%g, sigma=%g)"
        % (config.output_len_median, config.output_len_sigma),
        config.location or "-",
    )


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _nonnegative_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number
