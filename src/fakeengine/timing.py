"""The latency model.

Deliberately simple and analytic rather than interpolated from measured
profiles: the point is that TTFT and ITL move the way a real disaggregated
deployment's do (prefill scales with input length, decode degrades with batch
size, KV transfer scales with payload) so routing and scheduling decisions
under test see plausible gradients. Dynamo's own mocker offers an interpolated
model (``lib/mocker/src/common/perf_model.rs``) if higher fidelity is ever
needed; the knobs here are named to match its vocabulary.
"""

from __future__ import annotations

import asyncio

from typing import Optional

from fakeengine.config import SimConfig
from fakeengine.topology import Link

_BYTES_PER_GB = 1e9


class TimingModel:
    def __init__(self, config: SimConfig) -> None:
        self.config = config

    def prefill_ms(self, num_tokens: int) -> float:
        cfg = self.config
        return (
            cfg.prefill_overhead_ms
            + (num_tokens / 1000.0) * cfg.prefill_ms_per_1k
            + self.kv_generation_ms(num_tokens)
        )

    def kv_generation_ms(self, uncached_tokens: int) -> float:
        return uncached_tokens * self.config.kv_generation_ms_per_1k / 1000.0

    def decode_step_ms(self, running_requests: int) -> float:
        """One decode step for the whole batch.

        Batch decode is one forward pass, so the cost grows sub-linearly with
        batch size rather than once per request -- which is exactly why
        continuous batching pays off, and why a per-request sleep would model
        the wrong thing.
        """
        cfg = self.config
        extra = max(running_requests - 1, 0) * cfg.itl_ms_per_running_req
        return cfg.itl_ms + extra

    def transfer_ms(self, num_tokens: int, link: Optional[Link] = None) -> float:
        """KV payload over ``link``, or over the global link when ``None``."""
        cfg = self.config
        bandwidth = link.bandwidth_gb_s if link else cfg.kv_bandwidth_gb_s
        overhead = link.latency_ms if link else cfg.kv_transfer_overhead_ms
        payload_bytes = num_tokens * cfg.kv_bytes_per_token
        seconds = payload_bytes / (bandwidth * _BYTES_PER_GB)
        return overhead + seconds * 1000.0

    async def sleep_ms(self, milliseconds: float) -> None:
        if milliseconds <= 0:
            return
        await asyncio.sleep(milliseconds / 1000.0 / self.config.speedup_ratio)
