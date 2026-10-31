"""The latency model.

Prefill remains a simple analytic fit while decode uses engine-only P50
anchors measured on disaggregated prefill/decode workers. The goal is
deterministic, production-shaped TTFT and ITL rather than sampled tail latency:
prefill interruptions emerge from the scheduler, and KV transfer continues to
scale with payload and topology.
"""

from __future__ import annotations

import asyncio

from typing import Optional

from fakeengine.config import SimConfig
from fakeengine.topology import Link

_BYTES_PER_GB = 1e9
_DISAGG_DECODE_ENGINE_P50_MS = (
    (1, 3.6),
    (2, 4.1),
    (4, 6.2),
    (8, 7.5),
    (16, 8.6),
    (24, 9.8),
    (32, 10.4),
)
_REFERENCE_ITL_MS = _DISAGG_DECODE_ENGINE_P50_MS[0][1]


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
        batch size rather than once per request. The calibrated curve captures
        that shape; the two existing knobs shift its batch-one baseline and
        optionally add a linear per-request adjustment.
        """
        cfg = self.config
        batch_size = max(int(running_requests), 1)
        calibrated = _interpolate(_DISAGG_DECODE_ENGINE_P50_MS, batch_size)
        baseline_shift = cfg.itl_ms - _REFERENCE_ITL_MS
        extra = (batch_size - 1) * cfg.itl_ms_per_running_req
        return calibrated + baseline_shift + extra

    def transfer_ms(self, num_tokens: int, link: Optional[Link] = None) -> float:
        """KV payload over ``link``, or over the global link when ``None``."""
        cfg = self.config
        bandwidth = link.bandwidth_gb_s if link else cfg.kv_bandwidth_gb_s
        overhead = link.latency_ms if link else cfg.kv_transfer_overhead_ms
        payload_bytes = num_tokens * cfg.kv_bytes_per_token
        seconds = payload_bytes / (bandwidth * _BYTES_PER_GB)
        return overhead + seconds * 1000.0

    def kv_return_transfer_ms(
        self,
        generated_tokens: int,
        link: Optional[Link] = None,
    ) -> float:
        """Generated-response KV bytes over the reverse link."""
        cfg = self.config
        bandwidth = (
            link.bandwidth_gb_s
            if link is not None
            else cfg.kv_return_bandwidth_gb_s
        )
        overhead = (
            link.latency_ms
            if link is not None
            else cfg.kv_return_overhead_ms
        )
        payload_bytes = generated_tokens * cfg.kv_return_bytes_per_token
        seconds = payload_bytes / (bandwidth * _BYTES_PER_GB)
        return overhead + seconds * 1000.0

    async def sleep_ms(self, milliseconds: float) -> None:
        if milliseconds <= 0:
            return
        await asyncio.sleep(milliseconds / 1000.0 / self.config.speedup_ratio)


def _interpolate(points: tuple[tuple[int, float], ...], value: int) -> float:
    """Linearly interpolate a monotonic profile and extrapolate its last edge."""
    if value <= points[0][0]:
        return points[0][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if value <= right_x:
            ratio = (value - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    left_x, left_y = points[-2]
    right_x, right_y = points[-1]
    slope = (right_y - left_y) / (right_x - left_x)
    return right_y + (value - right_x) * slope
