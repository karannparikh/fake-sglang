"""Analytic and sampled latency models."""

from __future__ import annotations

import asyncio
import random

from typing import Optional, Protocol

from fakeengine.config import SimConfig
from fakeengine.topology import Link

_BYTES_PER_GB = 1e9
_QUANTILE_PROBABILITIES = (0.50, 0.90, 0.95, 0.99)

# H200 disaggregated decode engine latency, conditioned on active batch size.
_H200_DECODE_ENGINE_MS = (
    (1, (3.6, 6.2, 7.0, 8.4)),
    (2, (4.1, 7.2, 7.8, 8.7)),
    (4, (6.2, 8.0, 8.5, 9.7)),
    (8, (7.5, 8.8, 9.3, 11.7)),
    (16, (8.6, 10.0, 10.4, 12.0)),
    (24, (9.8, 11.0, 11.4, 12.9)),
    (32, (10.4, 12.0, 12.7, 14.8)),
)

# H200 disaggregated prefill engine latency, conditioned on recomputed context.
# The workload coordinates are the representative midpoints used by the source
# context buckets. The 8K point is a rough half-scale estimate until direct
# engine-only measurements are available below 16K.
_H200_PREFILL_ENGINE_MS = (
    (8_192, (130.0, 1_000.0, 2_500.0, 8_500.0)),
    (24_576, (260.0, 2_000.0, 5_000.0, 17_000.0)),
    (49_152, (1_050.0, 3_000.0, 5_400.0, 16_000.0)),
    (98_304, (2_450.0, 5_300.0, 6_800.0, 21_000.0)),
    (196_608, (6_500.0, 14_800.0, 22_000.0, 57_000.0)),
    (393_216, (19_000.0, 46_000.0, 73_000.0, 124_000.0)),
)


class _RandomSource(Protocol):
    def random(self) -> float: ...


class TimingModel:
    def __init__(
        self,
        config: SimConfig,
        *,
        seed_offset: int = 0,
        rng: Optional[_RandomSource] = None,
    ) -> None:
        self.config = config
        self._rng = rng or random.Random(config.timing_seed + seed_offset)

    def prefill_ms(self, num_tokens: int) -> float:
        cfg = self.config
        if cfg.timing_profile == "h200":
            sampled = _sample_profile_ms(
                _H200_PREFILL_ENGINE_MS,
                max(int(num_tokens), 0),
                self._rng.random(),
            )
            return sampled + self.kv_generation_ms(num_tokens)
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
        if cfg.timing_profile == "h200":
            return _sample_profile_ms(
                _H200_DECODE_ENGINE_MS,
                max(int(running_requests), 1),
                self._rng.random(),
            )
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


def _sample_profile_ms(
    profile: tuple[tuple[int, tuple[float, float, float, float]], ...],
    workload: int,
    percentile: float,
) -> float:
    quantiles = _interpolate_workload_quantiles(profile, workload)
    p50, p90, p95, p99 = quantiles
    lower = max(0.0, p50 - (p90 - p50) * (0.50 / 0.40))
    upper = p99 + (p99 - p95) * (0.01 / 0.04)
    inverse_cdf = (
        (0.0, lower),
        (_QUANTILE_PROBABILITIES[0], p50),
        (_QUANTILE_PROBABILITIES[1], p90),
        (_QUANTILE_PROBABILITIES[2], p95),
        (_QUANTILE_PROBABILITIES[3], p99),
        (1.0, upper),
    )
    return _interpolate(inverse_cdf, min(max(percentile, 0.0), 1.0))


def _interpolate_workload_quantiles(
    profile: tuple[tuple[int, tuple[float, float, float, float]], ...],
    workload: int,
) -> tuple[float, float, float, float]:
    if workload <= profile[0][0]:
        return profile[0][1]
    for (left_x, left), (right_x, right) in zip(profile, profile[1:]):
        if workload <= right_x:
            ratio = (workload - left_x) / (right_x - left_x)
            return tuple(
                left_value + ratio * (right_value - left_value)
                for left_value, right_value in zip(left, right)
            )
    return profile[-1][1]


def _interpolate(
    points: tuple[tuple[float, float], ...],
    value: float,
) -> float:
    if value <= points[0][0]:
        return points[0][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if value <= right_x:
            ratio = (value - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    return points[-1][1]
