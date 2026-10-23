"""Observed prefix reuse, exported through Dynamo's SGLang metrics callback."""

from __future__ import annotations

import atexit
import os
import tempfile


class PrefixCacheMetrics:
    def __init__(self, dp_rank: int = 0) -> None:
        # Dynamo collects SGLang's shared-memory metrics, not the default registry.
        # Set this before importing prometheus_client, as real SGLang does.
        if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
            directory = tempfile.TemporaryDirectory(prefix="fake-sglang-prometheus-")
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = directory.name
            atexit.register(directory.cleanup)

        from prometheus_client import Counter, Gauge, Histogram, values

        if not values.ValueClass._multiprocess:
            raise RuntimeError(
                "fake-sglang metrics require PROMETHEUS_MULTIPROC_DIR to be set "
                "before importing prometheus_client; restart the worker with it set"
            )

        prefix = "sglang:simulated_prefix_cache_"
        self.query_tokens = Counter(
            prefix + "query_tokens",
            "Prompt tokens at initial prefill admission, excluding transferred KV, no-peer probes and retries.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.hit_tokens = Counter(
            prefix + "hit_tokens",
            "Prompt tokens actually reused from the simulated local prefix cache at initial admission.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.hit_ratio = Histogram(
            prefix + "hit_ratio",
            "Local prefix tokens reused / prompt tokens per initial prefill admission; excludes transferred KV.",
            ["dp_rank"], buckets=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1),
            registry=None,
        ).labels(str(dp_rank))
        self.kv_generation_seconds = Counter(
            "sglang:simulated_kv_generation_seconds",
            "Modeled user-prefill K/V-generation seconds for uncached tokens, after speedup scaling; excludes no-peer probes.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.capacity_tokens = Gauge(
            prefix + "capacity_tokens", "Configured simulated KV token slots.",
            ["dp_rank"], registry=None, multiprocess_mode="livemostrecent",
        ).labels(str(dp_rank))
        self.resident_tokens = Gauge(
            prefix + "resident_tokens", "Reusable full-block prefix tokens, including referenced prefixes.",
            ["dp_rank"], registry=None, multiprocess_mode="livemostrecent",
        ).labels(str(dp_rank))
        self.allocated_tokens = Gauge(
            prefix + "allocated_tokens", "Occupied KV token slots, including prefix cache and active partial blocks.",
            ["dp_rank"], registry=None, multiprocess_mode="livemostrecent",
        ).labels(str(dp_rank))
        self.evictable_tokens = Gauge(
            prefix + "evictable_tokens", "Unreferenced resident prefix tokens; eviction still respects parent dependencies.",
            ["dp_rank"], registry=None, multiprocess_mode="livemostrecent",
        ).labels(str(dp_rank))
        self.evicted_tokens = Counter(
            prefix + "evicted_tokens", "Prefix tokens removed by allocation pressure, excluding explicit cache clears.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.cleared_tokens = Counter(
            prefix + "cleared_tokens", "Prefix tokens removed by explicit cache clears, not allocation pressure.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self._last_evicted = self._last_cleared = 0

    def observe(self, prompt_tokens: int, hit_tokens: int) -> None:
        if not 0 <= hit_tokens <= prompt_tokens or prompt_tokens <= 0:
            raise ValueError("Expected 0 <= hit_tokens <= prompt_tokens and a nonempty prompt")
        self.query_tokens.inc(prompt_tokens)
        self.hit_tokens.inc(hit_tokens)
        self.hit_ratio.observe(hit_tokens / prompt_tokens)

    def record_kv_generation(self, seconds: float) -> None:
        self.kv_generation_seconds.inc(seconds)

    def update_pool(self, pool) -> None:
        self.capacity_tokens.set(pool.num_blocks * pool.block_size)
        self.resident_tokens.set(pool.cached_blocks * pool.block_size)
        self.allocated_tokens.set(pool.used_blocks * pool.block_size)
        self.evictable_tokens.set(pool.evictable_blocks * pool.block_size)
        self.evicted_tokens.inc((pool.evicted_blocks - self._last_evicted) * pool.block_size)
        self.cleared_tokens.inc((pool.cleared_blocks - self._last_cleared) * pool.block_size)
        self._last_evicted, self._last_cleared = pool.evicted_blocks, pool.cleared_blocks
