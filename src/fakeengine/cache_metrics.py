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
        return_prefix = "sglang:simulated_kv_return_"
        self.kv_return_attempts = Counter(
            return_prefix + "attempts",
            "Background response-KV return tasks accepted for scheduling.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.kv_return_stored = Counter(
            return_prefix + "stored",
            "Response-KV returns committed at the original prefill rank.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.kv_return_rejected = Counter(
            return_prefix + "rejected",
            "Response-KV returns rejected before storage, by bounded reason.",
            ["dp_rank", "reason"], registry=None,
        )
        self.kv_return_failed = Counter(
            return_prefix + "failed",
            "Response-KV background failures, by bounded reason.",
            ["dp_rank", "reason"], registry=None,
        )
        self.kv_return_tokens = Counter(
            return_prefix + "tokens",
            "Generated response tokens successfully returned.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.kv_return_bytes = Counter(
            return_prefix + "bytes",
            "Simulated response-KV payload bytes successfully returned.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.kv_return_transfer_seconds = Counter(
            return_prefix + "transfer_seconds",
            "Modeled response-KV transfer seconds for completed attempts.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.kv_return_commit_seconds = Counter(
            return_prefix + "commit_seconds",
            "Control-plane commit seconds for completed attempts.",
            ["dp_rank"], registry=None,
        ).labels(str(dp_rank))
        self.kv_return_claim_delay_seconds = Histogram(
            return_prefix + "claim_delay_seconds",
            "Elapsed seconds from completed response generation to decode-retention claim attempt.",
            ["dp_rank", "result"],
            buckets=(0.001, 0.01, 0.1, 1, 5, 10, 30, 60, 120, 300),
            registry=None,
        )
        self.kv_return_active_tasks = Gauge(
            return_prefix + "active_tasks",
            "Tracked response-KV background tasks.",
            ["dp_rank"], registry=None, multiprocess_mode="livemostrecent",
        ).labels(str(dp_rank))
        self.kv_return_retained_blocks = Gauge(
            return_prefix + "retained_blocks",
            "Decode KV blocks held by response-retention leases.",
            ["dp_rank"], registry=None, multiprocess_mode="livemostrecent",
        ).labels(str(dp_rank))
        self._dp_rank = str(dp_rank)
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

    def record_kv_return_attempt(self) -> None:
        self.kv_return_attempts.inc()

    def record_kv_return_stored(
        self,
        tokens: int,
        payload_bytes: int,
        transfer_ms: float,
        commit_ms: float,
    ) -> None:
        self.kv_return_stored.inc()
        self.kv_return_tokens.inc(tokens)
        self.kv_return_bytes.inc(payload_bytes)
        self.kv_return_transfer_seconds.inc(transfer_ms / 1000.0)
        self.kv_return_commit_seconds.inc(commit_ms / 1000.0)

    def record_kv_return_rejected(self, reason: str) -> None:
        self.kv_return_rejected.labels(self._dp_rank, reason).inc()

    def record_kv_return_failed(self, reason: str) -> None:
        self.kv_return_failed.labels(self._dp_rank, reason).inc()

    def record_kv_return_claim_delay(
        self,
        seconds: float,
        result: str,
    ) -> None:
        self.kv_return_claim_delay_seconds.labels(
            self._dp_rank,
            result,
        ).observe(seconds)

    def update_kv_return(
        self,
        *,
        active_tasks: int,
        retained_blocks: int,
    ) -> None:
        self.kv_return_active_tasks.set(active_tasks)
        self.kv_return_retained_blocks.set(retained_blocks)
