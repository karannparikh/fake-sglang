"""Observed hits must be local reuse, not predictions or transferred KV."""

import os
import subprocess
import sys
import textwrap
from unittest.mock import Mock

import pytest

from fakeengine.cache_metrics import PrefixCacheMetrics
from fakeengine.config import SimConfig
from fakeengine.scheduler import Request, Scheduler
from fakeengine.stats import MetricsPusher
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.engine import Engine


def scheduler_for(server_args, **overrides):
    return Scheduler(
        SimConfig.from_server_args(server_args(**overrides)),
        cache_metrics=Mock(spec=PrefixCacheMetrics),
    )


async def test_cold_warm_and_partial_prefix_hits(server_args):
    scheduler = scheduler_for(server_args, page_size=4)
    try:
        for rid, prompt, expected_hit in [
            ("cold", list(range(12)), 0),
            ("warm", list(range(12)), 12),
            ("partial", list(range(8)) + [100, 101, 102, 103], 8),
        ]:
            scheduler.submit(Request(rid, prompt, 1))
            await scheduler._prefill_step()
            scheduler.cache_metrics.observe.assert_called_with(12, expected_hit)
        assert scheduler.prefix_cache_queries == 3
        assert scheduler.prefix_cache_query_tokens == 36
        assert scheduler.prefix_cache_hit_tokens == 20
        snapshot = MetricsPusher(scheduler, "ipc:///unused")._snapshot()
        assert snapshot.gpu_prefix_cache_hit_rate == pytest.approx(20 / 36)
    finally:
        await scheduler.stop()


@pytest.mark.parametrize("length,expected_hit", [(3, 0), (4, 4), (5, 4)])
async def test_only_full_blocks_hit_but_all_prompt_tokens_are_queried(
    server_args, length, expected_hit,
):
    scheduler = scheduler_for(server_args, page_size=4)
    try:
        for rid in ("cold", "warm"):
            scheduler.submit(Request(rid, list(range(length)), 1))
            await scheduler._prefill_step()
        assert scheduler.prefix_cache_query_tokens == 2 * length
        assert scheduler.prefix_cache_hit_tokens == expected_hit
        scheduler.cache_metrics.observe.assert_called_with(length, expected_hit)
    finally:
        await scheduler.stop()


async def test_transferred_kv_is_excluded_from_both_counters(server_args):
    scheduler = scheduler_for(server_args, page_size=4)
    request = scheduler.submit(Request("transferred", list(range(12)), 1, prebuilt=True))
    try:
        await scheduler._prefill_step()
        assert request.cached_tokens == 12  # Preserve the existing backend contract.
        assert scheduler.prefix_cache_query_tokens == 0
        assert scheduler.prefix_cache_hit_tokens == 0
        assert scheduler.prefix_cache_queries == 0
        scheduler.cache_metrics.observe.assert_not_called()
    finally:
        await scheduler.stop()


async def test_no_peer_probes_and_warmup_are_not_inference_observations(server_args, unused_port):
    engine = Engine(server_args(
        disaggregation_mode="prefill", disaggregation_bootstrap_port=unused_port,
        page_size=4,
    ))
    try:
        for prompt in ([1], [10, 11, 12, 13]):
            stream = await engine.async_generate(
                input_ids=prompt, sampling_params={"max_new_tokens": 1},
                bootstrap_host=FAKE_BOOTSTRAP_HOST,
                bootstrap_port=unused_port, bootstrap_room=0,
            )
            async for _ in stream:
                pass
        assert engine.scheduler.prefix_cache_queries == 0
        assert engine.scheduler.prefix_cache_query_tokens == 0
        assert engine.scheduler.prefix_cache_hit_tokens == 0

        stream = await engine.async_generate(
            input_ids=[1], sampling_params={"max_new_tokens": 1},
        )
        async for _ in stream:
            pass
        assert engine.scheduler.prefix_cache_queries == 1
        assert engine.scheduler.prefix_cache_query_tokens == 1
    finally:
        await engine._async_shutdown()


async def test_chunks_and_preemption_do_not_repeat_observations(server_args):
    scheduler = scheduler_for(server_args, page_size=4, max_prefill_tokens=4)
    request = scheduler.submit(Request("chunked", list(range(10)), 4))
    try:
        for _ in range(3):
            await scheduler._prefill_step()
        assert request.prefilled == 10
        assert scheduler._preempt_one(protect=Request("other", [99], 2))
        while request in scheduler.waiting:
            await scheduler._prefill_step()
        assert scheduler.prefix_cache_queries == 1
        assert scheduler.prefix_cache_query_tokens == 10
        assert scheduler.prefix_cache_hit_tokens == 0
        scheduler.cache_metrics.observe.assert_called_once_with(10, 0)
    finally:
        await scheduler.stop()


async def test_blocked_cached_admission_does_not_reacquire_or_count(server_args):
    scheduler = scheduler_for(
        server_args, page_size=4, fake_num_kv_blocks=4, max_prefill_tokens=4,
    )
    try:
        scheduler.submit(Request("seed", list(range(8)), 1))
        await scheduler._prefill_step()
        await scheduler._prefill_step()
        held = scheduler.pool.allocate(2)
        request = scheduler.submit(Request("blocked", list(range(12)), 2))
        for _ in range(3):
            assert not await scheduler._prefill_step()
            assert len(request.blocks) == 2
            assert scheduler.prefix_cache_queries == 1
        scheduler.pool.free(held)
        await scheduler._prefill_step()
        await scheduler._prefill_step()
        await scheduler._prefill_step()
        assert scheduler.prefix_cache_queries == 2
        assert scheduler.prefix_cache_hit_tokens == 8
        assert scheduler.prefix_cache_query_tokens == 20
        assert scheduler.abort(request.rid)
        assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    finally:
        await scheduler.stop()


async def test_failed_growth_and_abort_before_admission_are_not_observed(server_args):
    scheduler = scheduler_for(server_args, page_size=4, fake_num_kv_blocks=4)
    held = scheduler.pool.allocate(3)
    try:
        scheduler.submit(Request("blocked", list(range(8)), 2))
        for _ in range(3):
            assert not await scheduler._prefill_step()
        assert scheduler.prefix_cache_queries == 0
        assert scheduler.abort("blocked")
        scheduler.pool.free(held)
        scheduler.submit(Request("admitted", list(range(8)), 2))
        await scheduler._prefill_step()
        assert scheduler.abort("admitted")
        scheduler.cache_metrics.observe.assert_called_once_with(8, 0)
    finally:
        await scheduler.stop()


async def test_flush_removes_hits_without_resetting_counters(server_args):
    scheduler = scheduler_for(server_args, page_size=4)
    try:
        for rid in ("cold", "warm"):
            scheduler.submit(Request(rid, list(range(8)), 1))
            await scheduler._prefill_step()
        scheduler.flush_cache()
        scheduler.submit(Request("after-flush", list(range(8)), 1))
        await scheduler._prefill_step()
        assert scheduler.prefix_cache_queries == 3
        assert scheduler.prefix_cache_query_tokens == 24
        assert scheduler.prefix_cache_hit_tokens == 8
        scheduler.cache_metrics.observe.assert_called_with(8, 0)
    finally:
        await scheduler.stop()


@pytest.mark.parametrize("prompt,hit", [(0, 0), (-1, 0), (2, -1), (2, 3)])
def test_invalid_observations_fail_explicitly(prompt, hit):
    metrics = PrefixCacheMetrics.__new__(PrefixCacheMetrics)
    with pytest.raises(ValueError, match="nonempty prompt"):
        metrics.observe(prompt, hit)


def test_engine_metrics_export_through_multiprocess_collector():
    env = os.environ.copy()
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    env.pop("prometheus_multiproc_dir", None)
    script = textwrap.dedent("""
        import asyncio
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.server_args import ServerArgs
        from fakeengine.scheduler import Request

        engine = Engine(ServerArgs(
            model_path="stub-model", enable_metrics=True, page_size=4,
            fake_num_kv_blocks=4, fake_speedup_ratio=10,
            fake_kv_generation_ms_per_1k_tokens=100,
        ))
        from prometheus_client import CollectorRegistry
        from prometheus_client.multiprocess import MultiProcessCollector
        registry = CollectorRegistry()
        MultiProcessCollector(registry)

        def value(name):
            return registry.get_sample_value(name, {"dp_rank": "0"})

        prefix = "sglang:simulated_prefix_cache_"
        assert value(prefix + "capacity_tokens") == 16
        assert value(prefix + "query_tokens_total") == 0

        async def run():
            try:
                for rid in ("cold", "warm"):
                    engine.scheduler.submit(Request(rid, list(range(8)), 1))
                    await engine.scheduler._prefill_step()
                assert value(prefix + "query_tokens_total") == 16
                assert value(prefix + "hit_tokens_total") == 8
                assert value(prefix + "hit_ratio_count") == 2
                assert value(prefix + "hit_ratio_sum") == 1
                assert value("sglang:simulated_kv_generation_seconds_total") == 0.00008
                assert value(prefix + "resident_tokens") == 8
                assert value(prefix + "allocated_tokens") == 8
                assert value(prefix + "evictable_tokens") == 8
                engine.scheduler.flush_cache()
                assert value(prefix + "resident_tokens") == 0
                assert value(prefix + "cleared_tokens_total") == 8
                assert value(prefix + "evicted_tokens_total") == 0
            finally:
                await engine._async_shutdown()

        asyncio.run(run())
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_engine_metrics_reach_genuine_dynamo_callback():
    pytest.importorskip("dynamo.sglang.publisher")
    # Prometheus selects its value implementation at import time. Exercise the
    # real worker initialization order in an isolated process, not pytest's.
    env = os.environ.copy()
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    env.pop("prometheus_multiproc_dir", None)
    script = textwrap.dedent("""
        import asyncio
        from types import SimpleNamespace
        from dynamo.sglang.publisher import setup_prometheus_registry
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.server_args import ServerArgs
        from fakeengine.scheduler import Request

        engine = Engine(ServerArgs(
            model_path="stub-model", served_model_name="stub-model",
            enable_metrics=True, page_size=4, fake_speedup_ratio=200,
        ))
        callbacks = []
        typed_callbacks = []
        endpoint = SimpleNamespace(
            metrics=SimpleNamespace(
                register_prometheus_expfmt_callback=callbacks.append,
                register_prometheus_typed_callback=typed_callbacks.append,
            ),
            connection_id=lambda: 123,
        )
        config = SimpleNamespace(dynamo_args=SimpleNamespace(
            namespace="test", component="prefill", endpoint="generate",
        ))
        setup_prometheus_registry(engine, endpoint, config)
        from prometheus_client.parser import text_string_to_metric_families

        def samples():
            return {
                sample.name: sample
                for family in text_string_to_metric_families(callbacks[0]())
                for sample in family.samples
                if not sample.name.endswith("_bucket")
            }

        prefix = "sglang:simulated_prefix_cache_"
        assert samples()[prefix + "query_tokens_total"].value == 0
        assert samples()[prefix + "capacity_tokens"].value == engine.sim_config.total_kv_tokens
        assert samples()[prefix + "resident_tokens"].value == 0

        async def run():
            try:
                for rid, prebuilt in [("cold", False), ("warm", False), ("decode", True)]:
                    engine.scheduler.submit(Request(rid, list(range(5)), 1, prebuilt=prebuilt))
                    await engine.scheduler._prefill_step()
                data = samples()
                assert data[prefix + "query_tokens_total"].value == 10
                assert data[prefix + "hit_tokens_total"].value == 4
                assert data[prefix + "hit_ratio_count"].value == 2
                assert data[prefix + "hit_ratio_sum"].value == 0.8
                labels = data[prefix + "hit_tokens_total"].labels
                assert labels["dp_rank"] == "0"
                assert labels["worker_id"] == "7b"
                assert labels["dynamo_component"] == "prefill"
                assert labels["dynamo_namespace"] == "test"
                assert labels["model_name"] == "stub-model"
                pool = engine.scheduler.pool
                assert data[prefix + "resident_tokens"].value == pool.cached_blocks * 4
                assert data[prefix + "allocated_tokens"].value == pool.used_blocks * 4
                assert data[prefix + "evictable_tokens"].value == pool.evictable_blocks * 4
                assert data[prefix + "evicted_tokens_total"].value == 0
                held = pool.allocate(pool.num_blocks)
                data = samples()
                assert data[prefix + "evicted_tokens_total"].value == 4
                assert data[prefix + "resident_tokens"].value == 0
                assert data[prefix + "allocated_tokens"].value == pool.num_blocks * 4
                pool.free(held)
                engine.scheduler.submit(Request("after-pressure", list(range(4)), 1))
                await engine.scheduler._prefill_step()
                engine.scheduler.flush_cache()
                data = samples()
                assert data[prefix + "evicted_tokens_total"].value == 4
                assert data[prefix + "cleared_tokens_total"].value == 4
                assert data[prefix + "resident_tokens"].value == 0
            finally:
                await engine._async_shutdown()

        asyncio.run(run())
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
