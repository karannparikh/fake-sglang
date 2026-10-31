import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import zmq
import zmq.asyncio

from fakeengine.config import SimConfig
from fakeengine.stats import MetricsPusher
from sglang.srt.disaggregation.kv_events import KVEventBatch, ZmqEventPublisher
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.observability.forward_pass_metrics import decode


async def generate(engine, rank, tokens, rid, **kwargs):
    stream = await engine.async_generate(
        input_ids=tokens, routed_dp_rank=rank, rid=rid,
        sampling_params={"max_new_tokens": 1}, **kwargs,
    )
    return [chunk async for chunk in stream]


@pytest.fixture
def packed_args(server_args):
    def make(**kwargs):
        return server_args(dp_size=2, enable_dp_attention=True, **kwargs)
    return make


@pytest.fixture
def ipc_endpoint():
    with tempfile.TemporaryDirectory(prefix="packed-", dir="/tmp") as directory:
        yield f"ipc://{directory}/fpm"


@pytest.fixture
def kv_port():
    for _ in range(20):
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port == 65535:
                continue
            try:
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port
    pytest.fail("Could not reserve two consecutive KV-event ports")


async def test_packed_capacity_is_per_rank(server_args):
    args = server_args(
        dp_size=10, max_running_requests=80, fake_num_kv_blocks=8192, page_size=16,
        enable_dp_attention=True,
    )
    config = SimConfig.from_server_args(args)
    assert config.max_running_requests == 8
    assert config.total_kv_tokens == 131072
    assert SimConfig.from_server_args(server_args(dp_size=10)).max_running_requests == 32
    engine = Engine(args)
    try:
        assert len(engine.schedulers) == 10
        assert len({id(s.pool) for s in engine.schedulers}) == 10
        for scheduler, info in zip(engine.schedulers, engine._scheduler_init_result.scheduler_infos):
            assert scheduler.pool.num_blocks * scheduler.pool.block_size == 131072
            assert info["max_total_num_tokens"] == 131072
            assert info["max_running_requests"] == 8
    finally:
        await engine._async_shutdown()


@pytest.mark.parametrize("options", [
    {"dp_size": 0}, {"dp_size": -1}, {"dp_size": True}, {"dp_size": "2"},
    {"dp_size": 2.0}, {"max_running_requests": 0}, {"max_running_requests": -1},
    {"max_running_requests": True}, {"max_running_requests": "8"},
    {"dp_size": 2, "max_running_requests": 3},
    {"dp_size": 2, "max_running_requests": 1},
    {"dp_size": 2, "enable_dp_attention": False},
    {"dp_size": 2, "enable_dp_attention": True, "nnodes": 2},
    {"dp_size": 2, "enable_dp_attention": True, "node_rank": 1},
    {"dp_size": 2, "enable_dp_attention": True, "is_embedding": True},
])
def test_invalid_packed_configuration_fails(server_args, options):
    with pytest.raises(ValueError):
        Engine(server_args=server_args(**options))


@pytest.mark.parametrize("disable_radix_cache", [False, True])
async def test_rank_cache_salt_load_and_flush_are_isolated(packed_args, disable_radix_cache):
    engine = Engine(packed_args(
        page_size=4, fake_num_kv_blocks=32, max_running_requests=16,
        disable_radix_cache=disable_radix_cache,
    ))
    try:
        prompt = list(range(16))
        for rank in range(2):
            for salt in ("tenant-a", "tenant-b", None):
                for repeat in range(2):
                    result = await generate(
                        engine, rank, prompt, f"{rank}-{salt}-{repeat}", cache_salt=salt,
                    )
                    expected = 16 if repeat and not disable_radix_cache else 0
                    assert result[-1]["meta_info"]["cached_tokens"] == expected
        assert engine.schedulers[0].pool is not engine.schedulers[1].pool
        assert [s.prefix_cache_queries for s in engine.schedulers] == [6, 6]
        assert engine.scheduler is engine.schedulers[0]
        assert engine.kv_events is engine._rank_events[0]
        assert engine._metrics is engine._rank_metrics[0]
        assert len(engine._scheduler_init_result.scheduler_infos) == 2
        for rank, scheduler in enumerate(engine.schedulers):
            info = engine._scheduler_init_result.scheduler_infos[rank]
            assert info["max_total_num_tokens"] == 128
            assert info["max_running_requests"] == 8
            metrics = MetricsPusher(scheduler, "unused", dp_rank=rank)._snapshot()
            assert metrics.data_parallel_rank == rank
            assert metrics.kv_total_blocks == 128
            assert metrics.request_total_slots == 8
            assert metrics.gpu_prefix_cache_hit_rate == (0 if disable_radix_cache else 0.5)
        await engine.tokenizer_manager.flush_cache()
        assert all(s.pool.cached_blocks == 0 for s in engine.schedulers)
        for rank in range(2):
            result = await generate(engine, rank, prompt, f"flushed-{rank}", cache_salt="tenant-a")
            assert result[-1]["meta_info"]["cached_tokens"] == 0
    finally:
        await engine._async_shutdown()


async def test_one_rank_can_progress_while_another_is_busy(packed_args):
    engine = Engine(packed_args())
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocked_sleep(_milliseconds):
        entered.set()
        await release.wait()

    engine.schedulers[0].timing.sleep_ms = blocked_sleep
    task = asyncio.create_task(generate(engine, 0, list(range(16)), "blocked"))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        result = await asyncio.wait_for(generate(engine, 1, [101, 102], "free"), 2)
        assert result[-1]["meta_info"]["completion_tokens"] == 1
        assert not task.done()
        assert engine._rank_metrics[0]._snapshot().kv_active_blocks == 16
        assert engine._rank_metrics[1]._snapshot().request_active_slots == 0
        engine.tokenizer_manager.abort_request("blocked")
        assert engine.schedulers[0].by_rid.get("blocked") is None
        assert engine.schedulers[1].prefix_cache_queries == 1
        result = await asyncio.wait_for(task, 2)
        assert result[-1]["meta_info"]["finish_reason"]["type"] == "abort"
    finally:
        release.set()
        await task
        await engine._async_shutdown()


@pytest.mark.parametrize("rank", [-1, 2, True, False, "1", 1.0, [], {}])
@pytest.mark.parametrize("argument", ["routed_dp_rank", "data_parallel_rank"])
async def test_invalid_routed_rank_fails_without_admission(packed_args, rank, argument):
    engine = Engine(packed_args())
    try:
        if argument == "data_parallel_rank":
            with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="rank"):
                await engine.async_generate(input_ids=[1, 2], **{argument: rank})
        else:
            with pytest.raises(ValueError, match="rank"):
                await engine.async_generate(input_ids=[1, 2], **{argument: rank})
        assert all(not s.by_rid for s in engine.schedulers)
    finally:
        await engine._async_shutdown()


async def test_unrouted_round_robin_alias_precedence_and_tokenizer_forwarding(packed_args):
    engine = Engine(packed_args(page_size=4))
    try:
        for i in range(4):
            await generate(engine, None, [1, 2, 3, 4], f"unrouted-{i}")
        assert [s.prefix_cache_queries for s in engine.schedulers] == [2, 2]
        with pytest.warns(DeprecationWarning):
            await generate(engine, None, [1, 2, 3, 4], "alias", data_parallel_rank=1)
        with pytest.warns(DeprecationWarning):
            await generate(engine, 0, [1, 2, 3, 4], "primary", data_parallel_rank=1)
        assert [s.prefix_cache_queries for s in engine.schedulers] == [3, 3]
        for i in range(2):
            stream = engine.tokenizer_manager.generate_request(SimpleNamespace(
                input_ids=[1, 2, 3, 4], sampling_params={"max_new_tokens": 1},
                routed_dp_rank=1, cache_salt="tokenizer-tenant",
            ))
            result = [chunk async for chunk in stream]
            assert result[-1]["meta_info"]["cached_tokens"] == i * 4
        assert [s.prefix_cache_queries for s in engine.schedulers] == [3, 5]
    finally:
        await engine._async_shutdown()


@pytest.mark.parametrize("rank", [True, False, "0", 0.0, -1, 1])
async def test_single_rank_rejects_invalid_route_types(server_args, rank):
    engine = Engine(server_args())
    try:
        with pytest.raises(ValueError):
            await generate(engine, rank, [1, 2], "invalid")
        assert engine._resolve_routed_dp_rank(0, None) is None
    finally:
        await engine._async_shutdown()


@pytest.mark.parametrize("mode", ["null", "decode"])
@pytest.mark.parametrize("tokenizer", [False, True])
async def test_close_and_abort_all_release_every_rank(packed_args, mode, tokenizer):
    engine = Engine(packed_args(disaggregation_mode=mode))
    streams = []
    try:
        for rank in range(2):
            args = dict(
                input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 10000},
                rid=f"active-{rank}", routed_dp_rank=rank,
                bootstrap_host=FAKE_BOOTSTRAP_HOST,
                bootstrap_room=rank if mode == "decode" else None,
            )
            stream = (
                engine.tokenizer_manager.generate_request(SimpleNamespace(**args))
                if tokenizer else await engine.async_generate(**args)
            )
            streams.append(stream)
            await asyncio.wait_for(anext(stream), 2)
        await streams[1].aclose()
        assert engine.schedulers[1].by_rid == {}
        assert "active-0" in engine.schedulers[0].by_rid
        engine.tokenizer_manager.abort_request(abort_all=True)
        assert all(not s.by_rid and not s.running for s in engine.schedulers)
        assert all(s.pool.can_allocate(s.pool.num_blocks) for s in engine.schedulers)
    finally:
        for stream in streams:
            await stream.aclose()
        await engine._async_shutdown()


async def test_all_rank_telemetry_reaches_real_sockets(packed_args, ipc_endpoint, kv_port):
    engine = Engine(packed_args(
        page_size=4, fake_num_kv_blocks=32, max_running_requests=16,
        enable_forward_pass_metrics=True,
        forward_pass_metrics_ipc_name=ipc_endpoint,
        forward_pass_metrics_worker_id="packed-worker",
        kv_events_config=json.dumps({"publisher": "zmq", "endpoint": f"tcp://*:{kv_port}"}),
    ))
    context = zmq.asyncio.Context()
    sockets = []
    publishers = [s.fpm for s in engine.schedulers]
    try:
        fpm_subs, kv_subs = [], []
        for rank in range(2):
            for endpoint, subscribers in [
                (f"{ipc_endpoint}.{rank}", fpm_subs),
                (f"tcp://127.0.0.1:{kv_port + rank}", kv_subs),
            ]:
                subscriber = context.socket(zmq.SUB)
                sockets.append(subscriber)
                subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                subscriber.connect(endpoint)
                subscribers.append(subscriber)
        metrics = context.socket(zmq.PULL)
        sockets.append(metrics)
        metrics.bind(engine.port_args.metrics_ipc_name)
        await asyncio.sleep(0.2)
        for rank in range(2):
            prompt = list(range(4 * (rank + 1)))
            await generate(engine, rank, prompt, f"wire-{rank}", cache_salt="tenant")
            batch = msgspec.msgpack.decode(
                (await asyncio.wait_for(kv_subs[rank].recv_multipart(), 2))[2],
                type=KVEventBatch,
            )
            assert batch.attn_dp_rank == rank
            assert batch.events[0].token_ids == prompt
            assert batch.events[0].cache_salt == "tenant"
            async def next_prefill():
                while True:
                    frames = await fpm_subs[rank].recv_multipart()
                    snapshot = decode(frames[2])
                    assert snapshot.dp_rank == rank
                    assert snapshot.worker_id == "packed-worker"
                    assert snapshot.counter_id == int.from_bytes(frames[1], "big")
                    if snapshot.scheduled_requests.num_prefill_requests:
                        return snapshot
            snapshot = await asyncio.wait_for(next_prefill(), 2)
            assert snapshot.scheduled_requests.sum_prefill_tokens == len(prompt)

        async def collect_metrics():
            seen = {}
            while len(seen) < 2:
                snapshot = await metrics.recv_pyobj()
                seen[snapshot.data_parallel_rank] = snapshot
            return seen
        seen = await asyncio.wait_for(collect_metrics(), 2)
        assert set(seen) == {0, 1}
        assert all(s.request_total_slots == 8 and s.kv_total_blocks == 128 for s in seen.values())
    finally:
        await engine._async_shutdown()
        for sock in sockets:
            sock.close(linger=0)
        context.term()
    assert all(p._socket.closed and p._context.closed for p in publishers)
    assert all(e._publisher._pub.closed and not e._publisher._thread.is_alive()
               for e in engine._rank_events)
    assert all(m._task is None and m._socket is None for m in engine._rank_metrics)


@pytest.mark.parametrize("failure_stage", [
    "build_emitter", "ForwardPassPublisher", "Scheduler", "MetricsPusher",
])
def test_later_rank_init_failure_closes_every_created_resource(
    packed_args, ipc_endpoint, monkeypatch, failure_stage,
):
    import sglang.srt.entrypoints.engine as engine_module

    events, publishers = [], []
    original_fpm = engine_module.ForwardPassPublisher

    def make_events(*args, **kwargs):
        events.append(Mock())
        return events[-1]

    def make_fpm(*args, **kwargs):
        publishers.append(original_fpm(*args, **kwargs))
        return publishers[-1]

    monkeypatch.setattr(engine_module, "build_emitter", make_events)
    monkeypatch.setattr(engine_module, "ForwardPassPublisher", make_fpm)
    original = getattr(engine_module, failure_stage)
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("rank one init failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(engine_module, failure_stage, fail_second)
    with pytest.raises(RuntimeError, match="rank one init failed"):
        Engine(packed_args(
            enable_forward_pass_metrics=True,
            forward_pass_metrics_ipc_name=ipc_endpoint,
            forward_pass_metrics_worker_id="packed-worker",
        ))
    assert events and publishers
    for event in events:
        event.shutdown.assert_called_once()
    assert all(p._socket.closed and p._context.closed for p in publishers)


def test_failed_kv_bind_closes_partial_rank_socket(monkeypatch, unused_port):
    context = zmq.Context()
    blocker = context.socket(zmq.PUB)
    blocker.bind(f"tcp://*:{unused_port}")
    created = []
    original_socket = zmq.Context.socket

    def record_socket(self, *args, **kwargs):
        sock = original_socket(self, *args, **kwargs)
        created.append(sock)
        return sock

    monkeypatch.setattr(zmq.Context, "socket", record_socket)
    try:
        with pytest.raises(zmq.ZMQError):
            ZmqEventPublisher(endpoint=f"tcp://*:{unused_port}")
        assert created and all(s.closed for s in created)
    finally:
        blocker.close(linger=0)
        context.term()


@pytest.mark.parametrize("failed_resource", ["scheduler", "metrics"])
async def test_shutdown_closes_all_ranks_even_after_task_failure(
    packed_args, ipc_endpoint, failed_resource,
):
    engine = Engine(packed_args(
        enable_forward_pass_metrics=True,
        forward_pass_metrics_ipc_name=ipc_endpoint,
        forward_pass_metrics_worker_id="packed-worker",
    ))
    publishers = [s.fpm for s in engine.schedulers]
    await asyncio.sleep(0)
    resource = engine.schedulers[1] if failed_resource == "scheduler" else engine._rank_metrics[1]
    resource._task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await resource._task

    async def fail():
        raise RuntimeError("background task failed")

    resource._task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="background task failed"):
        await engine._async_shutdown()
    assert all(s._task is None for s in engine.schedulers)
    assert all(m._task is None and m._socket is None for m in engine._rank_metrics)
    assert all(p._socket.closed and p._context.closed for p in publishers)
    await engine._async_shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        await generate(engine, 0, [1], "late")


async def test_shutdown_before_bootstrap_starts_does_not_leak(packed_args, unused_port):
    engine = Engine(packed_args(
        disaggregation_mode="prefill", disaggregation_bootstrap_port=unused_port,
    ))
    ready = engine._bootstrap_ready
    await engine._async_shutdown()
    assert ready.done()
    assert engine._bootstrap_server._runner is None
    assert all(s._task is None for s in engine.schedulers)


async def test_packed_disaggregation_preserves_topology_and_rank_selection(
    packed_args, unused_port, monkeypatch,
):
    from fakeengine.transfer import KVReceiver

    topology = json.dumps({
        "levels": ["zone"],
        "links": {
            "zone": {"bandwidth_gb_s": 100, "latency_ms": 0.1},
            "default": {"bandwidth_gb_s": 1, "latency_ms": 5},
        },
    })
    prefill = Engine(packed_args(
        disaggregation_mode="prefill", disaggregation_bootstrap_port=unused_port,
        fake_topology=topology, fake_location="a", page_size=4,
        enable_kv_return=True,
    ))
    decode_engine = Engine(packed_args(
        disaggregation_mode="decode", fake_topology=topology, fake_location="b",
        page_size=4,
    ))
    metadata = []
    original_receive = KVReceiver.receive
    for scheduler in decode_engine.schedulers:
        monkeypatch.setattr(scheduler, "submit", Mock(wraps=scheduler.submit))

    async def receive(self, num_tokens):
        result = await original_receive(self, num_tokens)
        metadata.append(result)
        return result

    monkeypatch.setattr(KVReceiver, "receive", receive)
    try:
        for rank in range(2):
            args = dict(
                bootstrap_host="127.0.0.1", bootstrap_port=unused_port,
                bootstrap_room=100 + rank,
            )
            prompt = list(range(16))
            first = await generate(prefill, rank, prompt, f"prefill-{rank}", **args)
            last = await generate(decode_engine, rank, prompt, f"decode-{rank}", **args)
            assert first[-1]["output_ids"] == last[-1]["output_ids"]
            assert last[-1]["meta_info"]["cached_tokens"] == 16
            assert metadata[-1]["link"] == "default"
            assert metadata[-1]["transfer_ms"] > 5
            assert (
                metadata[-1]["kv_return"]["compatibility"]["prefill_dp_rank"]
                == rank
            )
            decode_engine.schedulers[rank].submit.assert_called_once()
            assert decode_engine.schedulers[rank].submit.call_args.args[0].prebuilt
            if rank == 0:
                decode_engine.schedulers[1].submit.assert_not_called()
        assert [s.prefix_cache_queries for s in prefill.schedulers] == [1, 1]
        assert [s.prefix_cache_queries for s in decode_engine.schedulers] == [0, 0]
        assert prefill._bootstrap_server._rooms == {}
    finally:
        await prefill._async_shutdown()
        await decode_engine._async_shutdown()
    assert prefill._bootstrap_server._runner is None
    assert decode_engine._bootstrap_client._session.closed


def test_packed_prometheus_metrics_are_rank_labeled():
    env = os.environ.copy()
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    env.pop("prometheus_multiproc_dir", None)
    script = textwrap.dedent("""
        import asyncio
        from sglang.srt.entrypoints.engine import Engine
        from sglang.srt.server_args import ServerArgs

        engine = Engine(ServerArgs(
            model_path="stub-model", dp_size=2, enable_dp_attention=True,
            enable_metrics=True, page_size=4, fake_num_kv_blocks=32,
            fake_speedup_ratio=200,
        ))
        from prometheus_client import CollectorRegistry
        from prometheus_client.multiprocess import MultiProcessCollector
        registry = CollectorRegistry()
        MultiProcessCollector(registry)
        def value(name, rank):
            return registry.get_sample_value(
                "sglang:simulated_prefix_cache_" + name, {"dp_rank": str(rank)},
            )

        async def run():
            try:
                for rank in (0, 0, 1):
                    stream = await engine.async_generate(
                        input_ids=list(range(8)), routed_dp_rank=rank,
                        sampling_params={"max_new_tokens": 1},
                    )
                    async for _ in stream:
                        pass
                for rank in (0, 1):
                    assert value("capacity_tokens", rank) == 128
                    assert value("resident_tokens", rank) == 8
                    assert value("query_tokens_total", rank) == (16 if rank == 0 else 8)
                    assert value("hit_tokens_total", rank) == (8 if rank == 0 else 0)
                    assert value("hit_ratio_count", rank) == (2 if rank == 0 else 1)
            finally:
                await engine._async_shutdown()
        asyncio.run(run())
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
