"""Forward-pass snapshots must describe simulated work and reach Dynamo's decoder."""

import asyncio
import tempfile
from unittest.mock import Mock

import pytest
import zmq
import zmq.asyncio

from fakeengine.config import SimConfig
from fakeengine.fpm import ForwardPassPublisher
from fakeengine.scheduler import Request, Scheduler
from sglang.srt.observability.forward_pass_metrics import decode
from sglang.srt.entrypoints.engine import Engine


@pytest.fixture
def fpm_endpoint():
    # macOS has a 103-byte UNIX socket path limit; pytest's paths can exceed it.
    with tempfile.TemporaryDirectory(prefix="fpm-", dir="/tmp") as directory:
        yield f"ipc://{directory}/fpm"


@pytest.mark.parametrize("endpoint,worker_id", [("", "worker"), ("ipc:///tmp/fpm", "")])
def test_fpm_requires_endpoint_and_identity(endpoint, worker_id):
    with pytest.raises(ValueError, match="IPC endpoint and worker identity"):
        ForwardPassPublisher(endpoint, worker_id)


def test_engine_closes_publishers_on_initialization_failure(server_args, monkeypatch, fpm_endpoint):
    import sglang.srt.entrypoints.engine as engine_module

    events = Mock()
    monkeypatch.setattr(engine_module, "build_emitter", lambda *args, **kwargs: events)
    def fail(*args, **kwargs):
        raise RuntimeError("scheduler init failed")

    monkeypatch.setattr(engine_module, "Scheduler", fail)
    with pytest.raises(RuntimeError, match="scheduler init failed"):
        Engine(server_args(
            enable_forward_pass_metrics=True,
            forward_pass_metrics_ipc_name=fpm_endpoint,
            forward_pass_metrics_worker_id="test-worker",
        ))
    events.shutdown.assert_called_once()
    # Rebinding proves initialization failure released the actual ZMQ endpoint.
    publisher = ForwardPassPublisher(fpm_endpoint, "test-worker")
    publisher.close()


async def test_engine_wires_single_rank_publisher(server_args, fpm_endpoint):
    engine = Engine(server_args(
        enable_forward_pass_metrics=True,
        forward_pass_metrics_ipc_name=fpm_endpoint,
        forward_pass_metrics_worker_id="engine-worker",
    ))
    publisher = engine.scheduler.fpm
    context = zmq.asyncio.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(f"{fpm_endpoint}.0")
    try:
        await asyncio.sleep(0.1)
        stream = await engine.async_generate(
            input_ids=[1, 2, 3, 4], sampling_params={"max_new_tokens": 1},
            routed_dp_rank=0,
        )
        async for _ in stream:
            pass
        while True:
            frames = await asyncio.wait_for(subscriber.recv_multipart(), timeout=2)
            snapshot = decode(frames[2])
            assert snapshot.worker_id == "engine-worker"
            assert snapshot.dp_rank == 0
            if snapshot.scheduled_requests.sum_prefill_tokens:
                assert snapshot.scheduled_requests.sum_prefill_tokens == 4
                break
    finally:
        await engine._async_shutdown()
        subscriber.close(linger=0)
        context.term()
    assert publisher._socket.closed
    assert publisher._context.closed


async def test_chunked_prefill_reports_scheduled_and_queued_work(server_args):
    publisher = Mock(spec=ForwardPassPublisher)
    config = SimConfig.from_server_args(server_args(max_prefill_tokens=6))
    scheduler = Scheduler(config, fpm=publisher)
    scheduler.submit(Request("first", list(range(10)), max_new_tokens=2))
    scheduler.submit(Request("second", list(range(100, 104)), max_new_tokens=2))
    try:
        assert await scheduler._prefill_step()
        scheduled, queued, wall_time = publisher.publish.call_args.args
        assert scheduled.num_prefill_requests == 1
        assert scheduled.sum_prefill_tokens == 6
        assert scheduled.sum_prefill_kv_tokens == 0
        assert queued.num_prefill_requests == 1
        assert queued.sum_prefill_tokens == 4
        assert wall_time == pytest.approx(
            scheduler.timing.prefill_ms(6) / (1000 * config.speedup_ratio)
        )

        assert await scheduler._prefill_step()
        scheduled, queued, _ = publisher.publish.call_args.args
        assert scheduled.num_prefill_requests == 2
        assert scheduled.sum_prefill_tokens == 6
        assert scheduled.sum_prefill_kv_tokens == 6
        assert queued.num_prefill_requests == 0
    finally:
        await scheduler.stop()


async def test_prebuilt_kv_does_not_report_prefill_compute(server_args):
    publisher = Mock(spec=ForwardPassPublisher)
    scheduler = Scheduler(SimConfig.from_server_args(server_args()), fpm=publisher)
    request = Request("transferred", list(range(16)), 3, prebuilt=True, first_token=999)
    scheduler.submit(request)
    try:
        await scheduler._prefill_step()
        scheduled, _, _ = publisher.publish.call_args.args
        assert scheduled.num_prefill_requests == 0
        assert scheduled.sum_prefill_tokens == 0
        await scheduler._decode_step()
        scheduled, queued, _ = publisher.publish.call_args.args
        assert scheduled.num_decode_requests == 1
        assert scheduled.sum_decode_kv_tokens == 17
        assert queued.num_decode_requests == 0
    finally:
        await scheduler.stop()


async def test_cached_prefix_is_not_billed_as_prefill(server_args):
    publisher = Mock(spec=ForwardPassPublisher)
    scheduler = Scheduler(SimConfig.from_server_args(server_args(page_size=4)), fpm=publisher)
    try:
        scheduler.submit(Request("cold", list(range(16)), 1))
        await scheduler._prefill_step()
        assert publisher.publish.call_args.args[0].sum_prefill_tokens == 16
        scheduler.submit(Request("warm", list(range(16)), 1))
        await scheduler._prefill_step()
        assert publisher.publish.call_args.args[0].sum_prefill_tokens == 0
    finally:
        await scheduler.stop()


@pytest.mark.parametrize("dp_rank", [0, 3])
@pytest.mark.parametrize("decoder", ["local", "dynamo"])
async def test_zmq_frames_decode_and_idle_clears_load(server_args, fpm_endpoint, dp_rank, decoder):
    decode_frame = (
        pytest.importorskip("dynamo.common.forward_pass_metrics").decode
        if decoder == "dynamo" else decode
    )
    endpoint = fpm_endpoint
    publisher = ForwardPassPublisher(endpoint, "test-worker", dp_rank=dp_rank)
    scheduler = Scheduler(SimConfig.from_server_args(server_args()), fpm=publisher)
    context = zmq.asyncio.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(f"{endpoint}.{dp_rank}")
    try:
        await asyncio.sleep(0.1)
        scheduler.submit(Request("request", [1, 2, 3, 4], 2))
        await scheduler._prefill_step()
        await scheduler._decode_step()
        snapshots = []
        for _ in range(2):
            frames = await asyncio.wait_for(subscriber.recv_multipart(), timeout=2)
            assert len(frames) == 3
            snapshot = decode_frame(frames[2])
            assert snapshot is not None
            assert snapshot.worker_id == "test-worker"
            assert snapshot.dp_rank == dp_rank
            assert int.from_bytes(frames[1], "big") == snapshot.counter_id
            assert decode(frames[2]).counter_id == snapshot.counter_id
            snapshots.append(snapshot)
        assert snapshots[0].scheduled_requests.sum_prefill_tokens == 4
        assert snapshots[1].scheduled_requests.num_decode_requests == 1
        assert snapshots[1].counter_id > snapshots[0].counter_id

        scheduler.start()
        frames = await asyncio.wait_for(subscriber.recv_multipart(), timeout=2)
        idle = decode_frame(frames[2])
        assert idle.wall_time == 0
        assert idle.scheduled_requests.num_decode_requests == 0
        assert idle.queued_requests.num_prefill_requests == 0
        assert idle.queued_requests.num_decode_requests == 0
    finally:
        await scheduler.stop()
        subscriber.close(linger=0)
        context.term()
