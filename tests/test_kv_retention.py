import asyncio

import pytest

from fakeengine.config import SimConfig
from fakeengine.scheduler import FINISH_LENGTH, Request, Scheduler
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.engine import Engine


class EventRecorder:
    def __init__(self):
        self.stored = []
        self.removed = []

    def publish_stored(self, **kwargs):
        self.stored.append(kwargs)

    def publish_removed(self, block_ids):
        self.removed.extend(block_ids)


def scheduler_for(server_args, *, clock=None, **overrides):
    settings = dict(
        page_size=4,
        fake_num_kv_blocks=8,
        fake_kv_return_retention_timeout_ms=1000,
    )
    settings.update(overrides)
    return Scheduler(
        SimConfig.from_server_args(server_args(**settings)),
        clock=clock,
    )


async def finish_prebuilt(
    scheduler,
    prompt,
    generated_tokens,
    *,
    rid="decode",
    retain=True,
    cache_salt=None,
):
    request = scheduler.submit(Request(
        rid,
        prompt,
        generated_tokens,
        prebuilt=True,
        first_token=9001,
        cache_salt=cache_salt,
        retain_response_kv=retain,
    ))
    assert await scheduler._prefill_step()
    while not request.finished:
        await scheduler._decode_step()
    return request


@pytest.mark.parametrize(
    "prompt,generated_tokens,expected_blocks",
    [
        (list(range(4)), 1, (6,)),
        (list(range(5)), 1, (6,)),
        (list(range(4)), 5, (6, 5)),
        (list(range(5)), 4, (6, 5)),
    ],
)
async def test_retention_preserves_exact_response_block_slice(
    server_args, prompt, generated_tokens, expected_blocks,
):
    scheduler = scheduler_for(server_args)
    request = await finish_prebuilt(
        scheduler, prompt, generated_tokens, cache_salt="tenant-a",
    )

    lease = request.retention_lease
    assert lease is scheduler.get_retention_lease(request.rid)
    assert lease.request_id == request.rid
    assert lease.prompt_token_count == len(prompt)
    assert lease.generated_token_count == generated_tokens
    assert lease.generated_token_ids == tuple(request._generated_token_ids)
    assert lease.block_ids == expected_blocks
    assert lease.block_count == len(expected_blocks)
    assert lease.retained_bytes == (
        len(expected_blocks)
        * scheduler.config.block_size
        * scheduler.config.kv_bytes_per_token
    )
    assert lease.cache_salt == "tenant-a"
    assert lease.deadline > lease.created_at
    assert not lease.released
    assert "generated_token_ids" not in repr(lease)


async def test_completion_is_not_delayed_and_blocks_stay_unavailable_until_release(
    server_args,
):
    scheduler = scheduler_for(server_args, fake_num_kv_blocks=4)
    request = await finish_prebuilt(scheduler, list(range(5)), 1)
    lease = request.retention_lease

    token_ids, finish_reason = await request.next_output()
    assert token_ids == [9001]
    assert finish_reason == "length"
    assert not request.output_notification_pending
    assert scheduler.stats()["kv_retention_active_leases"] == 1
    assert scheduler.stats()["kv_used_blocks"] == lease.block_count

    available = scheduler.pool.allocate(
        scheduler.pool.num_blocks - lease.block_count
    )
    assert available is not None
    assert not set(available).intersection(lease.block_ids)
    assert scheduler.pool.allocate(1) is None
    scheduler.pool.free(available)

    assert scheduler.release_retention(lease)
    assert scheduler.release_retention(lease)
    assert lease.released
    assert lease.release_reason == "released"
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert scheduler.stats()["kv_retention_active_leases"] == 0


async def test_expiry_and_scheduler_shutdown_release_every_lease(server_args):
    scheduler = scheduler_for(server_args)
    first = await finish_prebuilt(scheduler, list(range(4)), 1, rid="expiry")
    lease = first.retention_lease

    assert scheduler.expire_retention_leases(lease.deadline - 0.001) == 0
    assert not lease.released
    assert scheduler.expire_retention_leases(lease.deadline) == 1
    assert lease.released
    assert lease.release_reason == "expired"

    second = await finish_prebuilt(scheduler, list(range(5)), 1, rid="shutdown")
    await scheduler.stop()
    assert second.retention_lease.released
    assert second.retention_lease.release_reason == "abort_all"
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_scheduler_loop_expires_idle_lease_without_a_new_request(server_args):
    scheduler = scheduler_for(
        server_args,
        fake_kv_return_retention_timeout_ms=10,
    )
    scheduler.start()
    try:
        request = scheduler.submit(Request(
            "automatic-expiry",
            list(range(4)),
            1,
            prebuilt=True,
            first_token=17,
            retain_response_kv=True,
        ))
        token_ids, finish_reason = await asyncio.wait_for(
            request.next_output(),
            1,
        )
        assert token_ids == [17]
        assert finish_reason == "length"
        lease = request.retention_lease
        await asyncio.sleep(0.05)
        assert lease.released
        assert lease.release_reason == "expired"
        assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    finally:
        await scheduler.stop()


async def test_expiry_during_decode_sleep_releases_capacity_before_growth(
    server_args,
):
    now = [100.0]
    scheduler = scheduler_for(
        server_args,
        clock=lambda: now[0],
        fake_num_kv_blocks=2,
        fake_kv_return_retention_timeout_ms=100,
    )
    retained = await finish_prebuilt(
        scheduler,
        list(range(4)),
        1,
        rid="retained",
    )
    lease = retained.retention_lease
    contender = scheduler.submit(Request(
        "contender",
        list(range(4)),
        2,
        prebuilt=True,
        first_token=23,
    ))
    assert await scheduler._prefill_step()
    assert contender in scheduler.running
    assert not lease.released
    assert not scheduler.pool.can_allocate(1)

    async def cross_lease_deadline(_milliseconds):
        await asyncio.sleep(0)
        now[0] = lease.deadline

    scheduler.timing.sleep_ms = cross_lease_deadline
    await scheduler._decode_step()

    assert lease.released
    assert lease.release_reason == "expired"
    assert contender.finished
    token_ids, finish_reason = await contender.next_output()
    assert token_ids[0] == 23
    assert len(token_ids) == 2
    assert finish_reason == "length"
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_abort_cancel_and_opt_out_create_no_lease(server_args):
    scheduler = scheduler_for(server_args)
    aborted = scheduler.submit(Request(
        "aborted",
        list(range(5)),
        4,
        prebuilt=True,
        retain_response_kv=True,
    ))
    assert await scheduler._prefill_step()
    assert scheduler.abort(aborted.rid)
    assert aborted.retention_lease is None
    assert scheduler.get_retention_lease(aborted.rid) is None

    disabled = await finish_prebuilt(
        scheduler, list(range(5)), 2, rid="disabled", retain=False,
    )
    assert disabled.retention_lease is None
    assert disabled.retention_rejection_reason is None
    assert scheduler.active_retention_leases == ()
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


@pytest.mark.parametrize(
    "overrides,expected_reason",
    [
        ({"fake_kv_return_retention_max_blocks": 1}, "block_budget"),
        ({"fake_kv_return_retention_max_bytes": 1}, "byte_budget"),
    ],
)
async def test_budget_rejection_preserves_successful_response(
    server_args, overrides, expected_reason,
):
    scheduler = scheduler_for(server_args, **overrides)
    request = await finish_prebuilt(scheduler, list(range(4)), 5)

    assert request.retention_lease is None
    assert request.retention_rejection_reason == expected_reason
    token_ids, finish_reason = await request.next_output()
    assert len(token_ids) == 5
    assert finish_reason == "length"
    assert scheduler.stats()["kv_retention_rejections"] == 1
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_lease_count_budget_declines_only_the_optimization(server_args):
    scheduler = scheduler_for(
        server_args,
        fake_kv_return_retention_max_leases=1,
    )
    first = await finish_prebuilt(scheduler, list(range(4)), 1, rid="first")
    second = await finish_prebuilt(scheduler, list(range(4)), 1, rid="second")

    assert first.retention_lease is not None
    assert second.retention_lease is None
    assert second.retention_rejection_reason == "lease_budget"
    token_ids, finish_reason = await second.next_output()
    assert len(token_ids) == 1
    assert finish_reason == "length"
    scheduler.abort_all()
    assert first.retention_lease.released
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_preemption_keeps_complete_generated_ids_and_finish_is_single_shot(
    server_args,
):
    scheduler = scheduler_for(server_args)
    request = scheduler.submit(Request(
        "preempted",
        list(range(4)),
        4,
        prebuilt=True,
        retain_response_kv=True,
    ))
    assert await scheduler._prefill_step()
    first_token = request._generated_token_ids[0]
    assert scheduler._preempt_one(protect=Request("protected", [99], 1))
    assert request in scheduler.waiting
    assert await scheduler._prefill_step()
    while not request.finished:
        await scheduler._decode_step()

    lease = request.retention_lease
    assert lease.generated_token_count == 4
    assert lease.generated_token_ids[0] == first_token
    assert len(lease.generated_token_ids) == 4
    scheduler._finish(request, FINISH_LENGTH)
    assert scheduler.active_retention_leases == (lease,)
    assert scheduler.release_retention(lease)
    assert scheduler.release_retention(lease)
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_prebuilt_retention_does_not_publish_local_kv_events(server_args):
    events = EventRecorder()
    scheduler = Scheduler(
        SimConfig.from_server_args(server_args(
            page_size=4,
            fake_num_kv_blocks=8,
            fake_kv_return_retention_timeout_ms=1000,
        )),
        kv_events=events,
    )
    request = await finish_prebuilt(scheduler, list(range(5)), 4)
    lease = request.retention_lease

    assert events.stored == []
    assert events.removed == []
    scheduler.flush_cache()
    assert events.stored == []
    assert events.removed == []
    assert scheduler.release_retention(lease)
    assert events.removed == []


async def test_decode_engine_opt_in_exposes_lease_without_response_metadata(
    server_args,
):
    engine = Engine(server_args(
        disaggregation_mode="decode",
        page_size=4,
        fake_num_kv_blocks=8,
        fake_kv_return_retention_timeout_ms=10_000,
    ))
    try:
        stream = await engine.async_generate(
            input_ids=list(range(5)),
            sampling_params={"max_new_tokens": 2},
            rid="engine-retention",
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_room=7,
            cache_salt="tenant-b",
            retain_response_kv=True,
        )
        chunks = [chunk async for chunk in stream]
        lease = engine.scheduler.get_retention_lease("engine-retention")

        assert chunks[-1]["meta_info"]["finish_reason"]["type"] == "length"
        assert "retention" not in chunks[-1]["meta_info"]
        assert "kv_return" not in chunks[-1]["meta_info"]
        assert lease.generated_token_ids == tuple(
            token
            for chunk in chunks
            for token in chunk["output_ids"]
        )
        assert lease.cache_salt == "tenant-b"
    finally:
        await engine._async_shutdown()


async def test_decode_stream_close_before_terminal_delivery_releases_lease(
    server_args,
):
    engine = Engine(server_args(
        disaggregation_mode="decode",
        page_size=4,
        fake_num_kv_blocks=4,
        fake_kv_return_retention_timeout_ms=10_000,
    ))
    stream = await engine.async_generate(
        input_ids=list(range(5)),
        sampling_params={"max_new_tokens": 2},
        rid="closed-before-terminal",
        bootstrap_host=FAKE_BOOTSTRAP_HOST,
        bootstrap_room=8,
        retain_response_kv=True,
    )
    try:
        first = await asyncio.wait_for(anext(stream), 1)
        assert first["meta_info"]["finish_reason"] is None
        lease = None
        for _ in range(100):
            lease = engine.scheduler.get_retention_lease(
                "closed-before-terminal"
            )
            if lease is not None:
                break
            await asyncio.sleep(0.01)
        assert lease is not None
        assert not lease.released

        await stream.aclose()

        assert lease.released
        assert lease.release_reason == "undelivered"
        assert engine.scheduler.active_retention_leases == ()
        assert engine.scheduler.pool.can_allocate(
            engine.scheduler.pool.num_blocks
        )
    finally:
        await stream.aclose()
        await engine._async_shutdown()
