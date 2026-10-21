import asyncio

import pytest

from fakeengine.config import SimConfig
from fakeengine.kvcache import BlockPool, block_hashes, external_id
from fakeengine.scheduler import Request, Scheduler
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.entrypoints.engine import Engine


async def test_blocked_cached_admission_does_not_reacquire(server_args):
    scheduler = Scheduler(SimConfig.from_server_args(server_args(
        page_size=4, fake_num_kv_blocks=4, max_prefill_tokens=4,
    )))
    scheduler.submit(Request("seed", list(range(8)), 1))
    await scheduler._prefill_step()
    await scheduler._prefill_step()
    held = scheduler.pool.allocate(2)
    request = scheduler.submit(Request("blocked", list(range(12)), 2))
    for _ in range(3):
        assert not await scheduler._prefill_step()
        assert len(request.blocks) == 2
    scheduler.pool.free(held)
    for _ in range(3):
        assert await scheduler._prefill_step()
    assert request.cached_tokens == 8
    assert scheduler.abort(request.rid)
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_exact_capacity_prompt_finishes_cold_and_cached(server_args):
    scheduler = Scheduler(SimConfig.from_server_args(server_args(
        fake_num_kv_blocks=2, page_size=4, max_prefill_tokens=4,
    )))
    for rid in ("cold", "warm"):
        request = scheduler.submit(Request(rid, list(range(8)), 1))
        for _ in range(2):
            assert await scheduler._prefill_step()
        assert request.finished
        assert request.cached_tokens == (8 if rid == "warm" else 0)
        assert scheduler.pool.can_allocate(2)


async def test_abort_during_prefill_does_not_resurrect_request(server_args):
    scheduler = Scheduler(SimConfig.from_server_args(server_args()))
    entered, resume = asyncio.Event(), asyncio.Event()

    async def pause(_):
        entered.set()
        await resume.wait()

    scheduler._sleep_step = pause
    request = scheduler.submit(Request("cancel", [1, 2, 3], 2))
    task = asyncio.create_task(scheduler._prefill_step())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert scheduler.abort(request.rid)
        resume.set()
        await asyncio.wait_for(task, 1)
        assert scheduler.running == []
        assert scheduler.by_rid == {}
        assert request.generated == []
        assert request.queue.get_nowait() == ("done", "abort")
        assert request.queue.empty()
        assert scheduler.pool.cached_blocks == 0
    finally:
        resume.set()
        await task


@pytest.mark.parametrize("mode", ["null", "decode"])
async def test_closing_stream_releases_request(server_args, mode):
    engine = Engine(server_args(disaggregation_mode=mode))
    try:
        stream = await engine.async_generate(
            input_ids=[1, 2, 3], sampling_params={"max_new_tokens": 10000},
            rid="close", bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_room=1 if mode == "decode" else None,
        )
        await asyncio.wait_for(anext(stream), 1)
        await stream.aclose()
        assert engine.scheduler.by_rid == {}
        assert engine.scheduler.running == []
        assert engine.scheduler.pool.can_allocate(engine.scheduler.pool.num_blocks)
    finally:
        await engine._async_shutdown()


def test_eviction_keeps_parents_until_children_leave():
    removed = []
    pool = BlockPool(4, 4, on_evict=removed.extend)
    hashes = block_hashes(list(range(16)), 4)
    blocks = pool.allocate(4)
    pool.insert_prefix(hashes, blocks)
    pool.free(blocks)
    allocated = pool.allocate(1)
    assert removed == [external_id(hashes[-1])]
    assert pool.match_prefix(hashes) == 3
    pool.free(allocated)
    pool.clear_cache()
    assert removed == [external_id(h) for h in reversed(hashes)]
    assert pool.cached_blocks == 0
    assert pool.used_blocks == 0


def test_eviction_preserves_shared_parent_of_other_branches():
    pool = BlockPool(4, 2)
    left = block_hashes([1, 2, 3, 4], 2)
    blocks = pool.allocate(2)
    pool.insert_prefix(left, blocks)
    pool.free(blocks)
    right = block_hashes([1, 2, 5, 6], 2)
    shared = pool.acquire_prefix(right)
    blocks = shared + pool.allocate(1)
    pool.insert_prefix(right, blocks)
    pool.free(blocks)
    pool.allocate(2)
    assert pool.match_prefix(right) == 2
    assert pool.match_prefix(left) == 1
