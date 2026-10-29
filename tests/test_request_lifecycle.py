import asyncio
import random
import time

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

    scheduler.timing.sleep_ms = pause
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


def _naive_victim(pool):
    """The eviction rule spelled out: the least recently used cached block
    that nothing references and no cached block names as its parent."""
    parents = set(pool._parents.values())
    for digest, block in pool._cached.items():
        if pool._refcount.get(block, 0) == 0 and digest not in parents:
            return digest
    return None


def _naive_evictable(pool):
    return sum(1 for b in pool._cached.values() if pool._refcount.get(b, 0) == 0)


def test_eviction_order_and_evictable_count_follow_the_lru_rule():
    rng = random.Random(7)
    pool = BlockPool(48, 2)
    held = []
    for _ in range(3000):
        op = rng.random()
        if op < 0.45:
            prompt = [rng.randrange(3) for _ in range(2 * rng.randint(1, 6))]
            hashes = block_hashes(prompt, 2)
            shared = pool.acquire_prefix(hashes)
            fresh = pool.allocate(len(hashes) - len(shared))
            if fresh is None:
                pool.free(shared)
                continue
            pool.insert_prefix(hashes, shared + fresh)
            held.append(shared + fresh)
        elif op < 0.85 and held:
            pool.free(held.pop(rng.randrange(len(held))))
        elif op < 0.95:
            pool.match_prefix(block_hashes([rng.randrange(3) for _ in range(6)], 2))
        else:
            expected = _naive_victim(pool)
            removed = []
            assert pool._evict_one(collect=removed) == (expected is not None)
            assert removed == ([external_id(expected)] if expected is not None else [])
        assert pool.evictable_blocks == _naive_evictable(pool)
        assert pool.used_blocks == pool.num_blocks - len(set(pool._free_clean))
    for blocks in held:
        pool.free(blocks)
    pool.clear_cache()
    assert pool.cached_blocks == 0 and pool.evictable_blocks == 0 and pool.used_blocks == 0


def test_eviction_cost_does_not_grow_with_the_cache():
    pool = BlockPool(20_000, 4)
    for i in range(200):
        hashes = block_hashes([i] * 400, 4)
        blocks = pool.allocate(len(hashes))
        pool.insert_prefix(hashes, blocks)
        pool.free(blocks)
    started = time.perf_counter()
    for i in range(100):
        hashes = block_hashes([10**6 + i] * 400, 4)
        blocks = pool.allocate(len(hashes))
        pool.insert_prefix(hashes, blocks)
        pool.free(blocks)
        assert pool.can_allocate(1)
    # 10,000 evictions out of a full 20,000-block cache; a full scan per eviction takes seconds.
    assert time.perf_counter() - started < 1.0
