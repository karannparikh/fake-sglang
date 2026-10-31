"""Behavior of the simulator itself: tokens, KV accounting, scheduling."""

from __future__ import annotations

import asyncio

import pytest

from fakeengine import tokens as token_gen
from fakeengine.config import SimConfig
from fakeengine.kvcache import BlockPool, block_hashes
from fakeengine.scheduler import Request, Scheduler
from fakeengine.timing import TimingModel


def _config(server_args, **overrides) -> SimConfig:
    config = SimConfig.from_server_args(server_args(**overrides))
    return config


# -- tokens ----------------------------------------------------------------


def test_token_stream_is_deterministic_per_prompt():
    a = token_gen.take([1, 2, 3], 32000, 16)
    b = token_gen.take([1, 2, 3], 32000, 16)
    assert a == b


def test_different_prompts_give_different_tokens():
    assert token_gen.take([1, 2, 3], 32000, 16) != token_gen.take([1, 2, 4], 32000, 16)


def test_tokens_avoid_the_special_id_range():
    # Emitting a BOS/EOS id would terminate the frontend's stream early.
    assert all(t >= 256 for t in token_gen.take([7], 32000, 200))


def test_handoff_token_is_stable_and_in_vocab():
    first = token_gen.handoff_token("req-abc", 1000)
    assert first == token_gen.handoff_token("req-abc", 1000)
    assert 0 <= first < 1000


# -- KV block pool ---------------------------------------------------------


def test_cumulative_hashes_distinguish_prefix_from_suffix():
    shared_prefix = block_hashes([1, 2, 3, 4], 2)
    other_prefix = block_hashes([9, 9, 3, 4], 2)
    assert shared_prefix[0] != other_prefix[0]
    # Same trailing block, different history => different cumulative hash.
    assert shared_prefix[1] != other_prefix[1]


def test_prefix_match_stops_at_first_miss():
    pool = BlockPool(num_blocks=8, block_size=2)
    hashes = block_hashes([1, 2, 3, 4, 5, 6], 2)
    blocks = pool.allocate(3)
    pool.insert_prefix(hashes, blocks)
    pool.free(blocks)

    diverging = block_hashes([1, 2, 99, 100, 5, 6], 2)
    assert pool.match_prefix(diverging) == 1


def test_allocation_fails_only_when_referenced_blocks_fill_the_pool():
    pool = BlockPool(num_blocks=4, block_size=1)
    held = pool.allocate(4)
    assert pool.allocate(1) is None          # all four are referenced
    pool.free(held)
    assert pool.allocate(4) is not None


def test_cached_but_unreferenced_blocks_are_evictable():
    pool = BlockPool(num_blocks=4, block_size=1)
    hashes = block_hashes([1, 2, 3, 4], 1)
    blocks = pool.allocate(4)
    pool.insert_prefix(hashes, blocks)
    pool.free(blocks)
    # Cached, but nothing references them, so a new request may take them.
    assert pool.allocate(4) is not None


# -- timing ----------------------------------------------------------------


def test_prefill_scales_with_input_length(server_args):
    timing = TimingModel(_config(server_args))
    assert timing.prefill_ms(2000) > timing.prefill_ms(1000) > timing.prefill_ms(0)


@pytest.mark.parametrize(
    ("batch_size", "expected_ms"),
    [
        (1, 3.6),
        (2, 4.1),
        (4, 6.2),
        (8, 7.5),
        (16, 8.6),
        (24, 9.8),
        (32, 10.4),
    ],
)
def test_default_disaggregated_decode_profile_matches_engine_only_p50(
    server_args,
    batch_size,
    expected_ms,
):
    timing = TimingModel(_config(
        server_args,
        disaggregation_mode="decode",
    ))
    assert timing.decode_step_ms(batch_size) == pytest.approx(expected_ms)


def test_decode_profile_interpolates_and_honors_adjustments(server_args):
    timing = TimingModel(_config(
        server_args,
        disaggregation_mode="decode",
    ))
    adjusted = TimingModel(_config(
        server_args,
        disaggregation_mode="decode",
        fake_itl_ms=5.6,
        fake_itl_ms_per_running_req=0.1,
    ))

    assert timing.decode_step_ms(3) == pytest.approx(5.15)
    assert adjusted.decode_step_ms(4) == pytest.approx(8.5)


@pytest.mark.parametrize(
    ("representative_tokens", "expected_ms"),
    [
        (512, 350),
        (2_560, 430),
        (6_144, 580),
        (12_288, 990),
        (24_576, 1_570),
        (49_152, 2_790),
        (98_304, 5_130),
        (196_608, 10_600),
        (393_216, 25_700),
    ],
)
def test_default_disaggregated_prefill_fit_tracks_no_cache_p50(
    server_args,
    representative_tokens,
    expected_ms,
):
    timing = TimingModel(_config(
        server_args,
        disaggregation_mode="prefill",
    ))
    assert timing.prefill_ms(representative_tokens) == pytest.approx(
        expected_ms,
        rel=0.18,
    )


def test_auto_timing_profile_separates_aggregated_and_disaggregated(server_args):
    aggregated = _config(server_args)
    prefill = _config(server_args, disaggregation_mode="prefill")
    decode = _config(server_args, disaggregation_mode="decode")

    assert aggregated.timing_profile == "aggregated"
    assert prefill.timing_profile == decode.timing_profile == "disaggregated"
    assert TimingModel(aggregated).prefill_ms(1000) == pytest.approx(59.0)
    assert TimingModel(prefill).prefill_ms(1000) == pytest.approx(375.0)
    assert TimingModel(aggregated).decode_step_ms(32) == pytest.approx(19.85)
    assert TimingModel(decode).decode_step_ms(32) == pytest.approx(10.4)


def test_explicit_timing_profile_and_knobs_override_role_defaults(server_args):
    timing = TimingModel(_config(
        server_args,
        disaggregation_mode="decode",
        fake_timing_profile="aggregated",
        fake_prefill_ms_per_1k_tokens=10,
        fake_prefill_overhead_ms=2,
        fake_itl_ms=4,
        fake_itl_ms_per_running_req=0.5,
    ))

    assert timing.config.timing_profile == "aggregated"
    assert timing.prefill_ms(2000) == pytest.approx(22)
    assert timing.decode_step_ms(4) == pytest.approx(5.5)


def test_decode_step_degrades_with_batch_size(server_args):
    timing = TimingModel(_config(server_args))
    assert timing.decode_step_ms(8) > timing.decode_step_ms(1)


def test_transfer_time_is_inversely_proportional_to_bandwidth(server_args):
    fast = TimingModel(_config(server_args, fake_kv_bandwidth_gb_s=64.0))
    slow = TimingModel(_config(server_args, fake_kv_bandwidth_gb_s=8.0))
    overhead = fast.config.kv_transfer_overhead_ms
    fast_payload = fast.transfer_ms(10_000) - overhead
    slow_payload = slow.transfer_ms(10_000) - overhead
    assert slow_payload == pytest.approx(fast_payload * 8, rel=1e-6)


# -- scheduler -------------------------------------------------------------


async def _drain(request: Request) -> tuple:
    tokens, reason = [], None
    while True:
        output_ids, reason = await request.next_output()
        tokens.extend(output_ids)
        if reason is not None:
            return tokens, reason


async def test_scheduler_generates_the_requested_number_of_tokens(server_args):
    scheduler = Scheduler(_config(server_args))
    scheduler.start()
    request = scheduler.submit(Request("r1", [1, 2, 3], max_new_tokens=5))
    tokens, reason = await _drain(request)
    assert len(tokens) == 5
    assert reason == "length"
    await scheduler.stop()


async def test_scheduler_matches_the_standalone_token_stream(server_args):
    config = _config(server_args)
    scheduler = Scheduler(config)
    scheduler.start()
    prompt = [5, 6, 7]
    request = scheduler.submit(Request("r1", prompt, max_new_tokens=4))
    tokens, _ = await _drain(request)
    assert tokens == token_gen.take(prompt, config.vocab_size, 4)
    await scheduler.stop()


async def test_prebuilt_request_leads_with_the_inherited_token(server_args):
    """A disaggregated decode continues prefill's answer instead of restarting it."""
    config = _config(server_args)
    scheduler = Scheduler(config)
    scheduler.start()
    prompt = [5, 6, 7]
    aggregated = token_gen.take(prompt, config.vocab_size, 4)

    request = scheduler.submit(
        Request("r1", prompt, max_new_tokens=4, prebuilt=True, first_token=aggregated[0])
    )
    tokens, _ = await _drain(request)
    assert tokens == aggregated
    await scheduler.stop()


async def test_prebuilt_request_reports_the_prompt_as_cached(server_args):
    scheduler = Scheduler(_config(server_args))
    scheduler.start()
    prompt = list(range(64))
    request = scheduler.submit(
        Request("r1", prompt, max_new_tokens=2, prebuilt=True, first_token=999)
    )
    await _drain(request)
    assert request.cached_tokens == len(prompt)
    await scheduler.stop()


async def test_repeated_prompt_reports_a_prefix_cache_hit(server_args):
    config = _config(server_args)
    scheduler = Scheduler(config)
    scheduler.start()
    prompt = list(range(100, 164))

    first = scheduler.submit(Request("r1", prompt, max_new_tokens=2))
    await _drain(first)
    assert first.cached_tokens == 0

    second = scheduler.submit(Request("r2", prompt, max_new_tokens=2))
    await _drain(second)
    assert second.cached_tokens == len(prompt)
    await scheduler.stop()


async def test_cache_salt_isolates_the_prefix_cache(server_args):
    """Only requests with the same salt share KV, as in SGLang: an identical
    prompt under another salt, or under none, is recomputed."""
    scheduler = Scheduler(_config(server_args))
    scheduler.start()
    prompt = list(range(300, 364))

    async def cached(rid, salt):
        request = scheduler.submit(Request(rid, prompt, max_new_tokens=2, cache_salt=salt))
        await _drain(request)
        return request.cached_tokens

    assert await cached("a1", "tenant-a") == 0
    assert await cached("a2", "tenant-a") == len(prompt)
    assert await cached("b1", "tenant-b") == 0
    assert await cached("n1", None) == 0
    assert await cached("n2", "") == len(prompt)      # "" is no salt
    await scheduler.stop()


async def test_disable_radix_cache_turns_off_reuse_and_events(server_args):
    """``--disable-radix-cache``: every prompt is recomputed, and the router is
    never told a prefix is cached, as in SGLang."""

    class Recorder:
        def __init__(self):
            self.stored = []

        def publish_stored(self, **kwargs):
            self.stored.append(kwargs)

        def publish_removed(self, block_ids):
            pass

    events = Recorder()
    scheduler = Scheduler(_config(server_args, disable_radix_cache=True), kv_events=events)
    scheduler.start()
    prompt = list(range(100, 164))
    for rid in ("r1", "r2"):
        request = scheduler.submit(Request(rid, prompt, max_new_tokens=2))
        await _drain(request)
        assert request.cached_tokens == 0
    assert events.stored == []
    await scheduler.stop()


async def test_batched_requests_all_complete_under_a_running_limit(server_args):
    """Continuous batching must not drop or stall requests past the limit."""
    scheduler = Scheduler(_config(server_args, max_running_requests=2))
    scheduler.start()
    requests = [
        scheduler.submit(Request(f"r{i}", [i, i + 1, i + 2], max_new_tokens=3))
        for i in range(6)
    ]
    results = await asyncio.gather(*(_drain(r) for r in requests))
    assert [len(tokens) for tokens, _ in results] == [3] * 6
    await scheduler.stop()


async def test_requests_survive_kv_pressure_by_preemption(server_args):
    """A pool far too small for the batch must still finish every request."""
    scheduler = Scheduler(
        _config(server_args, fake_num_kv_blocks=64, max_running_requests=8)
    )
    scheduler.start()
    requests = [
        scheduler.submit(Request(f"r{i}", list(range(i, i + 8)), max_new_tokens=6))
        for i in range(6)
    ]
    results = await asyncio.gather(*(_drain(r) for r in requests))
    assert all(reason == "length" for _, reason in results)
    assert all(len(tokens) == 6 for tokens, _ in results)
    await scheduler.stop()


async def test_abort_finishes_the_request_promptly(server_args):
    scheduler = Scheduler(_config(server_args))
    scheduler.start()
    request = scheduler.submit(Request("r1", [1, 2, 3], max_new_tokens=10_000))
    await asyncio.sleep(0.05)
    assert scheduler.abort("r1") is True
    _, reason = await _drain(request)
    assert reason == "abort"
    await scheduler.stop()


async def test_finished_requests_return_their_kv_blocks(server_args):
    scheduler = Scheduler(_config(server_args))
    scheduler.start()
    request = scheduler.submit(Request("r1", list(range(32)), max_new_tokens=3))
    await _drain(request)
    # Blocks may stay cached for prefix reuse, but none may stay referenced.
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    await scheduler.stop()


# -- block pool: allocation policy the router depends on --------------------


def test_new_content_does_not_evict_a_cached_prefix_while_clean_blocks_remain():
    """Clean-first allocation.

    Taking a cached block while untouched ones are free would discard a live
    prefix on every new request -- and, because eviction is published, would
    tell the router a prefix vanished moments after it appeared.
    """
    removed: list = []
    pool = BlockPool(num_blocks=8, block_size=1, on_evict=removed.extend)

    first = block_hashes([1, 2], 1)
    blocks = pool.allocate(2)
    pool.insert_prefix(first, blocks)
    pool.free(blocks)

    pool.allocate(2)                       # different content, plenty of room
    assert pool.match_prefix(first) == 2
    assert removed == []


def test_eviction_is_least_recently_used():
    pool = BlockPool(num_blocks=2, block_size=1)
    older = block_hashes([1], 1)
    newer = block_hashes([2], 1)

    a = pool.allocate(1)
    pool.insert_prefix(older, a)
    pool.free(a)
    b = pool.allocate(1)
    pool.insert_prefix(newer, b)
    pool.free(b)

    pool.match_prefix(newer)               # touch it, making `older` the LRU
    pool.allocate(1)                       # forces exactly one eviction
    assert pool.match_prefix(older) == 0
    assert pool.match_prefix(newer) == 1


def test_acquiring_a_prefix_costs_no_new_blocks():
    """A cache hit must actually save memory, not just report a number."""
    pool = BlockPool(num_blocks=8, block_size=1)
    hashes = block_hashes([1, 2, 3, 4], 1)
    blocks = pool.allocate(4)
    pool.insert_prefix(hashes, blocks)
    pool.free(blocks)

    before = pool.used_blocks
    shared = pool.acquire_prefix(hashes)
    assert shared == blocks
    assert pool.used_blocks == before      # shared, not re-allocated


def test_acquired_blocks_cannot_be_evicted_underneath_a_request():
    pool = BlockPool(num_blocks=4, block_size=1)
    hashes = block_hashes([1, 2, 3, 4], 1)
    blocks = pool.allocate(4)
    pool.insert_prefix(hashes, blocks)
    pool.free(blocks)

    held = pool.acquire_prefix(hashes)
    assert len(held) == 4
    assert pool.allocate(1) is None        # every block is referenced again
    assert pool.match_prefix(hashes) == 4


def test_a_failed_allocation_strands_no_blocks():
    pool = BlockPool(num_blocks=3, block_size=1)
    held = pool.allocate(3)
    assert pool.allocate(2) is None        # cannot be satisfied
    pool.free(held)
    assert pool.allocate(3) is not None    # nothing was lost on the way out
