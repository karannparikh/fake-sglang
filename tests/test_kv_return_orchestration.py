from __future__ import annotations

import asyncio
from copy import deepcopy
import time

import pytest

from fakeengine.bootstrap import KVReturnTransportError
from fakeengine.kvcache import block_hashes
from sglang import Engine
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST


PROMPT = list(range(8))


class RecordingEvents:
    def __init__(self):
        self.stored = []
        self.removed = []

    def publish_stored(self, **kwargs):
        self.stored.append(kwargs)

    def publish_removed(self, block_ids):
        self.removed.extend(block_ids)

    def shutdown(self):
        return None


def _args(port, mode, **overrides):
    settings = {
        "disaggregation_mode": mode,
        "disaggregation_bootstrap_port": port,
        "enable_kv_return": True,
        "fake_kv_return_max_tokens": 8,
        "fake_kv_return_deadline_ms": 500,
        "fake_kv_return_bytes_per_token": 32,
        "fake_kv_return_retention_max_leases": 4,
        "page_size": 4,
        "fake_num_kv_blocks": 16,
    }
    settings.update(overrides)
    return settings


async def _collect(stream):
    chunks = [chunk async for chunk in stream]
    tokens = [
        token
        for chunk in chunks
        for token in chunk["output_ids"]
    ]
    return tokens, chunks


async def _prefill(
    engine,
    port,
    room,
    *,
    prompt=PROMPT,
    cache_salt=None,
    rank=None,
):
    stream = await engine.async_generate(
        input_ids=prompt,
        sampling_params={"max_new_tokens": 1},
        stream=True,
        rid=f"prefill-{room}",
        bootstrap_host="127.0.0.1",
        bootstrap_port=port,
        bootstrap_room=room,
        cache_salt=cache_salt,
        routed_dp_rank=rank,
    )
    return await _collect(stream)


async def _decode(
    engine,
    port,
    room,
    *,
    prompt=PROMPT,
    max_new_tokens=4,
    cache_salt=None,
    rank=None,
):
    stream = await engine.async_generate(
        input_ids=prompt,
        sampling_params={"max_new_tokens": max_new_tokens},
        stream=True,
        rid=f"decode-{room}",
        bootstrap_host="127.0.0.1",
        bootstrap_port=port,
        bootstrap_room=room,
        cache_salt=cache_salt,
        routed_dp_rank=rank,
    )
    return await _collect(stream)


async def _wait_for_returns(engine, timeout=2):
    async def drain():
        while engine._kv_return_tasks:
            await asyncio.gather(
                *tuple(engine._kv_return_tasks),
                return_exceptions=True,
            )
            await asyncio.sleep(0)

    await asyncio.wait_for(drain(), timeout)


async def test_end_to_end_return_extends_original_prefill_and_next_turn_hits(
    server_args,
    unused_port,
):
    salt = "tenant-a"
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill"),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode"),
    ))
    aggregate = Engine(server_args=server_args(
        page_size=4,
        fake_num_kv_blocks=16,
    ))
    events = RecordingEvents()
    try:
        aggregate_stream = await aggregate.async_generate(
            input_ids=PROMPT,
            sampling_params={"max_new_tokens": 4},
            cache_salt=salt,
        )
        expected, _ = await _collect(aggregate_stream)

        await _prefill(
            prefill,
            unused_port,
            3001,
            cache_salt=salt,
        )
        prefill.scheduler.kv_events = events
        actual, chunks = await _decode(
            decode,
            unused_port,
            3001,
            cache_salt=salt,
        )

        assert actual == expected
        assert chunks[-1]["meta_info"]["kv_return"] == {
            "status": "scheduled",
            "num_tokens": 4,
            "kv_bytes": 128,
        }
        await _wait_for_returns(decode)

        returned = block_hashes(PROMPT + actual, 4, salt)
        assert prefill.scheduler.pool.match_prefix(returned) == 3
        assert len(events.stored) == 1
        assert events.stored[0]["token_ids"] == tuple(PROMPT + actual)
        assert events.stored[0]["new_indices"] == [2]

        next_stream = await prefill.async_generate(
            input_ids=PROMPT + actual,
            sampling_params={"max_new_tokens": 1},
            cache_salt=salt,
        )
        _, next_chunks = await _collect(next_stream)
        assert next_chunks[-1]["meta_info"]["cached_tokens"] == 12

        stats = decode.scheduler.stats()
        assert stats["kv_return_attempts"] == 1
        assert stats["kv_return_stored"] == 1
        assert stats["kv_return_tokens"] == 4
        assert stats["kv_return_bytes"] == 128
        assert stats["kv_return_transfer_ms"] > 0
        assert stats["kv_return_commit_ms"] >= 0
        assert stats["kv_return_active_tasks"] == 0
        assert stats["kv_retention_blocks"] == 0
    finally:
        await aggregate._async_shutdown()
        await decode._async_shutdown()
        await prefill._async_shutdown()


async def test_terminal_response_precedes_delayed_return_and_race_may_miss(
    server_args,
    unused_port,
):
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill"),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode"),
    ))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_transfer(_seconds):
        entered.set()
        await release.wait()

    decode._kv_return_sleep = delayed_transfer
    try:
        await _prefill(prefill, unused_port, 3002)
        started = time.monotonic()
        response, chunks = await asyncio.wait_for(
            _decode(decode, unused_port, 3002),
            timeout=1,
        )
        elapsed = time.monotonic() - started

        assert len(response) == 4
        assert elapsed < 0.5
        assert chunks[-1]["meta_info"]["finish_reason"]["type"] == "length"
        assert chunks[-1]["meta_info"]["kv_return"]["status"] == "scheduled"
        await asyncio.wait_for(entered.wait(), 1)
        assert decode.scheduler.stats()["kv_return_active_tasks"] == 1
        assert decode.scheduler.stats()["kv_retention_blocks"] > 0
        [state] = decode._kv_return_tasks.values()
        assert time.monotonic() < state.claim.deadline
        assert decode.scheduler.expire_retention_leases(
            now=state.lease.deadline,
        ) == 0
        decode.scheduler.abort_all()
        assert decode.scheduler.owns_retention_claim(state.claim)
        assert not state.lease.released
        assert decode.scheduler.stats()["kv_return_active_tasks"] == 1
        assert decode.scheduler.stats()["kv_retention_blocks"] == (
            state.lease.block_count
        )

        immediate_stream = await prefill.async_generate(
            input_ids=PROMPT + response,
            sampling_params={"max_new_tokens": 1},
        )
        _, immediate = await _collect(immediate_stream)
        assert immediate[-1]["meta_info"]["cached_tokens"] == len(PROMPT)

        release.set()
        await _wait_for_returns(decode)
        assert decode.scheduler.stats()["kv_return_stored"] == 1
        assert decode.scheduler.active_retention_leases == ()
        eventual_stream = await prefill.async_generate(
            input_ids=PROMPT + response,
            sampling_params={"max_new_tokens": 1},
        )
        _, eventual = await _collect(eventual_stream)
        assert eventual[-1]["meta_info"]["cached_tokens"] == (
            len(PROMPT) + len(response)
        )
    finally:
        release.set()
        await decode._async_shutdown()
        await prefill._async_shutdown()


async def test_retention_task_deadline_aborts_before_slow_transfer_commit(
    server_args,
    unused_port,
):
    settings = {
        "fake_kv_return_retention_timeout_ms": 50,
        "fake_kv_return_overhead_ms": 200,
        "fake_speedup_ratio": 1,
    }
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill", **settings),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode", **settings),
    ))
    events = RecordingEvents()
    commit_calls = 0
    original_commit = prefill.scheduler.kv_return.commit

    def record_commit(commit):
        nonlocal commit_calls
        commit_calls += 1
        return original_commit(commit)

    prefill.scheduler.kv_return.commit = record_commit
    try:
        await _prefill(prefill, unused_port, 3012)
        prefill.scheduler.kv_events = events
        response, chunks = await _decode(decode, unused_port, 3012)

        assert chunks[-1]["meta_info"]["kv_return"]["status"] == "scheduled"
        await _wait_for_returns(decode)

        stats = decode.scheduler.stats()
        assert commit_calls == 0
        assert events.stored == []
        assert prefill.scheduler.pool.match_prefix(
            block_hashes(PROMPT + response, 4)
        ) == 2
        assert prefill.scheduler.kv_return._attempts == {}
        assert prefill.scheduler.stats()["kv_used_blocks"] == 2
        assert stats["kv_return_stored"] == 0
        assert stats["kv_return_tokens"] == 0
        assert stats["kv_return_failed_by_reason"] == {
            "transfer_timeout": 1,
        }
        assert stats["kv_return_active_tasks"] == 0
        assert stats["kv_retention_active_leases"] == 0
        assert stats["kv_retention_blocks"] == 0
        assert decode.scheduler.pool.can_allocate(
            decode.scheduler.pool.num_blocks
        )
    finally:
        await decode._async_shutdown()
        await prefill._async_shutdown()


@pytest.mark.parametrize(
    ("prefill_overrides", "decode_overrides", "mutate_ticket"),
    [
        ({"enable_kv_return": False}, {}, False),
        ({}, {"enable_kv_return": False}, False),
        ({}, {"disable_radix_cache": True}, False),
        ({}, {}, True),
    ],
    ids=[
        "no-ticket",
        "decode-disabled",
        "decode-radix-disabled",
        "invalid-ticket",
    ],
)
async def test_ineligible_requests_preserve_response_without_return(
    server_args,
    unused_port,
    prefill_overrides,
    decode_overrides,
    mutate_ticket,
):
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill", **prefill_overrides),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode", **decode_overrides),
    ))
    try:
        await _prefill(prefill, unused_port, 3003)
        if mutate_ticket:
            metadata = prefill._bootstrap_server._rooms[3003]
            bad_ticket = deepcopy(metadata["kv_return"])
            bad_ticket["cache_salt"] = "other-tenant"
            metadata["kv_return"] = bad_ticket

        tokens, chunks = await _decode(decode, unused_port, 3003)
        assert len(tokens) == 4
        assert chunks[-1]["meta_info"]["finish_reason"]["type"] == "length"
        assert "kv_return" not in chunks[-1]["meta_info"]
        assert decode._kv_return_tasks == {}
        assert decode.scheduler.active_retention_leases == ()
        assert decode.scheduler.pool.can_allocate(
            decode.scheduler.pool.num_blocks
        )
        assert decode.scheduler.stats()["kv_return_rejected_by_reason"] == (
            {"invalid_ticket": 1} if mutate_ticket else {}
        )
    finally:
        await decode._async_shutdown()
        await prefill._async_shutdown()


async def test_sentinel_and_aggregated_paths_do_not_return(server_args):
    decode = Engine(server_args=server_args(
        **_args(8998, "decode"),
    ))
    aggregate = Engine(server_args=server_args(
        enable_kv_return=True,
        page_size=4,
        fake_num_kv_blocks=16,
    ))
    try:
        sentinel_stream = await decode.async_generate(
            input_ids=PROMPT,
            sampling_params={"max_new_tokens": 2},
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_room=1,
        )
        sentinel, sentinel_chunks = await _collect(sentinel_stream)
        aggregate_stream = await aggregate.async_generate(
            input_ids=PROMPT,
            sampling_params={"max_new_tokens": 2},
        )
        aggregated, aggregate_chunks = await _collect(aggregate_stream)

        assert len(sentinel) == len(aggregated) == 2
        assert "kv_return" not in sentinel_chunks[-1]["meta_info"]
        assert "kv_return" not in aggregate_chunks[-1]["meta_info"]
        assert decode.scheduler.stats()["kv_return_attempts"] == 0
        assert aggregate.scheduler.stats()["kv_return_attempts"] == 0
    finally:
        await aggregate._async_shutdown()
        await decode._async_shutdown()


async def test_cache_salt_isolation_survives_return(server_args, unused_port):
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill"),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode"),
    ))
    try:
        await _prefill(
            prefill,
            unused_port,
            3004,
            cache_salt="tenant-a",
        )
        response, _ = await _decode(
            decode,
            unused_port,
            3004,
            cache_salt="tenant-a",
        )
        await _wait_for_returns(decode)

        same_stream = await prefill.async_generate(
            input_ids=PROMPT + response,
            sampling_params={"max_new_tokens": 1},
            cache_salt="tenant-a",
        )
        _, same = await _collect(same_stream)
        other_stream = await prefill.async_generate(
            input_ids=PROMPT + response,
            sampling_params={"max_new_tokens": 1},
            cache_salt="tenant-b",
        )
        _, other = await _collect(other_stream)

        assert same[-1]["meta_info"]["cached_tokens"] == 12
        assert other[-1]["meta_info"]["cached_tokens"] == 0
    finally:
        await decode._async_shutdown()
        await prefill._async_shutdown()


@pytest.mark.parametrize(
    "failure",
    ["prepare-rejection", "timeout", "transport"],
)
async def test_background_failures_release_lease_and_preserve_response(
    server_args,
    unused_port,
    failure,
):
    blocks = 3 if failure == "prepare-rejection" else 16
    deadline = 30 if failure == "timeout" else 500
    prefill = Engine(server_args=server_args(
        **_args(
            unused_port,
            "prefill",
            fake_num_kv_blocks=blocks,
            fake_kv_return_deadline_ms=deadline,
        ),
    ))
    decode = Engine(server_args=server_args(
        **_args(
            unused_port,
            "decode",
            fake_kv_return_deadline_ms=deadline,
        ),
    ))
    held = None
    try:
        await _prefill(prefill, unused_port, 3005)
        if failure == "prepare-rejection":
            held = prefill.scheduler.pool.allocate_clean(1)
            assert held is not None
        elif failure == "timeout":
            async def never_finishes(_seconds):
                await asyncio.Event().wait()

            decode._kv_return_sleep = never_finishes
        else:
            async def transport_error(*_args, **_kwargs):
                raise KVReturnTransportError("simulated")

            decode._bootstrap_client.prepare_kv_return = transport_error

        tokens, chunks = await _decode(decode, unused_port, 3005)
        assert len(tokens) == 4
        assert chunks[-1]["meta_info"]["finish_reason"]["type"] == "length"
        assert chunks[-1]["meta_info"]["kv_return"]["status"] == "scheduled"
        await _wait_for_returns(decode)

        stats = decode.scheduler.stats()
        assert stats["kv_return_active_tasks"] == 0
        assert stats["kv_retention_blocks"] == 0
        assert decode.scheduler.pool.can_allocate(
            decode.scheduler.pool.num_blocks
        )
        if failure == "prepare-rejection":
            assert stats["kv_return_rejected_by_reason"] == {
                "prepare_capacity_rejected": 1,
            }
        elif failure == "timeout":
            assert stats["kv_return_failed_by_reason"] == {
                "transfer_timeout": 1,
            }
        else:
            assert stats["kv_return_failed_by_reason"] == {
                "transport": 1,
            }
    finally:
        if held is not None:
            prefill.scheduler.pool.free(held)
        await decode._async_shutdown()
        await prefill._async_shutdown()


async def test_close_before_terminal_schedules_nothing(server_args, unused_port):
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill"),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode"),
    ))
    try:
        await _prefill(prefill, unused_port, 3006)
        stream = await decode.async_generate(
            input_ids=PROMPT,
            sampling_params={"max_new_tokens": 4},
            rid="cancel-before-terminal",
            bootstrap_host="127.0.0.1",
            bootstrap_port=unused_port,
            bootstrap_room=3006,
        )
        first = await asyncio.wait_for(anext(stream), 1)
        assert first["meta_info"]["finish_reason"] is None
        await stream.aclose()

        assert decode.scheduler.stats()["kv_return_attempts"] == 0
        assert decode._kv_return_tasks == {}
        assert decode.scheduler.active_retention_leases == ()
        assert prefill.scheduler.pool.match_prefix(
            block_hashes(PROMPT + first["output_ids"], 4)
        ) == 2
    finally:
        await decode._async_shutdown()
        await prefill._async_shutdown()


async def test_disconnect_after_terminal_still_runs_bounded_return(
    server_args,
    unused_port,
):
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill"),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode"),
    ))
    try:
        await _prefill(prefill, unused_port, 3007)
        stream = await decode.async_generate(
            input_ids=PROMPT,
            sampling_params={"max_new_tokens": 1},
            rid="disconnect-after-terminal",
            bootstrap_host="127.0.0.1",
            bootstrap_port=unused_port,
            bootstrap_room=3007,
        )
        terminal = await asyncio.wait_for(anext(stream), 1)
        assert terminal["meta_info"]["finish_reason"]["type"] == "length"
        assert terminal["meta_info"]["kv_return"]["status"] == "scheduled"
        await stream.aclose()
        await _wait_for_returns(decode)

        assert decode.scheduler.stats()["kv_return_stored"] == 1
        assert decode.scheduler.active_retention_leases == ()
    finally:
        await decode._async_shutdown()
        await prefill._async_shutdown()


async def test_task_bound_rejects_lower_priority_return(server_args, unused_port):
    settings = {
        "max_running_requests": 1,
        "fake_kv_return_retention_max_leases": 2,
    }
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill", **settings),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode", **settings),
    ))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked(_seconds):
        entered.set()
        await release.wait()

    decode._kv_return_sleep = blocked
    try:
        await _prefill(prefill, unused_port, 3008)
        _, first = await _decode(decode, unused_port, 3008)
        assert first[-1]["meta_info"]["kv_return"]["status"] == "scheduled"
        await asyncio.wait_for(entered.wait(), 1)

        await _prefill(prefill, unused_port, 3009)
        tokens, second = await _decode(decode, unused_port, 3009)
        assert len(tokens) == 4
        assert "kv_return" not in second[-1]["meta_info"]
        stats = decode.scheduler.stats()
        assert stats["kv_return_attempts"] == 1
        assert stats["kv_return_rejected_by_reason"] == {"task_limit": 1}
        assert stats["kv_return_active_tasks"] == 1
        assert len(decode.scheduler.active_retention_leases) == 1

        release.set()
        await _wait_for_returns(decode)
        assert decode.scheduler.active_retention_leases == ()
    finally:
        release.set()
        await decode._async_shutdown()
        await prefill._async_shutdown()


async def test_packed_return_targets_ticket_prefill_rank(
    server_args,
    unused_port,
):
    packed = {
        "dp_size": 2,
        "enable_dp_attention": True,
        "max_running_requests": 4,
    }
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill", **packed),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode", **packed),
    ))
    try:
        await _prefill(prefill, unused_port, 3010, rank=1)
        response, terminal = await _decode(
            decode,
            unused_port,
            3010,
            rank=0,
        )
        assert terminal[-1]["meta_info"]["kv_return"]["status"] == "scheduled"
        await _wait_for_returns(decode)

        hashes = block_hashes(PROMPT + response, 4)
        assert prefill.schedulers[1].pool.match_prefix(hashes) == 3
        assert prefill.schedulers[0].pool.match_prefix(hashes) == 0
    finally:
        await decode._async_shutdown()
        await prefill._async_shutdown()


async def test_shutdown_aborts_prepared_attempt_and_releases_all_leases(
    server_args,
    unused_port,
):
    prefill = Engine(server_args=server_args(
        **_args(unused_port, "prefill"),
    ))
    decode = Engine(server_args=server_args(
        **_args(unused_port, "decode"),
    ))
    entered = asyncio.Event()

    async def blocked(_seconds):
        entered.set()
        await asyncio.Event().wait()

    decode._kv_return_sleep = blocked
    try:
        await _prefill(prefill, unused_port, 3011)
        tokens, terminal = await _decode(decode, unused_port, 3011)
        assert len(tokens) == 4
        assert terminal[-1]["meta_info"]["kv_return"]["status"] == "scheduled"
        await asyncio.wait_for(entered.wait(), 1)
        assert prefill.scheduler.kv_return._attempts
        assert decode.scheduler.active_retention_leases

        await decode._async_shutdown()

        assert decode._kv_return_tasks == {}
        assert decode.scheduler.active_retention_leases == ()
        assert decode.scheduler.stats()["kv_return_active_tasks"] == 0
        assert prefill.scheduler.kv_return._attempts == {}
        assert decode._bootstrap_client._session.closed
    finally:
        await decode._async_shutdown()
        await prefill._async_shutdown()
