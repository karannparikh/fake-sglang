"""Disaggregated prefill -> KV transfer -> decode, in-process.

These drive the engine directly rather than through Dynamo, so they assert the
handoff contract itself: the room really crosses a socket, the transfer really
costs time proportional to the payload, and the answer is the same one an
aggregated worker would have produced.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from sglang import Engine
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST, KVPoll

PROMPT = list(range(512))
BOOTSTRAP_PORT = 19010


async def _collect(stream) -> tuple:
    tokens, meta = [], None
    async for chunk in stream:
        tokens.extend(chunk["output_ids"])
        meta = chunk["meta_info"]
    return tokens, meta


async def _prefill_then_decode(prefill, decode, room, *, max_new_tokens):
    stream = await prefill.async_generate(
        input_ids=PROMPT,
        sampling_params={"max_new_tokens": 1},
        stream=True,
        bootstrap_host="127.0.0.1",
        bootstrap_port=BOOTSTRAP_PORT,
        bootstrap_room=room,
    )
    prefill_tokens, _ = await _collect(stream)

    stream = await decode.async_generate(
        input_ids=PROMPT,
        sampling_params={"max_new_tokens": max_new_tokens},
        stream=True,
        bootstrap_host="127.0.0.1",
        bootstrap_port=BOOTSTRAP_PORT,
        bootstrap_room=room,
    )
    decode_tokens, decode_meta = await _collect(stream)
    return prefill_tokens, decode_tokens, decode_meta


async def _publish_prefill_room(prefill, room, *, host="127.0.0.1"):
    stream = await prefill.async_generate(
        input_ids=PROMPT,
        sampling_params={"max_new_tokens": 1},
        stream=True,
        bootstrap_host=host,
        bootstrap_port=prefill.sim_config.bootstrap_port,
        bootstrap_room=room,
    )
    await _collect(stream)
    return prefill._bootstrap_server._rooms.get(room)


@pytest.fixture
async def pd_pair(server_args):
    prefill = Engine(
        server_args=server_args(
            disaggregation_mode="prefill", disaggregation_bootstrap_port=BOOTSTRAP_PORT
        )
    )
    decode = Engine(server_args=server_args(disaggregation_mode="decode"))
    try:
        yield prefill, decode
    finally:
        await prefill._async_shutdown()
        await decode._async_shutdown()


async def test_disaggregated_answer_matches_the_aggregated_one(pd_pair, server_args):
    """The whole point: splitting prefill from decode must not change the answer."""
    prefill, decode = pd_pair
    aggregated = Engine(server_args=server_args())
    try:
        stream = await aggregated.async_generate(
            input_ids=PROMPT, sampling_params={"max_new_tokens": 8}, stream=True
        )
        agg_tokens, _ = await _collect(stream)

        _, disagg_tokens, _ = await _prefill_then_decode(
            prefill, decode, 1001, max_new_tokens=8
        )
        assert disagg_tokens == agg_tokens
    finally:
        await aggregated._async_shutdown()


async def test_decode_inherits_the_token_prefill_produced(pd_pair):
    prefill, decode = pd_pair
    prefill_tokens, decode_tokens, _ = await _prefill_then_decode(
        prefill, decode, 1002, max_new_tokens=4
    )
    assert decode_tokens[0] == prefill_tokens[0]


async def test_decode_reports_the_prompt_as_transferred_not_recomputed(pd_pair):
    prefill, decode = pd_pair
    _, _, meta = await _prefill_then_decode(prefill, decode, 1003, max_new_tokens=2)
    assert meta["cached_tokens"] == len(PROMPT)


async def test_room_is_released_once_the_transfer_lands(pd_pair):
    prefill, decode = pd_pair
    await _prefill_then_decode(prefill, decode, 1004, max_new_tokens=2)
    assert prefill._bootstrap_server._rooms == {}


@pytest.mark.parametrize("overrides", [
    {},
    {"enable_kv_return": True, "disable_radix_cache": True},
])
async def test_kv_return_metadata_is_omitted_when_ineligible(
    server_args,
    unused_port,
    overrides,
):
    prefill = Engine(server_args=server_args(
        disaggregation_mode="prefill",
        disaggregation_bootstrap_port=unused_port,
        **overrides,
    ))
    try:
        metadata = await _publish_prefill_room(prefill, 1100)
        assert metadata is not None
        assert set(metadata) == {
            "bootstrap_room",
            "num_tokens",
            "kv_bytes",
            "first_token",
            "location",
            "ready_ts",
        }
        assert "kv_return" not in metadata
    finally:
        await prefill._async_shutdown()


async def test_enabled_prefill_rooms_advertise_unique_return_tickets(
    server_args,
    unused_port,
):
    prefill = Engine(server_args=server_args(
        disaggregation_mode="prefill",
        disaggregation_bootstrap_port=unused_port,
        enable_kv_return=True,
        fake_kv_return_max_tokens=257,
        fake_kv_return_deadline_ms=1234.5,
    ))
    try:
        first = await _publish_prefill_room(prefill, 1101)
        second = await _publish_prefill_room(prefill, 1102)
        first_ticket = first["kv_return"]
        second_ticket = second["kv_return"]
        assert set(first_ticket) == {
            "nonce",
            "room_id",
            "attempt_id",
            "compatibility",
            "cache_salt",
            "max_generated_tokens",
            "max_payload_bytes",
            "deadline_ms",
        }
        assert first_ticket["room_id"] == 1101
        assert first_ticket["compatibility"] == {
            "protocol_version": 1,
            "prefill_dp_rank": 0,
            "served_model_name": prefill.sim_config.served_model_name,
            "model_identity": prefill.sim_config.model_path,
            "vocab_size": prefill.sim_config.vocab_size,
            "block_size": prefill.sim_config.block_size,
            "kv_bytes_per_token": prefill.sim_config.kv_return_bytes_per_token,
            "kv_layout_fingerprint": first_ticket["compatibility"][
                "kv_layout_fingerprint"
            ],
        }
        assert first_ticket["cache_salt"] is None
        assert first_ticket["max_generated_tokens"] == 257
        assert first_ticket["max_payload_bytes"] == (
            257 * prefill.sim_config.kv_return_bytes_per_token
        )
        assert first_ticket["deadline_ms"] == 1234.5
        assert len(first_ticket["nonce"]) >= 32
        assert first_ticket["nonce"] != second_ticket["nonce"]
        assert first_ticket["attempt_id"] != second_ticket["attempt_id"]
        assert "session_id" not in str(first_ticket)
        assert "session_id" not in str(second_ticket)
        assert set(prefill.scheduler.kv_return._attempts) == {
            first_ticket["nonce"],
            second_ticket["nonce"],
        }
    finally:
        await prefill._async_shutdown()


async def test_fake_prefill_bootstrap_host_never_publishes_return_metadata(
    server_args,
    unused_port,
):
    prefill = Engine(server_args=server_args(
        disaggregation_mode="prefill",
        disaggregation_bootstrap_port=unused_port,
        enable_kv_return=True,
    ))
    try:
        metadata = await _publish_prefill_room(
            prefill,
            1103,
            host=FAKE_BOOTSTRAP_HOST,
        )
        assert metadata is None
        assert prefill._bootstrap_server._rooms == {}
    finally:
        await prefill._async_shutdown()


async def test_transfer_time_tracks_bandwidth(pd_pair, server_args):
    """The evidence that KV transfer is modelled, not skipped."""
    prefill, _ = pd_pair
    slow = Engine(
        server_args=server_args(disaggregation_mode="decode", fake_kv_bandwidth_gb_s=1.0)
    )
    fast = Engine(
        server_args=server_args(
            disaggregation_mode="decode", fake_kv_bandwidth_gb_s=1000.0
        )
    )
    try:
        slow_ms = (await _timed_decode(prefill, slow, 1005))["transfer_ms"]
        fast_ms = (await _timed_decode(prefill, fast, 1006))["transfer_ms"]
        assert slow_ms > fast_ms * 10
    finally:
        await slow._async_shutdown()
        await fast._async_shutdown()


async def _timed_decode(prefill, decode, room) -> dict:
    stream = await prefill.async_generate(
        input_ids=PROMPT,
        sampling_params={"max_new_tokens": 1},
        stream=True,
        bootstrap_host="127.0.0.1",
        bootstrap_port=BOOTSTRAP_PORT,
        bootstrap_room=room,
    )
    await _collect(stream)

    from fakeengine.transfer import KVReceiver

    receiver = KVReceiver(
        decode.sim_config,
        decode._bootstrap_client,
        bootstrap_host="127.0.0.1",
        bootstrap_port=BOOTSTRAP_PORT,
        bootstrap_room=room,
    )
    assert receiver.poll() == KVPoll.Bootstrapping
    metadata = await receiver.receive(len(PROMPT))
    assert receiver.poll() == KVPoll.Success
    return metadata


async def test_fake_bootstrap_host_needs_no_peer(server_args):
    """Dynamo's health-check canary and prefill warmup both use this sentinel;
    if it waited on a peer, warmup would block for its 1800 s timeout."""
    decode = Engine(server_args=server_args(disaggregation_mode="decode"))
    try:
        started = time.monotonic()
        stream = await decode.async_generate(
            input_ids=[1, 2, 3],
            sampling_params={"max_new_tokens": 1},
            stream=True,
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_port=1,
            bootstrap_room=7,
        )
        tokens, _ = await _collect(stream)
        assert tokens
        assert time.monotonic() - started < 2.0
    finally:
        await decode._async_shutdown()


async def test_unclaimed_room_times_out_instead_of_hanging(pd_pair, server_args):
    prefill, _ = pd_pair
    impatient = Engine(
        server_args=server_args(
            disaggregation_mode="decode", fake_bootstrap_timeout_s=0.5
        )
    )
    try:
        stream = await impatient.async_generate(
            input_ids=[1, 2],
            sampling_params={"max_new_tokens": 1},
            stream=True,
            bootstrap_host="127.0.0.1",
            bootstrap_port=BOOTSTRAP_PORT,
            bootstrap_room=999_999,      # nobody ever publishes this
        )
        with pytest.raises(TimeoutError):
            await _collect(stream)
    finally:
        await impatient._async_shutdown()


async def test_unreachable_prefill_fails_fast(server_args):
    decode = Engine(
        server_args=server_args(
            disaggregation_mode="decode", fake_bootstrap_timeout_s=0.5
        )
    )
    try:
        stream = await decode.async_generate(
            input_ids=[1, 2],
            sampling_params={"max_new_tokens": 1},
            stream=True,
            bootstrap_host="127.0.0.1",
            bootstrap_port=1,            # nothing listening
            bootstrap_room=5,
        )
        with pytest.raises(TimeoutError):
            await _collect(stream)
    finally:
        await decode._async_shutdown()


async def test_unclaimed_rooms_are_swept(server_args):
    """A decode peer that never arrives must not leak a room forever."""
    from fakeengine.bootstrap import BootstrapServer

    server = BootstrapServer("127.0.0.1", 19011, room_ttl_s=1.0)
    await server.start()
    try:
        server.publish_room(42, num_tokens=8, kv_bytes=128, first_token=1)
        assert 42 in server._rooms
        await asyncio.sleep(1.6)         # one sweep interval past the TTL
        assert 42 not in server._rooms
    finally:
        await server.stop()
