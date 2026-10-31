from contextlib import asynccontextmanager
from dataclasses import asdict, replace
import asyncio
import json

import aiohttp
import pytest

from fakeengine.bootstrap import BootstrapServer
from fakeengine.config import SimConfig
from fakeengine.kvcache import block_hashes
from fakeengine.kv_return import (
    KV_RETURN_MAX_METADATA_LENGTH,
    KVReturnCompatibility,
    KVReturnPrepare,
    KVReturnTicket,
    kv_return_checksum,
)
from fakeengine.scheduler import Scheduler


def _scheduler(server_args, *, enabled=True):
    config = SimConfig.from_server_args(server_args(
        disaggregation_mode="prefill",
        enable_kv_return=enabled,
        fake_kv_return_max_tokens=8,
        fake_kv_return_deadline_ms=500,
        fake_kv_return_bytes_per_token=16,
        fake_num_kv_blocks=4,
        page_size=4,
    ))
    return Scheduler(config, dp_rank=0)


def _seed(scheduler, prompt=(1, 2, 3, 4)):
    hashes = block_hashes(prompt, scheduler.config.block_size)
    blocks = scheduler.pool.allocate(len(hashes))
    assert blocks is not None
    scheduler.pool.insert_prefix(hashes, blocks)
    scheduler.pool.free(blocks)


def _ticket(raw):
    return KVReturnTicket(
        **{
            **raw,
            "compatibility": KVReturnCompatibility(**raw["compatibility"]),
        }
    )


def _prepare(ticket, generated=(5, 6, 7, 8)):
    return KVReturnPrepare(
        ticket=ticket,
        attempt_id=ticket.attempt_id,
        prefill_dp_rank=ticket.compatibility.prefill_dp_rank,
        compatibility_fingerprint=ticket.compatibility.fingerprint,
        cache_salt=ticket.cache_salt,
        generated_token_count=len(generated),
        claimed_payload_bytes=(
            len(generated) * ticket.compatibility.kv_bytes_per_token
        ),
        token_ids_checksum=kv_return_checksum(ticket, generated),
    )


def _publish(server, scheduler, room):
    server.publish_room(
        room,
        num_tokens=4,
        kv_bytes=64,
        first_token=9,
        prefill_dp_rank=scheduler.dp_rank,
        kv_return_prompt_ids=(1, 2, 3, 4),
    )
    return _ticket(server._rooms[room]["kv_return"])


@asynccontextmanager
async def _running_server(scheduler, port):
    server = BootstrapServer(
        "127.0.0.1",
        port,
        kv_return_managers=[scheduler.kv_return],
    )
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


async def _post(session, port, operation, payload):
    async with session.post(
        f"http://127.0.0.1:{port}/kv-return/v1/{operation}",
        json=payload,
    ) as response:
        return response.status, await response.json()


async def test_prepare_commit_and_abort_are_idempotent(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args)
    _seed(scheduler)
    generated = (5, 6, 7, 8)
    async with _running_server(scheduler, unused_port) as server:
        committed_ticket = _publish(server, scheduler, 10)
        committed_prepare = _prepare(committed_ticket, generated)
        commit = {
            "prepare": asdict(committed_prepare),
            "generated_token_ids": list(generated),
        }
        aborted_ticket = _publish(server, scheduler, 11)
        aborted_prepare = _prepare(aborted_ticket)
        abort = {
            "ticket": asdict(aborted_ticket),
            "attempt_id": aborted_ticket.attempt_id,
        }
        async with aiohttp.ClientSession() as session:
            prepared = [
                await _post(
                    session,
                    unused_port,
                    "prepare",
                    asdict(committed_prepare),
                )
                for _ in range(2)
            ]
            committed = [
                await _post(session, unused_port, "commit", commit)
                for _ in range(2)
            ]
            await _post(
                session,
                unused_port,
                "prepare",
                asdict(aborted_prepare),
            )
            aborted = [
                await _post(session, unused_port, "abort", abort)
                for _ in range(2)
            ]

    assert [response[1]["status"] for response in prepared] == ["ready"] * 2
    assert prepared[1][1]["idempotent"] is True
    assert [response[1]["status"] for response in committed] == ["committed"] * 2
    assert committed[1][1]["idempotent"] is True
    assert [response[1]["status"] for response in aborted] == ["aborted"] * 2
    assert aborted[1][1]["idempotent"] is True
    assert scheduler.pool.match_prefix(
        block_hashes((1, 2, 3, 4) + generated, 4)
    ) == 2


async def test_server_cleanup_releases_control_state(server_args, unused_port):
    scheduler = _scheduler(server_args)
    _seed(scheduler)
    server = BootstrapServer(
        "127.0.0.1",
        unused_port,
        kv_return_managers=[scheduler.kv_return],
    )
    await server.start()
    _publish(server, scheduler, 20)
    prepared = _publish(server, scheduler, 21)
    async with aiohttp.ClientSession() as session:
        await _post(
            session,
            unused_port,
            "prepare",
            asdict(_prepare(prepared)),
        )

    await server.stop()

    assert scheduler.kv_return._attempts == {}
    assert scheduler.kv_return._terminal == {}
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_disabled_feature_preserves_legacy_room_routes(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args, enabled=False)
    async with _running_server(scheduler, unused_port) as server:
        server.publish_room(
            30,
            num_tokens=4,
            kv_bytes=64,
            first_token=7,
            prefill_dp_rank=0,
            kv_return_prompt_ids=(1, 2, 3, 4),
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{unused_port}/room/30"
            ) as response:
                room = await response.json()

    assert "kv_return" not in room
    assert room["first_token"] == 7
    assert scheduler.kv_return._attempts == {}


@pytest.mark.parametrize(
    "body,content_type,version,status,reason",
    [
        ("[1,2]", "application/json", 1, 400, "request_body_must_be_an_object"),
        ('{"ticket":', "application/json", 1, 400, "malformed_json"),
        ("{}", "text/plain", 1, 400, "content_type_must_be_application_json"),
        ("{}", "application/json", 2, 400, "prepare_missing_attempt_id"),
        (
            "x" * ((1 << 20) + 1),
            "application/json",
            1,
            413,
            "request_body_too_large",
        ),
    ],
)
async def test_invalid_request_envelopes_are_typed(
    server_args,
    unused_port,
    body,
    content_type,
    version,
    status,
    reason,
):
    scheduler = _scheduler(server_args)
    async with _running_server(scheduler, unused_port):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                (
                    f"http://127.0.0.1:{unused_port}"
                    f"/kv-return/v{version}/prepare"
                ),
                data=body,
                headers={"Content-Type": content_type},
            ) as response:
                payload = await response.json()

    assert response.status == status
    assert payload["status"] == "invalid_request"
    assert payload["reason"] == reason


@pytest.mark.parametrize(
    "operation,mutate,status,reason",
    [
        (
            "prepare",
            lambda prepare, _ticket: {
                **prepare,
                "generated_token_count": True,
            },
            400,
            "generated_token_count_must_be_an_integer_at_least_0",
        ),
        (
            "prepare",
            lambda prepare, _ticket: {**prepare, "session_id": "unknown"},
            400,
            "prepare_unknown_session_id",
        ),
        (
            "prepare",
            lambda prepare, _ticket: {
                **prepare,
                "compatibility_fingerprint": "not-a-fingerprint",
            },
            400,
            "compatibility_fingerprint_must_be_32_lowercase_hex_characters",
        ),
        (
            "commit",
            lambda prepare, ticket: {
                "prepare": prepare,
                "generated_token_ids": list(
                    range(ticket.max_generated_tokens + 1)
                ),
            },
            413,
            "generated_token_ids_exceed_ticket_limit",
        ),
    ],
)
async def test_control_fields_are_strictly_validated(
    server_args,
    unused_port,
    operation,
    mutate,
    status,
    reason,
):
    scheduler = _scheduler(server_args)
    _seed(scheduler)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 40)
        prepare = asdict(_prepare(ticket))
        async with aiohttp.ClientSession() as session:
            response = await _post(
                session,
                unused_port,
                operation,
                mutate(prepare, ticket),
            )

    assert response[0] == status
    assert response[1]["status"] == "invalid_request"
    assert response[1]["reason"] == reason


@pytest.mark.parametrize("case", ["nonce", "rank", "salt", "bytes"])
async def test_authenticated_controls_cannot_be_substituted(
    server_args,
    unused_port,
    case,
):
    scheduler = _scheduler(server_args)
    _seed(scheduler)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 50)
        prepare = _prepare(ticket)
        if case == "nonce":
            prepare = replace(
                prepare,
                ticket=replace(ticket, nonce="u" * 32),
            )
        elif case == "rank":
            compatibility = replace(
                ticket.compatibility,
                prefill_dp_rank=1,
            )
            changed = replace(ticket, compatibility=compatibility)
            prepare = replace(
                prepare,
                ticket=changed,
                prefill_dp_rank=1,
                compatibility_fingerprint=compatibility.fingerprint,
            )
        elif case == "salt":
            prepare = replace(prepare, cache_salt="other-tenant")
        else:
            prepare = replace(
                prepare,
                claimed_payload_bytes=prepare.claimed_payload_bytes + 1,
            )
        async with aiohttp.ClientSession() as session:
            status, payload = await _post(
                session,
                unused_port,
                "prepare",
                asdict(prepare),
            )

    assert status in {404, 409}
    assert payload["status"] in {"not_found", "incompatible"}


async def test_room_cleanup_releases_active_and_terminal_state(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args)
    _seed(scheduler)
    server = BootstrapServer(
        "127.0.0.1",
        unused_port,
        room_ttl_s=0.04,
        kv_return_managers=[scheduler.kv_return],
    )
    await server.start()
    active = _publish(server, scheduler, 60)
    terminal = _publish(server, scheduler, 61)
    async with aiohttp.ClientSession() as session:
        await _post(session, unused_port, "abort", {
            "ticket": asdict(terminal),
            "attempt_id": terminal.attempt_id,
        })
    await asyncio.sleep(0.12)
    assert active.nonce not in scheduler.kv_return._attempts
    await server.stop()

    assert scheduler.kv_return._attempts == {}
    assert scheduler.kv_return._terminal == {}
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_protocol_limits_are_enforced_before_manager_dispatch(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args)
    _seed(scheduler)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 70)
        prepare = asdict(_prepare(ticket))
        raw = json.dumps(prepare).replace(
            '"deadline_ms": 500.0',
            f'"deadline_ms": {"9" * 400}',
            1,
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{unused_port}/kv-return/v1/prepare",
                data=raw,
                headers={"Content-Type": "application/json"},
            ) as response:
                payload = response.status, await response.json()

    assert payload[0] == 400
    assert payload[1]["status"] == "invalid_request"

    oversized = "x" * (KV_RETURN_MAX_METADATA_LENGTH + 1)
    server = BootstrapServer(
        "127.0.0.1",
        1,
        kv_return_managers=[scheduler.kv_return],
    )
    server.publish_room(
        71,
        num_tokens=4,
        kv_bytes=64,
        prefill_dp_rank=0,
        kv_return_prompt_ids=(1, 2, 3, 4),
        kv_return_cache_salt=oversized,
    )
    assert "kv_return" not in server._rooms[71]
