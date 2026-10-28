from contextlib import asynccontextmanager
from dataclasses import asdict

import aiohttp

from fakeengine.bootstrap import BootstrapServer
from fakeengine.config import SimConfig
from fakeengine.kvcache import block_hashes
from fakeengine.kv_return import (
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
