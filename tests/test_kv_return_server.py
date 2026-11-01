from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import asdict, replace

import aiohttp
import pytest

from fakeengine.bootstrap import BootstrapServer
from fakeengine.config import SimConfig
from fakeengine.kv_return import (
    KV_RETURN_MAX_METADATA_LENGTH,
    KVReturnCompatibility,
    KVReturnPrepare,
    KVReturnTicket,
    kv_return_checksum,
)
from fakeengine.kvcache import block_hashes
from fakeengine.scheduler import Scheduler
from sglang import Engine


class RecordingEvents:
    def __init__(self):
        self.stored = []
        self.removed = []

    def publish_stored(self, **kwargs):
        self.stored.append(kwargs)

    def publish_removed(self, block_ids):
        self.removed.extend(block_ids)


def _scheduler(
    server_args,
    *,
    enabled=True,
    blocks=4,
    deadline_ms=500,
    model_identity=None,
    served_model_name=None,
):
    config = SimConfig.from_server_args(server_args(
        disaggregation_mode="prefill",
        enable_kv_return=enabled,
        fake_kv_return_max_tokens=8,
        fake_kv_return_deadline_ms=deadline_ms,
        fake_kv_return_bytes_per_token=16,
        fake_num_kv_blocks=blocks,
        page_size=4,
    ))
    if model_identity is not None:
        config.model_path = model_identity
    if served_model_name is not None:
        config.served_model_name = served_model_name
    events = RecordingEvents()
    return Scheduler(config, kv_events=events, dp_rank=0), events


def _seed(scheduler, prompt=(1, 2, 3, 4), salt=None):
    hashes = block_hashes(prompt, scheduler.config.block_size, salt)
    blocks = scheduler.pool.allocate(len(hashes))
    assert blocks is not None
    scheduler.pool.insert_prefix(hashes, blocks)
    scheduler.pool.free(blocks)
    return hashes


def _ticket(raw):
    compatibility = KVReturnCompatibility(**raw["compatibility"])
    return KVReturnTicket(
        nonce=raw["nonce"],
        room_id=raw["room_id"],
        attempt_id=raw["attempt_id"],
        compatibility=compatibility,
        cache_salt=raw["cache_salt"],
        max_generated_tokens=raw["max_generated_tokens"],
        max_payload_bytes=raw["max_payload_bytes"],
        deadline_ms=raw["deadline_ms"],
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


def _publish(
    server,
    scheduler,
    room,
    *,
    prompt=(1, 2, 3, 4),
    salt=None,
):
    server.publish_room(
        room,
        num_tokens=len(prompt),
        kv_bytes=len(prompt) * scheduler.config.kv_bytes_per_token,
        first_token=9,
        prefill_dp_rank=scheduler.dp_rank,
        kv_return_prompt_ids=prompt,
        kv_return_cache_salt=salt,
    )
    return _ticket(server._rooms[room]["kv_return"])


@asynccontextmanager
async def _running_server(
    scheduler,
    port,
    *,
    room_ttl_s=60,
):
    server = BootstrapServer(
        "127.0.0.1",
        port,
        room_ttl_s=room_ttl_s,
        kv_return_managers=[scheduler.kv_return],
    )
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


async def _post(session, port, operation, payload, *, version=1):
    async with session.post(
        f"http://127.0.0.1:{port}/kv-return/v{version}/{operation}",
        json=payload,
    ) as response:
        return response.status, await response.json()


async def test_prepare_commit_and_duplicates_use_manager_atomically(
    server_args,
    unused_port,
):
    scheduler, events = _scheduler(server_args)
    prompt = (1, 2, 3, 4)
    generated = (5, 6, 7, 8)
    salt = "tenant-a"
    _seed(scheduler, prompt, salt)

    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(
            server, scheduler, 10, prompt=prompt, salt=salt,
        )
        prepare = _prepare(ticket, generated)
        commit = {
            "prepare": asdict(prepare),
            "generated_token_ids": list(generated),
        }
        async with aiohttp.ClientSession() as session:
            first_prepare = await _post(
                session, unused_port, "prepare", asdict(prepare),
            )
            second_prepare = await _post(
                session, unused_port, "prepare", asdict(prepare),
            )
            first_commit = await _post(
                session, unused_port, "commit", commit,
            )
            second_commit = await _post(
                session, unused_port, "commit", commit,
            )

    assert first_prepare == (200, {
        "status": "ready",
        "reason": "reservation_ready",
        "idempotent": False,
        "reserved_blocks": 1,
    })
    assert second_prepare[1]["idempotent"] is True
    assert first_commit == (200, {
        "status": "committed",
        "reason": "prefix_committed",
        "idempotent": False,
        "reserved_blocks": 0,
    })
    assert second_commit[1]["idempotent"] is True
    assert scheduler.pool.match_prefix(
        block_hashes(prompt + generated, 4, salt)
    ) == 2
    assert len(events.stored) == 1


async def test_abort_and_duplicate_release_reservation(
    server_args,
    unused_port,
):
    scheduler, events = _scheduler(server_args)
    _seed(scheduler)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 11)
        prepare = _prepare(ticket)
        abort = {
            "ticket": asdict(ticket),
            "attempt_id": ticket.attempt_id,
        }
        async with aiohttp.ClientSession() as session:
            assert (
                await _post(session, unused_port, "prepare", asdict(prepare))
            )[1]["status"] == "ready"
            first = await _post(session, unused_port, "abort", abort)
            second = await _post(session, unused_port, "abort", abort)

    assert first == (200, {
        "status": "aborted",
        "reason": "attempt_aborted",
        "idempotent": False,
        "reserved_blocks": 0,
    })
    assert second[1]["idempotent"] is True
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []


@pytest.mark.parametrize(
    "body,reason",
    [
        ("[1,2]", "request_body_must_be_an_object"),
        ('{"ticket":', "malformed_json"),
    ],
)
async def test_malformed_json_is_typed(
    server_args,
    unused_port,
    body,
    reason,
):
    scheduler, _ = _scheduler(server_args)
    async with (
        _running_server(scheduler, unused_port),
        aiohttp.ClientSession() as session,
        session.post(
            f"http://127.0.0.1:{unused_port}/kv-return/v1/prepare",
            data=body,
            headers={"Content-Type": "application/json"},
        ) as response,
    ):
        payload = await response.json()
    assert response.status == 400
    assert payload == {
        "status": "invalid_request",
        "reason": reason,
        "idempotent": False,
        "reserved_blocks": 0,
    }


async def test_huge_numeric_json_failures_are_typed(
    server_args,
    unused_port,
):
    scheduler, _ = _scheduler(server_args)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 111)
        body = json.dumps(asdict(_prepare(ticket)))
        assert '"deadline_ms": 500.0' in body
        overflowing_number = body.replace(
            '"deadline_ms": 500.0',
            f'"deadline_ms": {"9" * 400}',
            1,
        )
        pathological_integer = body.replace(
            '"deadline_ms": 500.0',
            f'"deadline_ms": {"9" * 5000}',
            1,
        )
        assert len(pathological_integer) < (1 << 20)
        async with aiohttp.ClientSession() as session:
            responses = []
            for raw in (overflowing_number, pathological_integer):
                async with session.post(
                    (
                        f"http://127.0.0.1:{unused_port}"
                        "/kv-return/v1/prepare"
                    ),
                    data=raw,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    responses.append(
                        (response.status, await response.json())
                    )

    assert responses[0][0] == 400
    assert responses[0][1] == {
        "status": "invalid_request",
        "reason": "ticket_deadline_ms_must_be_a_finite_number",
        "idempotent": False,
        "reserved_blocks": 0,
    }
    assert responses[1][0] == 400
    assert responses[1][1]["status"] == "invalid_request"
    assert responses[1][1]["reason"] in {
        "malformed_json",
        "ticket_deadline_ms_must_be_a_finite_number",
    }
    assert responses[1][1]["idempotent"] is False
    assert responses[1][1]["reserved_blocks"] == 0


async def test_wrong_types_unknown_fields_and_oversized_bodies_are_rejected(
    server_args,
    unused_port,
):
    scheduler, _ = _scheduler(server_args)
    _seed(scheduler)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 12)
        prepare = asdict(_prepare(ticket))
        wrong_type = dict(prepare, generated_token_count=True)
        unknown = dict(prepare, session_id="forbidden")
        oversized_tokens = {
            "prepare": prepare,
            "generated_token_ids": list(range(
                ticket.max_generated_tokens + 1
            )),
        }
        async with aiohttp.ClientSession() as session:
            typed = await _post(
                session, unused_port, "prepare", wrong_type,
            )
            extra = await _post(
                session, unused_port, "prepare", unknown,
            )
            too_many = await _post(
                session, unused_port, "commit", oversized_tokens,
            )
            unsupported = await _post(
                session, unused_port, "prepare", prepare, version=2,
            )
            async with session.post(
                f"http://127.0.0.1:{unused_port}/kv-return/v1/prepare",
                data="{}",
                headers={"Content-Type": "text/plain"},
            ) as response:
                wrong_content_type = response.status, await response.json()
            async with session.post(
                f"http://127.0.0.1:{unused_port}/kv-return/v1/prepare",
                data="x" * ((1 << 20) + 1),
                headers={"Content-Type": "application/json"},
            ) as response:
                too_large = response.status, await response.json()

    assert typed[0] == 400
    assert typed[1]["reason"].startswith(
        "generated_token_count_must_be_an_integer"
    )
    assert extra[0] == 400
    assert extra[1]["reason"] == "prepare_unknown_session_id"
    assert too_many[0] == 413
    assert too_many[1]["reason"] == "generated_token_ids_exceed_ticket_limit"
    assert unsupported[0] == 409
    assert unsupported[1]["reason"] == "unsupported_protocol_version"
    assert wrong_content_type[0] == 400
    assert (
        wrong_content_type[1]["reason"]
        == "content_type_must_be_application_json"
    )
    assert too_large[0] == 413
    assert too_large[1]["reason"] == "request_body_too_large"


async def test_published_ticket_round_trips_at_protocol_string_limits(
    server_args,
    unused_port,
):
    boundary = "x" * KV_RETURN_MAX_METADATA_LENGTH
    scheduler, _ = _scheduler(
        server_args,
        model_identity=boundary,
        served_model_name=boundary,
    )
    _seed(scheduler, salt=boundary)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 112, salt=boundary)
        async with aiohttp.ClientSession() as session:
            response = await _post(
                session,
                unused_port,
                "prepare",
                asdict(_prepare(ticket)),
            )

    assert ticket.compatibility.model_identity == boundary
    assert ticket.compatibility.served_model_name == boundary
    assert ticket.cache_salt == boundary
    assert response[0] == 200
    assert response[1]["status"] == "ready"


@pytest.mark.parametrize("oversized_field", ["cache_salt", "model_identity"])
def test_unparseable_ticket_fields_are_rejected_before_publication(
    server_args,
    oversized_field,
):
    oversized = "x" * (KV_RETURN_MAX_METADATA_LENGTH + 1)
    scheduler, _ = _scheduler(
        server_args,
        model_identity=(
            oversized if oversized_field == "model_identity" else None
        ),
    )
    server = BootstrapServer(
        "127.0.0.1",
        1,
        kv_return_managers=[scheduler.kv_return],
    )
    server.publish_room(
        113,
        num_tokens=4,
        kv_bytes=64,
        prefill_dp_rank=0,
        kv_return_prompt_ids=(1, 2, 3, 4),
        kv_return_cache_salt=(
            oversized if oversized_field == "cache_salt" else None
        ),
    )

    assert server._rooms[113]["kv_return"] is None
    assert scheduler.kv_return._attempts == {}


@pytest.mark.parametrize(
    "case,expected_status,expected_reason",
    [
        ("nonce", 404, "ticket_not_found"),
        ("ticket", 409, "incompatible_control_contract"),
        ("layout", 409, "incompatible_control_contract"),
        ("rank", 409, "prefill_dp_rank_not_hosted"),
        ("salt", 409, "incompatible_control_contract"),
        ("bytes", 409, "incompatible_control_contract"),
    ],
)
async def test_wrong_authenticated_controls_are_rejected(
    server_args,
    unused_port,
    case,
    expected_status,
    expected_reason,
):
    scheduler, events = _scheduler(server_args)
    _seed(scheduler, salt="tenant-a")
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 13, salt="tenant-a")
        prepare = _prepare(ticket)
        if case == "nonce":
            prepare = replace(
                prepare,
                ticket=replace(ticket, nonce="u" * 32),
            )
        elif case == "ticket":
            changed = replace(ticket, attempt_id="other-attempt")
            prepare = replace(
                prepare,
                ticket=changed,
                attempt_id=changed.attempt_id,
            )
        elif case == "layout":
            compatibility = replace(
                ticket.compatibility,
                kv_layout_fingerprint="0" * 32,
            )
            changed = replace(ticket, compatibility=compatibility)
            prepare = replace(
                prepare,
                ticket=changed,
                compatibility_fingerprint=compatibility.fingerprint,
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
            prepare = replace(prepare, cache_salt="tenant-b")
        elif case == "bytes":
            prepare = replace(
                prepare,
                claimed_payload_bytes=prepare.claimed_payload_bytes + 1,
            )
        async with aiohttp.ClientSession() as session:
            status, payload = await _post(
                session, unused_port, "prepare", asdict(prepare),
            )

    assert status == expected_status
    assert payload["status"] in {"not_found", "incompatible"}
    assert payload["reason"] == expected_reason
    assert events.stored == events.removed == []


async def test_deadline_expiry_releases_capacity_and_stays_idempotent(
    server_args,
    unused_port,
):
    scheduler, _ = _scheduler(server_args, deadline_ms=30)
    _seed(scheduler)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 14)
        prepare = _prepare(ticket)
        async with aiohttp.ClientSession() as session:
            assert (
                await _post(session, unused_port, "prepare", asdict(prepare))
            )[1]["status"] == "ready"
            await asyncio.sleep(0.12)
            expired = await _post(
                session, unused_port, "prepare", asdict(prepare),
            )

    assert expired[0] == 410
    assert expired[1]["status"] == "expired"
    assert expired[1]["idempotent"] is True
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_room_expiry_clears_active_and_terminal_manager_state(
    server_args,
    unused_port,
):
    scheduler, _ = _scheduler(server_args)
    _seed(scheduler)
    async with _running_server(
        scheduler, unused_port, room_ttl_s=0.04,
    ) as server:
        active = _publish(server, scheduler, 15)
        terminal = _publish(server, scheduler, 16)
        active_prepare = _prepare(active)
        abort = {
            "ticket": asdict(terminal),
            "attempt_id": terminal.attempt_id,
        }
        async with aiohttp.ClientSession() as session:
            await _post(
                session, unused_port, "prepare", asdict(active_prepare),
            )
            await _post(session, unused_port, "abort", abort)
        assert scheduler.kv_return._attempts
        assert scheduler.kv_return._terminal
        await asyncio.sleep(0.12)
        assert server._rooms == {}
        assert server._room_returns == {}
        assert scheduler.kv_return._attempts == {}
        assert scheduler.kv_return._terminal == {}
        assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_server_stop_clears_issued_prepared_and_terminal_state(
    server_args,
    unused_port,
):
    scheduler, _ = _scheduler(server_args)
    _seed(scheduler)
    server = BootstrapServer(
        "127.0.0.1",
        unused_port,
        kv_return_managers=[scheduler.kv_return],
    )
    await server.start()
    issued = _publish(server, scheduler, 17)
    prepared = _publish(server, scheduler, 18)
    terminal = _publish(server, scheduler, 19)
    async with aiohttp.ClientSession() as session:
        await _post(
            session, unused_port, "prepare", asdict(_prepare(prepared)),
        )
        await _post(session, unused_port, "abort", {
            "ticket": asdict(terminal),
            "attempt_id": terminal.attempt_id,
        })
    assert issued.nonce
    await server.stop()

    assert server._sweeper is None
    assert server._runner is None
    assert server._rooms == {}
    assert server._room_returns == {}
    assert scheduler.kv_return._attempts == {}
    assert scheduler.kv_return._terminal == {}
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)

    await server.start()
    try:
        async with aiohttp.ClientSession() as session:
            base = f"http://127.0.0.1:{unused_port}"
            async with session.get(f"{base}/health") as response:
                health = response.status, await response.json()
            async with session.get(f"{base}/room/17") as response:
                stale_room = response.status, await response.json()
    finally:
        await server.stop()

    assert health == (200, {
        "status": "ok",
        "engine": "fake-sglang",
        "rooms": 0,
    })
    assert stale_room == (404, {
        "status": "pending",
        "bootstrap_room": 17,
    })
    assert scheduler.kv_return._attempts == {}
    assert scheduler.kv_return._terminal == {}
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


async def test_engine_shutdown_releases_server_reservation(
    server_args,
    unused_port,
):
    engine = Engine(server_args=server_args(
        disaggregation_mode="prefill",
        disaggregation_bootstrap_port=unused_port,
        enable_kv_return=True,
        fake_kv_return_max_tokens=8,
        fake_kv_return_deadline_ms=500,
        page_size=4,
    ))
    prompt = [1, 2, 3, 4]
    stream = await engine.async_generate(
        input_ids=prompt,
        sampling_params={"max_new_tokens": 1},
        stream=True,
        bootstrap_host="127.0.0.1",
        bootstrap_port=unused_port,
        bootstrap_room=23,
    )
    async for _ in stream:
        pass
    ticket = _ticket(engine._bootstrap_server._rooms[23]["kv_return"])
    async with aiohttp.ClientSession() as session:
        prepared = await _post(
            session,
            unused_port,
            "prepare",
            asdict(_prepare(ticket)),
        )
    assert prepared[1]["status"] == "ready"
    assert not engine.scheduler.pool.can_allocate(
        engine.scheduler.pool.num_blocks
    )

    await engine._async_shutdown()

    assert engine._bootstrap_server._runner is None
    assert engine._bootstrap_server._sweeper is None
    assert engine.scheduler.kv_return._attempts == {}
    assert engine.scheduler.kv_return._terminal == {}
    assert engine.scheduler.pool.can_allocate(engine.scheduler.pool.num_blocks)


async def test_capacity_rejection_happens_before_transfer_or_events(
    server_args,
    unused_port,
):
    scheduler, events = _scheduler(server_args, blocks=2)
    prompt_hashes = _seed(scheduler)
    held = scheduler.pool.allocate(1)
    assert held is not None
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 20)
        async with aiohttp.ClientSession() as session:
            rejected = await _post(
                session,
                unused_port,
                "prepare",
                asdict(_prepare(ticket)),
            )

    assert rejected[0] == 409
    assert rejected[1]["status"] == "capacity_rejected"
    assert rejected[1]["reason"] == "insufficient_clean_capacity"
    assert scheduler.pool.match_prefix(prompt_hashes) == 1
    assert events.stored == events.removed == []
    scheduler.pool.free(held)


async def test_checksum_failure_has_no_partial_cache_or_events(
    server_args,
    unused_port,
):
    scheduler, events = _scheduler(server_args)
    prompt = (1, 2, 3, 4)
    generated = (5, 6, 7, 8)
    prompt_hashes = _seed(scheduler, prompt)
    async with _running_server(scheduler, unused_port) as server:
        ticket = _publish(server, scheduler, 21, prompt=prompt)
        prepare = replace(
            _prepare(ticket, generated),
            token_ids_checksum="0" * 32,
        )
        async with aiohttp.ClientSession() as session:
            assert (
                await _post(session, unused_port, "prepare", asdict(prepare))
            )[1]["status"] == "ready"
            rejected = await _post(session, unused_port, "commit", {
                "prepare": asdict(prepare),
                "generated_token_ids": list(generated),
            })

    assert rejected[0] == 422
    assert rejected[1]["status"] == "checksum_mismatch"
    assert scheduler.pool.match_prefix(
        block_hashes(prompt + generated, 4)
    ) == len(prompt_hashes)
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []


async def test_disabled_feature_and_legacy_routes_are_unchanged(
    server_args,
    unused_port,
):
    scheduler, events = _scheduler(server_args, enabled=False)
    async with _running_server(scheduler, unused_port) as server:
        server.publish_room(
            22,
            num_tokens=4,
            kv_bytes=64,
            first_token=7,
            prefill_dp_rank=0,
            kv_return_prompt_ids=(1, 2, 3, 4),
        )
        async with aiohttp.ClientSession() as session:
            base = f"http://127.0.0.1:{unused_port}"
            async with session.get(f"{base}/health") as response:
                health = await response.json()
            async with session.get(f"{base}/route") as response:
                route = await response.json()
            async with session.get(f"{base}/room/22") as response:
                room = await response.json()

    assert health == {
        "status": "ok",
        "engine": "fake-sglang",
        "rooms": 1,
    }
    assert route == {
        "engine": "fake-sglang",
        "prefill_host": "127.0.0.1",
        "prefill_port": unused_port,
        "ranks": {},
        "follow_bootstrap_room": True,
    }
    assert {
        key: value for key, value in room.items() if key != "ready_ts"
    } == {
        "bootstrap_room": 22,
        "num_tokens": 4,
        "kv_bytes": 64,
        "first_token": 7,
        "location": None,
        "kv_return": None,
    }
    assert scheduler.kv_return._attempts == {}
    assert events.stored == events.removed == []
