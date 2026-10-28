from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
import asyncio
import json
import time
from unittest.mock import AsyncMock

from aiohttp import web
import pytest

from fakeengine.bootstrap import (
    BootstrapClient,
    BootstrapServer,
    KVReturnControlStatus,
    KVReturnProtocolError,
    KVReturnTransportError,
)
from fakeengine.config import SimConfig
from fakeengine.kvcache import block_hashes
from fakeengine.kv_return import KVReturnResult, KVReturnStatus
from fakeengine.kv_return_client import (
    KVReturnOutcomeStatus,
    return_generated_kv,
)
from fakeengine.scheduler import Scheduler


TOPOLOGY = {
    "levels": ["zone", "rack", "node"],
    "links": {
        "node": {"bandwidth_gb_s": 100, "latency_ms": 0.1},
        "rack": {"bandwidth_gb_s": 2, "latency_ms": 3},
        "zone": {"bandwidth_gb_s": 1, "latency_ms": 5},
        "default": {"bandwidth_gb_s": 0.5, "latency_ms": 9},
    },
}


def _config(server_args, mode, **overrides):
    settings = dict(
        disaggregation_mode=mode,
        enable_kv_return=True,
        fake_kv_return_max_tokens=16,
        fake_kv_return_deadline_ms=500,
        fake_kv_return_bytes_per_token=32,
        fake_num_kv_blocks=8,
        page_size=4,
    )
    settings.update(overrides)
    return SimConfig.from_server_args(server_args(**settings))


def _scheduler(server_args, **overrides):
    return Scheduler(_config(server_args, "prefill", **overrides), dp_rank=0)


def _seed(scheduler, prompt, salt=None):
    hashes = block_hashes(prompt, scheduler.config.block_size, salt)
    blocks = scheduler.pool.allocate(len(hashes))
    assert blocks is not None
    scheduler.pool.insert_prefix(hashes, blocks)
    scheduler.pool.free(blocks)


def _publish(
    server,
    scheduler,
    room,
    *,
    prompt=(1, 2, 3, 4),
    salt=None,
):
    _seed(scheduler, prompt, salt)
    server.publish_room(
        room,
        num_tokens=len(prompt),
        kv_bytes=len(prompt) * scheduler.config.kv_bytes_per_token,
        first_token=99,
        location=scheduler.config.location,
        prefill_dp_rank=scheduler.dp_rank,
        kv_return_prompt_ids=prompt,
        kv_return_cache_salt=salt,
    )
    return deepcopy(server._rooms[room])


@asynccontextmanager
async def _running_server(scheduler, port, server_type=BootstrapServer):
    server = server_type(
        "127.0.0.1",
        port,
        kv_return_managers=[scheduler.kv_return],
    )
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


async def _no_sleep(_seconds):
    return None


async def test_return_helper_uses_real_endpoints_and_generated_bytes_only(
    server_args,
    unused_port,
):
    prompt = tuple(range(1, 9))
    generated = (9, 10, 11, 12)
    scheduler = _scheduler(server_args)
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 1)
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(
                server,
                scheduler,
                31,
                prompt=prompt,
                salt="tenant-a",
            )
            outcome = await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=generated,
                sleep=_no_sleep,
            )
    finally:
        await client.close()

    assert outcome.status == KVReturnOutcomeStatus.COMMITTED
    assert outcome.prepare_response.status == KVReturnControlStatus.READY
    assert outcome.commit_response.status == KVReturnControlStatus.COMMITTED
    assert outcome.payload_bytes == len(generated) * 32
    assert outcome.payload_bytes != (len(prompt) + len(generated)) * 32
    assert outcome.attempt_id == metadata["kv_return"]["attempt_id"]
    assert len(outcome.attempt_id) >= 24
    assert scheduler.pool.match_prefix(
        block_hashes(prompt + generated, 4, "tenant-a")
    ) == 3


async def test_prepare_rejection_never_starts_payload_transfer(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args, fake_num_kv_blocks=2)
    held = scheduler.pool.allocate(1)
    assert held is not None
    decode = _config(server_args, "decode", fake_num_kv_blocks=2)
    client = BootstrapClient(0.001, 1)
    sleep = AsyncMock()
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(server, scheduler, 32)
            outcome = await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=(5, 6, 7, 8),
                sleep=sleep,
            )
    finally:
        await client.close()
        scheduler.pool.free(held)

    assert outcome.status == KVReturnOutcomeStatus.PREPARE_REJECTED
    assert (
        outcome.prepare_response.status
        == KVReturnControlStatus.CAPACITY_REJECTED
    )
    sleep.assert_not_awaited()


class _SlowPrepareServer(BootstrapServer):
    async def _prepare_kv_return(self, request):
        await asyncio.sleep(0.1)
        return await super()._prepare_kv_return(request)


async def test_one_client_timeout_covers_prepare(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args)
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 0.02)
    try:
        async with _running_server(
            scheduler,
            unused_port,
            _SlowPrepareServer,
        ) as server:
            metadata = _publish(server, scheduler, 33)
            with pytest.raises(TimeoutError, match="prepare"):
                await return_generated_kv(
                    decode,
                    client,
                    host="127.0.0.1",
                    port=unused_port,
                    room_metadata=metadata,
                    generated_token_ids=(5, 6, 7, 8),
                )
            await asyncio.sleep(0.12)
            assert scheduler.kv_return._attempts == {}
            assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    finally:
        await client.close()


async def test_one_client_timeout_covers_simulated_transfer_and_aborts(
    server_args,
    unused_port,
):
    scheduler = _scheduler(
        server_args,
        fake_kv_return_overhead_ms=100,
        fake_speedup_ratio=1,
    )
    decode = _config(
        server_args,
        "decode",
        fake_kv_return_overhead_ms=100,
        fake_speedup_ratio=1,
    )
    client = BootstrapClient(0.001, 0.03)
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(server, scheduler, 34)
            with pytest.raises(TimeoutError, match="simulated"):
                await return_generated_kv(
                    decode,
                    client,
                    host="127.0.0.1",
                    port=unused_port,
                    room_metadata=metadata,
                    generated_token_ids=(5, 6, 7, 8),
                )
            assert scheduler.kv_return._attempts == {}
            assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    finally:
        await client.close()


@pytest.mark.parametrize("clock_offset_s", [-86400, 86400])
async def test_prefill_wall_clock_skew_does_not_change_ticket_budget(
    server_args,
    unused_port,
    clock_offset_s,
):
    scheduler = _scheduler(server_args)
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 1)
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(server, scheduler, 44)
            metadata["ready_ts"] = time.time() + clock_offset_s
            outcome = await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=(5, 6, 7, 8),
                sleep=_no_sleep,
            )
    finally:
        await client.close()

    assert outcome.status == KVReturnOutcomeStatus.COMMITTED


class _BudgetPrepareServer(BootstrapServer):
    async def _prepare_kv_return(self, request):
        await asyncio.sleep(0.03)
        return await super()._prepare_kv_return(request)


async def test_prepare_and_transfer_share_one_monotonic_budget(
    server_args,
    unused_port,
):
    scheduler = _scheduler(
        server_args,
        fake_kv_return_overhead_ms=30,
        fake_speedup_ratio=1,
    )
    decode = _config(
        server_args,
        "decode",
        fake_kv_return_overhead_ms=30,
        fake_speedup_ratio=1,
    )
    client = BootstrapClient(0.001, 0.05)
    try:
        async with _running_server(
            scheduler,
            unused_port,
            _BudgetPrepareServer,
        ) as server:
            metadata = _publish(server, scheduler, 45)
            metadata["ready_ts"] = time.time() + 86400
            with pytest.raises(TimeoutError, match="simulated"):
                await return_generated_kv(
                    decode,
                    client,
                    host="127.0.0.1",
                    port=unused_port,
                    room_metadata=metadata,
                    generated_token_ids=(5, 6, 7, 8),
                )
            assert scheduler.kv_return._attempts == {}
            assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    finally:
        await client.close()


class _NonJsonPrepareServer(BootstrapServer):
    async def _prepare_kv_return(self, _request):
        return web.Response(text="not-json", content_type="text/plain")


class _MalformedJsonPrepareServer(BootstrapServer):
    async def _prepare_kv_return(self, _request):
        return web.Response(text="{", content_type="application/json")


@pytest.mark.parametrize(
    "server_type,error",
    [
        (_NonJsonPrepareServer, "application/json"),
        (_MalformedJsonPrepareServer, "not valid JSON"),
    ],
)
async def test_malformed_control_responses_are_explicit_protocol_errors(
    server_args,
    unused_port,
    server_type,
    error,
):
    scheduler = _scheduler(server_args)
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 1)
    try:
        async with _running_server(
            scheduler,
            unused_port,
            server_type,
        ) as server:
            metadata = _publish(server, scheduler, 35)
            with pytest.raises(KVReturnProtocolError, match=error):
                await return_generated_kv(
                    decode,
                    client,
                    host="127.0.0.1",
                    port=unused_port,
                    room_metadata=metadata,
                    generated_token_ids=(5, 6, 7, 8),
                    sleep=_no_sleep,
                )
    finally:
        await client.close()


class _WrongStatusPrepareServer(BootstrapServer):
    async def _prepare_kv_return(self, _request):
        return web.json_response(
            {
                "status": "ready",
                "reason": "reservation_ready",
                "idempotent": False,
                "reserved_blocks": 1,
            },
            status=409,
        )


async def test_wrong_response_status_shape_is_rejected(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args)
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 1)
    try:
        async with _running_server(
            scheduler,
            unused_port,
            _WrongStatusPrepareServer,
        ) as server:
            metadata = _publish(server, scheduler, 36)
            with pytest.raises(KVReturnProtocolError, match="disagrees"):
                await return_generated_kv(
                    decode,
                    client,
                    host="127.0.0.1",
                    port=unused_port,
                    room_metadata=metadata,
                    generated_token_ids=(5, 6, 7, 8),
                    sleep=_no_sleep,
                )
    finally:
        await client.close()


async def test_commit_rejection_returns_typed_failure_and_aborts(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args)
    scheduler.kv_return.commit = lambda _commit: KVReturnResult(
        KVReturnStatus.IMPORT_FAILED
    )
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 1)
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(server, scheduler, 37)
            outcome = await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=(5, 6, 7, 8),
                sleep=_no_sleep,
            )
            assert scheduler.kv_return._attempts == {}
            assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    finally:
        await client.close()

    assert outcome.status == KVReturnOutcomeStatus.COMMIT_REJECTED
    assert outcome.commit_response.status == KVReturnControlStatus.IMPORT_FAILED


async def test_transfer_failure_preserves_primary_error_after_abort(
    server_args,
    unused_port,
):
    class PayloadTransferError(RuntimeError):
        pass

    async def fail_transfer(_seconds):
        raise PayloadTransferError("payload transport failed")

    scheduler = _scheduler(server_args)
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 1)
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(server, scheduler, 38)
            with pytest.raises(
                PayloadTransferError,
                match="payload transport failed",
            ):
                await return_generated_kv(
                    decode,
                    client,
                    host="127.0.0.1",
                    port=unused_port,
                    room_metadata=metadata,
                    generated_token_ids=(5, 6, 7, 8),
                    sleep=fail_transfer,
                )
            assert scheduler.kv_return._attempts == {}
            assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    finally:
        await client.close()


async def test_retry_reuses_ticket_attempt_and_skips_duplicate_transfer(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args)
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 1)
    sleep = AsyncMock()
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(server, scheduler, 39)
            first = await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=(5, 6, 7, 8),
                sleep=sleep,
            )
            second = await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=(5, 6, 7, 8),
                sleep=sleep,
            )
    finally:
        await client.close()

    assert first.status == second.status == KVReturnOutcomeStatus.COMMITTED
    assert first.attempt_id == second.attempt_id
    assert second.prepare_response.status == KVReturnControlStatus.COMMITTED
    assert second.prepare_response.idempotent is True
    assert sleep.await_count == 1


async def test_reverse_topology_uses_decode_to_prefill_link(
    server_args,
    unused_port,
):
    topology = json.dumps(TOPOLOGY)
    scheduler = _scheduler(
        server_args,
        fake_topology=topology,
        fake_location="z/r/p",
    )
    decode = _config(
        server_args,
        "decode",
        fake_topology=topology,
        fake_location="z/r/d",
        fake_kv_return_bandwidth_gb_s=1000,
        fake_kv_return_overhead_ms=0,
    )
    client = BootstrapClient(0.001, 1)
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(server, scheduler, 40)
            outcome = await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=(5, 6, 7, 8),
                sleep=_no_sleep,
            )
    finally:
        await client.close()

    assert outcome.link == "rack"
    assert outcome.transfer_ms == pytest.approx(
        3 + outcome.payload_bytes / (2e9) * 1000
    )


async def test_unplaced_reverse_transfer_uses_return_specific_fallbacks(
    server_args,
    unused_port,
):
    scheduler = _scheduler(
        server_args,
        fake_kv_bandwidth_gb_s=999,
        fake_kv_transfer_overhead_ms=99,
        fake_kv_return_bandwidth_gb_s=4,
        fake_kv_return_overhead_ms=7,
    )
    decode = _config(
        server_args,
        "decode",
        fake_kv_bandwidth_gb_s=999,
        fake_kv_transfer_overhead_ms=99,
        fake_kv_return_bandwidth_gb_s=4,
        fake_kv_return_overhead_ms=7,
    )
    client = BootstrapClient(0.001, 1)
    try:
        async with _running_server(scheduler, unused_port) as server:
            metadata = _publish(server, scheduler, 41)
            outcome = await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=(5, 6, 7, 8),
                sleep=_no_sleep,
            )
    finally:
        await client.close()

    assert outcome.link is None
    assert outcome.transfer_ms == pytest.approx(
        7 + outcome.payload_bytes / (4e9) * 1000
    )


@pytest.mark.parametrize("mismatch", ["protocol", "model"])
async def test_inconsistent_local_ticket_is_rejected_before_network(
    server_args,
    unused_port,
    mismatch,
):
    scheduler = _scheduler(server_args)
    server = BootstrapServer(
        "127.0.0.1",
        unused_port,
        kv_return_managers=[scheduler.kv_return],
    )
    metadata = _publish(server, scheduler, 42)
    decode = _config(server_args, "decode")
    if mismatch == "protocol":
        metadata["kv_return"]["compatibility"]["protocol_version"] = 2
    else:
        decode.served_model_name = "other-model"
    client = BootstrapClient(0.001, 1)
    with pytest.raises(KVReturnProtocolError, match="incompatible"):
        await return_generated_kv(
            decode,
            client,
            host="127.0.0.1",
            port=unused_port,
            room_metadata=metadata,
            generated_token_ids=(5, 6, 7, 8),
            sleep=_no_sleep,
        )
    assert client._session is None


async def test_transport_failures_are_not_converted_to_success(
    server_args,
    unused_port,
):
    scheduler = _scheduler(server_args)
    server = BootstrapServer(
        "127.0.0.1",
        unused_port,
        kv_return_managers=[scheduler.kv_return],
    )
    metadata = _publish(server, scheduler, 43)
    decode = _config(server_args, "decode")
    client = BootstrapClient(0.001, 0.05)
    try:
        with pytest.raises(KVReturnTransportError, match="prepare request"):
            await return_generated_kv(
                decode,
                client,
                host="127.0.0.1",
                port=unused_port,
                room_metadata=metadata,
                generated_token_ids=(5, 6, 7, 8),
                sleep=_no_sleep,
            )
    finally:
        await client.close()
