"""Decode-side control and simulated transport for reverse KV return."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from fakeengine.bootstrap import (
    KVReturnControlResponse,
    KVReturnControlStatus,
    KVReturnProtocolError,
    parse_kv_return_ticket,
)
from fakeengine.config import SimConfig
from fakeengine.kv_return import (
    KV_RETURN_MAX_INTEGER,
    KVReturnAbort,
    KVReturnCommit,
    KVReturnPrepare,
    KVReturnTicket,
    kv_return_checksum,
    kv_return_compatibility,
)
from fakeengine.timing import TimingModel
from fakeengine.topology import Link

logger = logging.getLogger(__name__)


class KVReturnControlClient(Protocol):
    timeout_s: float

    async def prepare_kv_return(
        self,
        host: str,
        port: int,
        prepare: KVReturnPrepare,
        *,
        deadline: float | None = None,
    ) -> KVReturnControlResponse:
        ...

    async def commit_kv_return(
        self,
        host: str,
        port: int,
        commit: KVReturnCommit,
        *,
        deadline: float | None = None,
    ) -> KVReturnControlResponse:
        ...

    async def abort_kv_return(
        self,
        host: str,
        port: int,
        abort: KVReturnAbort,
        *,
        deadline: float | None = None,
    ) -> KVReturnControlResponse:
        ...


class KVReturnOutcomeStatus(str, Enum):
    COMMITTED = "committed"
    PREPARE_REJECTED = "prepare_rejected"
    COMMIT_REJECTED = "commit_rejected"


@dataclass(frozen=True)
class KVReturnOutcome:
    status: KVReturnOutcomeStatus
    attempt_id: str
    generated_token_count: int
    payload_bytes: int
    transfer_ms: float
    link: str | None
    prepare_response: KVReturnControlResponse
    commit_response: KVReturnControlResponse | None = None


async def return_generated_kv(
    config: SimConfig,
    client: KVReturnControlClient,
    *,
    host: str,
    port: int,
    room_metadata: Mapping[str, Any],
    generated_token_ids: Sequence[int],
    cache_salt: str | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> KVReturnOutcome:
    """Prepare, simulate generated-KV transfer, and commit one exact ticket."""
    started_at = time.monotonic()
    ticket, generated, prefill_location = _validate_operation(
        config,
        room_metadata,
        generated_token_ids,
        cache_salt=cache_salt,
    )
    deadline = started_at + min(
        client.timeout_s,
        _ticket_budget_s(ticket),
    )
    if deadline <= time.monotonic():
        raise TimeoutError("KV return ticket expired before prepare")

    prepare = KVReturnPrepare(
        ticket=ticket,
        attempt_id=ticket.attempt_id,
        prefill_dp_rank=ticket.compatibility.prefill_dp_rank,
        compatibility_fingerprint=ticket.compatibility.fingerprint,
        cache_salt=ticket.cache_salt,
        generated_token_count=len(generated),
        claimed_payload_bytes=(
            len(generated) * config.kv_return_bytes_per_token
        ),
        token_ids_checksum=kv_return_checksum(ticket, generated),
    )
    try:
        prepared = await client.prepare_kv_return(
            host,
            port,
            prepare,
            deadline=deadline,
        )
    except BaseException:
        await _best_effort_abort(client, host, port, ticket)
        raise
    payload_bytes = prepare.claimed_payload_bytes
    if (
        prepared.status == KVReturnControlStatus.COMMITTED
        and prepared.idempotent
    ):
        return KVReturnOutcome(
            status=KVReturnOutcomeStatus.COMMITTED,
            attempt_id=ticket.attempt_id,
            generated_token_count=len(generated),
            payload_bytes=payload_bytes,
            transfer_ms=0.0,
            link=None,
            prepare_response=prepared,
        )
    if prepared.status != KVReturnControlStatus.READY:
        return KVReturnOutcome(
            status=KVReturnOutcomeStatus.PREPARE_REJECTED,
            attempt_id=ticket.attempt_id,
            generated_token_count=len(generated),
            payload_bytes=payload_bytes,
            transfer_ms=0.0,
            link=None,
            prepare_response=prepared,
        )

    try:
        link = _reverse_link(config, prefill_location)
        transfer_ms = TimingModel(config).kv_return_transfer_ms(
            len(generated),
            link,
        )
        if not math.isfinite(transfer_ms) or transfer_ms < 0:
            raise KVReturnProtocolError(
                "KV return transfer duration must be finite and nonnegative"
            )
        sleep_seconds = transfer_ms / 1000.0 / config.speedup_ratio
        if sleep_seconds > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "KV return deadline expired before simulated transfer"
                )
            try:
                await asyncio.wait_for(
                    sleep(sleep_seconds),
                    timeout=remaining,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    "timed out during simulated KV return transfer"
                ) from exc
        if time.monotonic() >= deadline:
            raise TimeoutError("KV return deadline expired before commit")
        committed = await client.commit_kv_return(
            host,
            port,
            KVReturnCommit(prepare, generated),
            deadline=deadline,
        )
    except BaseException:
        await _best_effort_abort(client, host, port, ticket)
        raise

    if committed.status != KVReturnControlStatus.COMMITTED:
        await _best_effort_abort(client, host, port, ticket)
        return KVReturnOutcome(
            status=KVReturnOutcomeStatus.COMMIT_REJECTED,
            attempt_id=ticket.attempt_id,
            generated_token_count=len(generated),
            payload_bytes=payload_bytes,
            transfer_ms=transfer_ms,
            link=link.name if link is not None else None,
            prepare_response=prepared,
            commit_response=committed,
        )
    return KVReturnOutcome(
        status=KVReturnOutcomeStatus.COMMITTED,
        attempt_id=ticket.attempt_id,
        generated_token_count=len(generated),
        payload_bytes=payload_bytes,
        transfer_ms=transfer_ms,
        link=link.name if link is not None else None,
        prepare_response=prepared,
        commit_response=committed,
    )


def _validate_operation(
    config: SimConfig,
    room_metadata: Mapping[str, Any],
    generated_token_ids: Sequence[int],
    *,
    cache_salt: str | None,
) -> tuple[KVReturnTicket, tuple[int, ...], str | None]:
    if not isinstance(room_metadata, Mapping):
        raise KVReturnProtocolError("room metadata must be an object")
    if not config.enable_kv_return or config.disaggregation_mode != "decode":
        raise KVReturnProtocolError(
            "local decode worker is not enabled for KV return"
        )
    if "kv_return" not in room_metadata:
        raise KVReturnProtocolError("room metadata has no kv_return ticket")
    ticket = parse_kv_return_ticket(room_metadata["kv_return"])
    if ticket.cache_salt != (cache_salt or None):
        raise KVReturnProtocolError(
            "KV return ticket cache salt does not match the decode request"
        )
    local = kv_return_compatibility(
        config,
        ticket.compatibility.prefill_dp_rank,
    )
    if ticket.compatibility != local:
        raise KVReturnProtocolError(
            "KV return ticket is incompatible with the local decode worker"
        )
    if ticket.attempt_id != ticket.attempt_id.strip():
        raise KVReturnProtocolError("KV return attempt ID is not canonical")
    room = room_metadata.get("bootstrap_room")
    if (
        isinstance(room, bool)
        or not isinstance(room, int)
        or room != ticket.room_id
    ):
        raise KVReturnProtocolError(
            "KV return ticket room does not match room metadata"
        )
    expected_max_payload = (
        ticket.max_generated_tokens * ticket.compatibility.kv_bytes_per_token
    )
    if (
        expected_max_payload > KV_RETURN_MAX_INTEGER
        or ticket.max_payload_bytes != expected_max_payload
    ):
        raise KVReturnProtocolError(
            "KV return ticket payload limits are inconsistent"
        )

    try:
        generated = tuple(generated_token_ids)
    except TypeError as exc:
        raise KVReturnProtocolError(
            "generated token IDs must be a finite sequence"
        ) from exc
    if len(generated) > ticket.max_generated_tokens:
        raise KVReturnProtocolError(
            "generated token count exceeds the KV return ticket"
        )
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or not 0 <= token < config.vocab_size
        for token in generated
    ):
        raise KVReturnProtocolError(
            "generated token IDs are incompatible with the local vocabulary"
        )
    payload_bytes = len(generated) * config.kv_return_bytes_per_token
    if payload_bytes > ticket.max_payload_bytes:
        raise KVReturnProtocolError(
            "generated KV payload exceeds the KV return ticket"
        )

    prefill_location = room_metadata.get("location")
    if prefill_location is not None and (
        not isinstance(prefill_location, str) or not prefill_location
    ):
        raise KVReturnProtocolError(
            "prefill location must be null or a nonempty string"
        )
    return ticket, generated, prefill_location


def _ticket_budget_s(ticket: KVReturnTicket) -> float:
    budget = ticket.deadline_ms / 1000.0
    if not math.isfinite(budget) or budget <= 0:
        raise KVReturnProtocolError(
            "KV return ticket deadline must be a finite positive duration"
        )
    return budget


def _reverse_link(
    config: SimConfig,
    prefill_location: str | None,
) -> Link | None:
    topology = config.topology
    if topology is None or not config.location or not prefill_location:
        return None
    return topology.link_between(config.location, prefill_location)


async def _best_effort_abort(
    client: KVReturnControlClient,
    host: str,
    port: int,
    ticket: KVReturnTicket,
) -> None:
    cleanup_timeout = min(client.timeout_s, 0.25)
    try:
        await client.abort_kv_return(
            host,
            port,
            KVReturnAbort(ticket, ticket.attempt_id),
            deadline=time.monotonic() + cleanup_timeout,
        )
    except BaseException as exc:  # noqa: BLE001 - cleanup includes cancellation
        logger.debug(
            "fake-sglang: best-effort KV return abort failed (%s)",
            type(exc).__name__,
        )
