"""Decode-side control and simulated transport for reverse KV return."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import logging
import math
import time
from typing import (
    Any,
    Awaitable,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

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
    KVReturnTransferTicket,
    kv_return_checksum,
    kv_return_compatibility,
)
from fakeengine.timing import TimingModel
from fakeengine.topology import Link

logger = logging.getLogger(__name__)


class KVReturnLeaseLostError(RuntimeError):
    """The decode task no longer owns the retained source KV."""


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
    prepare_ms: float
    transfer_ms: float
    commit_ms: float
    link: Optional[str]
    prepare_response: KVReturnControlResponse
    commit_response: Optional[KVReturnControlResponse] = None


@dataclass(frozen=True)
class KVReturnAttemptContext:
    ticket: KVReturnTicket
    prefill_location: Optional[str]


def validate_kv_return_attempt(
    config: SimConfig,
    room_metadata: Mapping[str, Any],
    *,
    cache_salt: Optional[str],
) -> KVReturnAttemptContext:
    """Validate the forward-room ticket before decode retains response KV."""
    ticket, prefill_location = _validate_ticket(config, room_metadata)
    if ticket.cache_salt != (cache_salt or None):
        raise KVReturnProtocolError(
            "KV return ticket cache salt does not match the decode request"
        )
    return KVReturnAttemptContext(ticket, prefill_location)


async def return_generated_kv(
    config: SimConfig,
    client: KVReturnControlClient,
    *,
    host: str,
    port: int,
    room_metadata: Mapping[str, Any],
    generated_token_ids: Sequence[int],
    cache_salt: Optional[str] = None,
    attempt: Optional[KVReturnAttemptContext] = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    task_deadline: Optional[float] = None,
    owns_source: Optional[Callable[[], bool]] = None,
) -> KVReturnOutcome:
    """Prepare, simulate generated-KV transfer, and commit one exact ticket."""
    started_at = time.monotonic()
    context = attempt or validate_kv_return_attempt(
        config,
        room_metadata,
        cache_salt=cache_salt,
    )
    ticket = context.ticket
    generated = validate_kv_return_generated(
        config, context, generated_token_ids,
    )
    deadline = started_at + client.timeout_s
    if task_deadline is not None:
        if (
            isinstance(task_deadline, bool)
            or not isinstance(task_deadline, (int, float))
            or not math.isfinite(task_deadline)
        ):
            raise KVReturnProtocolError(
                "KV return task deadline must be a finite monotonic timestamp"
            )
        deadline = min(deadline, float(task_deadline))
    _require_source_ownership(owns_source)
    if deadline <= time.monotonic():
        raise TimeoutError("KV return task deadline expired before prepare")

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
    prepare_started = time.monotonic()
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
    prepare_ms = (time.monotonic() - prepare_started) * 1000.0
    payload_bytes = prepare.claimed_payload_bytes
    if (
        prepared.status == KVReturnControlStatus.COMMITTED
        and prepared.idempotent
    ):
        try:
            _require_source_ownership(owns_source)
        except BaseException:
            await _best_effort_abort(client, host, port, ticket)
            raise
        return KVReturnOutcome(
            status=KVReturnOutcomeStatus.COMMITTED,
            attempt_id=ticket.attempt_id,
            generated_token_count=len(generated),
            payload_bytes=payload_bytes,
            prepare_ms=prepare_ms,
            transfer_ms=0.0,
            commit_ms=0.0,
            link=None,
            prepare_response=prepared,
        )
    if prepared.status != KVReturnControlStatus.READY:
        return KVReturnOutcome(
            status=KVReturnOutcomeStatus.PREPARE_REJECTED,
            attempt_id=ticket.attempt_id,
            generated_token_count=len(generated),
            payload_bytes=payload_bytes,
            prepare_ms=prepare_ms,
            transfer_ms=0.0,
            commit_ms=0.0,
            link=None,
            prepare_response=prepared,
        )
    transfer_ticket = prepared.transfer_ticket
    if transfer_ticket is None:
        await _best_effort_abort(client, host, port, ticket)
        raise KVReturnProtocolError(
            "KV return prepare response omitted its transfer ticket"
        )
    if (
        transfer_ticket.handle_nonce != ticket.nonce
        or transfer_ticket.attempt_id != ticket.attempt_id
    ):
        await _best_effort_abort(client, host, port, ticket)
        raise KVReturnProtocolError(
            "KV return transfer ticket does not match its handoff handle"
        )
    try:
        deadline = time.monotonic() + min(
            client.timeout_s,
            _transfer_ticket_budget_s(transfer_ticket),
        )
    except BaseException:
        await _best_effort_abort(client, host, port, ticket)
        raise
    if task_deadline is not None:
        deadline = min(deadline, float(task_deadline))

    try:
        _require_source_ownership(owns_source)
        link = _reverse_link(config, context.prefill_location)
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
        _require_source_ownership(owns_source)
        commit_started = time.monotonic()
        committed = await client.commit_kv_return(
            host,
            port,
            KVReturnCommit(prepare, transfer_ticket, generated),
            deadline=deadline,
        )
        _require_source_ownership(owns_source)
    except BaseException:
        await _best_effort_abort(client, host, port, ticket)
        raise
    commit_ms = (time.monotonic() - commit_started) * 1000.0

    if committed.status != KVReturnControlStatus.COMMITTED:
        await _best_effort_abort(client, host, port, ticket)
        return KVReturnOutcome(
            status=KVReturnOutcomeStatus.COMMIT_REJECTED,
            attempt_id=ticket.attempt_id,
            generated_token_count=len(generated),
            payload_bytes=payload_bytes,
            prepare_ms=prepare_ms,
            transfer_ms=transfer_ms,
            commit_ms=commit_ms,
            link=link.name if link is not None else None,
            prepare_response=prepared,
            commit_response=committed,
        )
    return KVReturnOutcome(
        status=KVReturnOutcomeStatus.COMMITTED,
        attempt_id=ticket.attempt_id,
        generated_token_count=len(generated),
        payload_bytes=payload_bytes,
        prepare_ms=prepare_ms,
        transfer_ms=transfer_ms,
        commit_ms=commit_ms,
        link=link.name if link is not None else None,
        prepare_response=prepared,
        commit_response=committed,
    )


def _validate_ticket(
    config: SimConfig,
    room_metadata: Mapping[str, Any],
) -> Tuple[KVReturnTicket, Optional[str]]:
    if not isinstance(room_metadata, Mapping):
        raise KVReturnProtocolError("room metadata must be an object")
    if not config.enable_kv_return or config.disaggregation_mode != "decode":
        raise KVReturnProtocolError(
            "local decode worker is not enabled for KV return"
        )
    if "kv_return" not in room_metadata:
        raise KVReturnProtocolError("room metadata has no kv_return ticket")
    ticket = parse_kv_return_ticket(room_metadata["kv_return"])
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

    prefill_location = room_metadata.get("location")
    if prefill_location is not None and (
        not isinstance(prefill_location, str) or not prefill_location
    ):
        raise KVReturnProtocolError(
            "prefill location must be null or a nonempty string"
        )
    return ticket, prefill_location


def validate_kv_return_generated(
    config: SimConfig,
    attempt: KVReturnAttemptContext,
    generated_token_ids: Sequence[int],
) -> Tuple[int, ...]:
    """Validate a retained response against its immutable attempt context."""
    ticket = attempt.ticket
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
    return generated


def _require_source_ownership(
    owns_source: Optional[Callable[[], bool]],
) -> None:
    if owns_source is not None and not owns_source():
        raise KVReturnLeaseLostError(
            "retained response KV lease ownership was lost"
        )


def _transfer_ticket_budget_s(ticket: KVReturnTransferTicket) -> float:
    budget = ticket.deadline_ms / 1000.0
    if not math.isfinite(budget) or budget <= 0:
        raise KVReturnProtocolError(
            "KV return transfer-ticket deadline must be a finite positive "
            "duration"
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
