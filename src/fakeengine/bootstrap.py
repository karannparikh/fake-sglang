"""The prefill/decode rendezvous.

This is a *real* cross-process handshake over HTTP, not a mocked one. That
matters: it is the part of disaggregation most likely to be misconfigured in a
deployment (wrong advertised host, port collision, dead peer), and a fake that
skipped it would hide exactly the failures worth rehearsing. Only the KV
payload itself is simulated -- see :mod:`fakeengine.transfer`.

The route shape follows SGLang's ``CommonKVBootstrapServer``
(``disaggregation/common/conn.py``): ``GET /health``, ``PUT /route`` for rank
registration, ``GET /route`` for topology. ``/room/{id}`` is ours -- SGLang
carries per-request metadata over ZMQ rather than HTTP, and HTTP keeps the fake
engine dependency-light.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
import math
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from aiohttp import web

from fakeengine.kv_return import (
    KV_RETURN_FINGERPRINT_LENGTH,
    KV_RETURN_MAX_IDENTIFIER_LENGTH,
    KV_RETURN_MAX_INTEGER,
    KV_RETURN_MAX_METADATA_LENGTH,
    KVReturnAbort,
    KVReturnCommit,
    KVReturnCompatibility,
    KVReturnManager,
    KVReturnPrepare,
    KVReturnResult,
    KVReturnStatus,
    KVReturnTicket,
)

logger = logging.getLogger(__name__)

_KV_RETURN_MAX_BODY_BYTES = 1 << 20
_KV_RETURN_MAX_RESPONSE_BYTES = 1 << 16


class KVReturnClientError(RuntimeError):
    """Base class for reverse-return client failures."""


class KVReturnTransportError(KVReturnClientError):
    """The reverse-return peer could not be reached."""


class KVReturnProtocolError(KVReturnClientError):
    """The peer or room metadata violated the reverse-return protocol."""


class KVReturnControlStatus(str, Enum):
    READY = "ready"
    COMMITTED = "committed"
    ABORTED = "aborted"
    DISABLED = "disabled"
    EXPIRED = "expired"
    CAPACITY_REJECTED = "capacity_rejected"
    INCOMPATIBLE = "incompatible"
    INVALID_TOKEN_IDS = "invalid_token_ids"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    NOT_PREPARED = "not_prepared"
    IMPORT_FAILED = "import_failed"
    NOT_FOUND = "not_found"
    INVALID_REQUEST = "invalid_request"


@dataclass(frozen=True)
class KVReturnControlResponse:
    operation: str
    protocol_version: int
    http_status: int
    status: KVReturnControlStatus
    reason: str
    idempotent: bool
    reserved_blocks: int


class _ControlInputError(ValueError):
    def __init__(self, reason: str, http_status: int = 400) -> None:
        super().__init__(reason)
        self.reason = reason
        self.http_status = http_status


class BootstrapServer:
    """Runs in the prefill worker. Publishes rooms as their KV becomes ready."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        room_ttl_s: float = 120.0,
        kv_return_managers: Sequence[KVReturnManager] = (),
    ) -> None:
        self.host = host
        self.port = port
        self.room_ttl_s = room_ttl_s
        self._rooms: Dict[int, Dict[str, Any]] = {}
        self._ranks: Dict[str, Dict[str, Any]] = {}
        self._kv_return_managers = {
            manager.dp_rank: manager for manager in kv_return_managers
        }
        if len(self._kv_return_managers) != len(kv_return_managers):
            raise ValueError("KV return managers must have unique DP ranks")
        self._room_returns: Dict[int, Tuple[KVReturnManager, KVReturnTicket]] = {}
        self._runner: Optional[web.AppRunner] = None
        self._sweeper: Optional[asyncio.Task] = None

    async def start(self) -> None:
        app = web.Application(client_max_size=_KV_RETURN_MAX_BODY_BYTES)
        app.add_routes(
            [
                web.get("/health", self._health),
                web.get("/route", self._get_route),
                web.put("/route", self._put_route),
                web.get("/room/{room}", self._get_room),
                web.delete("/room/{room}", self._delete_room),
                web.post(
                    "/kv-return/v{version}/prepare",
                    self._prepare_kv_return,
                ),
                web.post(
                    "/kv-return/v{version}/commit",
                    self._commit_kv_return,
                ),
                web.post(
                    "/kv-return/v{version}/abort",
                    self._abort_kv_return,
                ),
            ]
        )
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        # Bind every interface but advertise a specific host: a decode worker
        # may reach us over loopback or the LAN address, and which one it uses
        # is decided by whatever Dynamo published in the model card.
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        await site.start()
        self._sweeper = asyncio.get_running_loop().create_task(self._sweep_rooms())
        logger.info(
            "fake-sglang bootstrap server listening on 0.0.0.0:%d (advertising %s)",
            self.port,
            self.host,
        )

    async def stop(self) -> None:
        try:
            sweeper = self._sweeper
            self._sweeper = None
            if sweeper is not None:
                sweeper.cancel()
                try:
                    await sweeper
                except asyncio.CancelledError:
                    pass
            runner = self._runner
            self._runner = None
            if runner is not None:
                await runner.cleanup()
        finally:
            for manager in self._kv_return_managers.values():
                manager.abort_all()
            self._room_returns.clear()
            self._rooms.clear()

    async def _sweep_rooms(self) -> None:
        """Drop rooms whose decode peer never came.

        A room is normally released by the receiver once the transfer lands. If
        the decode worker dies, is never scheduled, or the request is cancelled
        between the two legs, nothing releases it -- and a long-lived prefill
        worker would accumulate them forever. SGLang sweeps the same way, on
        SGLANG_DISAGGREGATION_WAITING_TIMEOUT.
        """
        interval = max(min(self.room_ttl_s / 4, 1.0), 0.01)
        deadlines = [
            manager.config.kv_return_deadline_ms / 1000.0
            for manager in self._kv_return_managers.values()
            if manager.config.enable_kv_return
        ]
        if deadlines:
            interval = max(min(interval, min(deadlines)), 0.01)
        while True:
            await asyncio.sleep(interval)
            for manager in self._kv_return_managers.values():
                manager.expire()
            cutoff = time.time() - self.room_ttl_s
            stale = [
                room
                for room, entry in self._rooms.items()
                if entry.get("ready_ts", 0) < cutoff
            ]
            for room in stale:
                self._rooms.pop(room, None)
                self._clear_room_return(room)
            if stale:
                logger.warning(
                    "fake-sglang: expired %d bootstrap room(s) never claimed by a "
                    "decode peer (oldest > %.0fs): %s",
                    len(stale),
                    self.room_ttl_s,
                    stale[:8],
                )

    def publish_room(
        self,
        room: int,
        *,
        num_tokens: int,
        kv_bytes: int,
        first_token: Optional[int] = None,
        location: Optional[str] = None,
        prefill_dp_rank: Optional[int] = None,
        kv_return_prompt_ids: Optional[Sequence[int]] = None,
        kv_return_cache_salt: Optional[str] = None,
    ) -> None:
        # ``first_token`` is prefill's real output. SGLang hands the same value
        # to decode through its metadata buffer rather than the KV payload
        # (``disaggregation/decode.py::_commit_transfer_to_req``); carrying it
        # here is what makes a disaggregated answer identical to an aggregated
        # one for the same prompt.
        room_id = int(room)
        self._clear_room_return(room_id)
        metadata: Dict[str, Any] = {
            "bootstrap_room": room_id,
            "num_tokens": int(num_tokens),
            "kv_bytes": int(kv_bytes),
            "first_token": first_token,
            "location": location,
            "ready_ts": time.time(),
        }
        if prefill_dp_rank is not None and kv_return_prompt_ids is not None:
            manager = self._kv_return_managers.get(prefill_dp_rank)
            if manager is not None:
                result = manager.issue_ticket(
                    room_id,
                    kv_return_prompt_ids,
                    kv_return_cache_salt,
                )
                if result.status == KVReturnStatus.ISSUED:
                    assert result.ticket is not None
                    metadata["kv_return"] = asdict(result.ticket)
                    self._room_returns[room_id] = (manager, result.ticket)
                elif result.status != KVReturnStatus.DISABLED:
                    logger.warning(
                        "fake-sglang: KV return ticket rejected for room %d "
                        "rank %d: %s",
                        room_id,
                        prefill_dp_rank,
                        result.status.value,
                    )
        self._rooms[room_id] = metadata

    def fail_room(self, room: int, reason: str) -> None:
        room_id = int(room)
        self._clear_room_return(room_id)
        self._rooms[room_id] = {
            "bootstrap_room": room_id,
            "error": reason,
            "ready_ts": time.time(),
        }

    def _clear_room_return(self, room: int) -> None:
        tracked = self._room_returns.pop(room, None)
        if tracked is not None:
            manager, ticket = tracked
            manager.clear_ticket(ticket)

    # -- handlers ----------------------------------------------------------

    async def _health(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {"status": "ok", "engine": "fake-sglang", "rooms": len(self._rooms)}
        )

    async def _get_route(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "engine": "fake-sglang",
                "prefill_host": self.host,
                "prefill_port": self.port,
                "ranks": self._ranks,
                "follow_bootstrap_room": True,
            }
        )

    async def _put_route(self, request: web.Request) -> web.Response:
        payload = await request.json()
        key = str(payload.get("rank_id", len(self._ranks)))
        self._ranks[key] = payload
        return web.json_response({"status": "registered", "rank_id": key})

    async def _get_room(self, request: web.Request) -> web.Response:
        room = int(request.match_info["room"])
        entry = self._rooms.get(room)
        if entry is None:
            # 404 is the "still bootstrapping" signal, not an error: the decode
            # worker can legitimately arrive before prefill has finished.
            return web.json_response({"status": "pending", "bootstrap_room": room}, status=404)
        if "error" in entry:
            return web.json_response(entry, status=410)
        return web.json_response(entry)

    async def _delete_room(self, request: web.Request) -> web.Response:
        room = int(request.match_info["room"])
        self._rooms.pop(room, None)
        self._room_returns.pop(room, None)
        return web.json_response({"status": "released", "bootstrap_room": room})

    async def _prepare_kv_return(self, request: web.Request) -> web.Response:
        try:
            version = _protocol_version(request)
            prepare = _parse_prepare(await _read_control_json(request))
        except _ControlInputError as exc:
            return _control_error(exc)
        manager, rejection = self._manager_for(version, prepare.ticket)
        if rejection is not None:
            return rejection
        assert manager is not None
        if prepare.prefill_dp_rank != manager.dp_rank:
            return _control_response(
                "incompatible", "prefill_dp_rank_mismatch", 409,
            )
        return _manager_response(manager.prepare(prepare))

    async def _commit_kv_return(self, request: web.Request) -> web.Response:
        try:
            version = _protocol_version(request)
            commit = _parse_commit(await _read_control_json(request))
        except _ControlInputError as exc:
            return _control_error(exc)
        manager, rejection = self._manager_for(version, commit.prepare.ticket)
        if rejection is not None:
            return rejection
        assert manager is not None
        if commit.prepare.prefill_dp_rank != manager.dp_rank:
            return _control_response(
                "incompatible", "prefill_dp_rank_mismatch", 409,
            )
        return _manager_response(manager.commit(commit))

    async def _abort_kv_return(self, request: web.Request) -> web.Response:
        try:
            version = _protocol_version(request)
            abort = _parse_abort(await _read_control_json(request))
        except _ControlInputError as exc:
            return _control_error(exc)
        manager, rejection = self._manager_for(version, abort.ticket)
        if rejection is not None:
            return rejection
        assert manager is not None
        return _manager_response(manager.abort(abort))

    def _manager_for(
        self,
        version: int,
        ticket: KVReturnTicket,
    ) -> Tuple[Optional[KVReturnManager], Optional[web.Response]]:
        manager = self._kv_return_managers.get(
            ticket.compatibility.prefill_dp_rank
        )
        if manager is None:
            return None, _control_response(
                "incompatible", "prefill_dp_rank_not_hosted", 409,
            )
        if version != manager.compatibility.protocol_version:
            return None, _control_response(
                "incompatible", "unsupported_protocol_version", 409,
            )
        config = manager.config
        if not (
            config.enable_kv_return
            and config.disaggregation_mode == "prefill"
            and config.prefix_caching
        ):
            return None, _control_response(
                "disabled", "kv_return_disabled", 403,
            )
        return manager, None


async def _read_control_json(request: web.Request) -> Mapping[str, Any]:
    if request.content_type != "application/json":
        raise _ControlInputError("content_type_must_be_application_json")
    if (
        request.content_length is not None
        and request.content_length > _KV_RETURN_MAX_BODY_BYTES
    ):
        raise _ControlInputError("request_body_too_large", 413)
    try:
        raw = await request.read()
    except web.HTTPRequestEntityTooLarge as exc:
        raise _ControlInputError("request_body_too_large", 413) from exc
    if len(raw) > _KV_RETURN_MAX_BODY_BYTES:
        raise _ControlInputError("request_body_too_large", 413)
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError, OverflowError, RecursionError) as exc:
        raise _ControlInputError("malformed_json") from exc
    if not isinstance(payload, dict):
        raise _ControlInputError("request_body_must_be_an_object")
    return payload


def _protocol_version(request: web.Request) -> int:
    raw = request.match_info["version"]
    if not raw.isascii() or not raw.isdecimal():
        raise _ControlInputError("protocol_version_must_be_a_nonnegative_integer")
    try:
        version = int(raw)
    except (ValueError, OverflowError) as exc:
        raise _ControlInputError(
            "protocol_version_must_be_a_nonnegative_integer"
        ) from exc
    if version > KV_RETURN_MAX_INTEGER:
        raise _ControlInputError("protocol_version_exceeds_protocol_limit")
    return version


def _parse_prepare(value: Any) -> KVReturnPrepare:
    payload = _object(
        value,
        "prepare",
        {
            "ticket",
            "attempt_id",
            "prefill_dp_rank",
            "compatibility_fingerprint",
            "cache_salt",
            "generated_token_count",
            "claimed_payload_bytes",
            "token_ids_checksum",
        },
    )
    return KVReturnPrepare(
        ticket=_parse_ticket(payload["ticket"]),
        attempt_id=_string(
            payload["attempt_id"],
            "attempt_id",
            max_length=KV_RETURN_MAX_IDENTIFIER_LENGTH,
        ),
        prefill_dp_rank=_integer(
            payload["prefill_dp_rank"], "prefill_dp_rank", minimum=0,
        ),
        compatibility_fingerprint=_hex(
            payload["compatibility_fingerprint"],
            "compatibility_fingerprint",
        ),
        cache_salt=_optional_string(
            payload["cache_salt"],
            "cache_salt",
            max_length=KV_RETURN_MAX_METADATA_LENGTH,
        ),
        generated_token_count=_integer(
            payload["generated_token_count"],
            "generated_token_count",
            minimum=0,
        ),
        claimed_payload_bytes=_integer(
            payload["claimed_payload_bytes"],
            "claimed_payload_bytes",
            minimum=0,
        ),
        token_ids_checksum=_hex(
            payload["token_ids_checksum"], "token_ids_checksum",
        ),
    )


def _parse_commit(value: Any) -> KVReturnCommit:
    payload = _object(
        value, "commit", {"prepare", "generated_token_ids"},
    )
    prepare = _parse_prepare(payload["prepare"])
    raw_tokens = payload["generated_token_ids"]
    if not isinstance(raw_tokens, list):
        raise _ControlInputError("generated_token_ids_must_be_an_array")
    if len(raw_tokens) > prepare.ticket.max_generated_tokens:
        raise _ControlInputError("generated_token_ids_exceed_ticket_limit", 413)
    return KVReturnCommit(
        prepare,
        tuple(
            _integer(token, "generated_token_ids_item", minimum=0)
            for token in raw_tokens
        ),
    )


def _parse_abort(value: Any) -> KVReturnAbort:
    payload = _object(value, "abort", {"ticket", "attempt_id"})
    return KVReturnAbort(
        _parse_ticket(payload["ticket"]),
        _string(
            payload["attempt_id"],
            "attempt_id",
            max_length=KV_RETURN_MAX_IDENTIFIER_LENGTH,
        ),
    )


def _parse_ticket(value: Any) -> KVReturnTicket:
    payload = _object(
        value,
        "ticket",
        {
            "nonce",
            "room_id",
            "attempt_id",
            "compatibility",
            "cache_salt",
            "max_generated_tokens",
            "max_payload_bytes",
            "deadline_ms",
        },
    )
    return KVReturnTicket(
        nonce=_string(
            payload["nonce"],
            "ticket_nonce",
            max_length=KV_RETURN_MAX_IDENTIFIER_LENGTH,
        ),
        room_id=_integer(payload["room_id"], "ticket_room_id", minimum=0),
        attempt_id=_string(
            payload["attempt_id"],
            "ticket_attempt_id",
            max_length=KV_RETURN_MAX_IDENTIFIER_LENGTH,
        ),
        compatibility=_parse_compatibility(payload["compatibility"]),
        cache_salt=_optional_string(
            payload["cache_salt"],
            "ticket_cache_salt",
            max_length=KV_RETURN_MAX_METADATA_LENGTH,
        ),
        max_generated_tokens=_integer(
            payload["max_generated_tokens"],
            "ticket_max_generated_tokens",
            minimum=1,
        ),
        max_payload_bytes=_integer(
            payload["max_payload_bytes"],
            "ticket_max_payload_bytes",
            minimum=1,
        ),
        deadline_ms=_finite_number(
            payload["deadline_ms"], "ticket_deadline_ms", positive=True,
        ),
    )


def parse_kv_return_ticket(value: Any) -> KVReturnTicket:
    """Strictly parse the ticket embedded in forward room metadata."""
    try:
        return _parse_ticket(value)
    except _ControlInputError as exc:
        raise KVReturnProtocolError(
            f"invalid kv_return ticket: {exc.reason}"
        ) from exc


def _parse_compatibility(value: Any) -> KVReturnCompatibility:
    payload = _object(
        value,
        "compatibility",
        {
            "protocol_version",
            "prefill_dp_rank",
            "served_model_name",
            "model_identity",
            "vocab_size",
            "block_size",
            "kv_bytes_per_token",
            "kv_layout_fingerprint",
        },
    )
    return KVReturnCompatibility(
        protocol_version=_integer(
            payload["protocol_version"],
            "compatibility_protocol_version",
            minimum=0,
        ),
        prefill_dp_rank=_integer(
            payload["prefill_dp_rank"],
            "compatibility_prefill_dp_rank",
            minimum=0,
        ),
        served_model_name=_string(
            payload["served_model_name"],
            "compatibility_served_model_name",
            max_length=KV_RETURN_MAX_METADATA_LENGTH,
        ),
        model_identity=_string(
            payload["model_identity"],
            "compatibility_model_identity",
            max_length=KV_RETURN_MAX_METADATA_LENGTH,
        ),
        vocab_size=_integer(
            payload["vocab_size"], "compatibility_vocab_size", minimum=1,
        ),
        block_size=_integer(
            payload["block_size"], "compatibility_block_size", minimum=1,
        ),
        kv_bytes_per_token=_integer(
            payload["kv_bytes_per_token"],
            "compatibility_kv_bytes_per_token",
            minimum=1,
        ),
        kv_layout_fingerprint=_hex(
            payload["kv_layout_fingerprint"],
            "compatibility_kv_layout_fingerprint",
        ),
    )


def _object(
    value: Any,
    name: str,
    fields: set,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _ControlInputError(f"{name}_must_be_an_object")
    missing = sorted(fields - set(value))
    unknown = sorted(set(value) - fields)
    if missing:
        raise _ControlInputError(f"{name}_missing_{missing[0]}")
    if unknown:
        raise _ControlInputError(f"{name}_unknown_{unknown[0]}")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _ControlInputError(f"{name}_must_be_an_integer_at_least_{minimum}")
    if value > KV_RETURN_MAX_INTEGER:
        raise _ControlInputError(f"{name}_exceeds_protocol_limit")
    return value


def _finite_number(value: Any, name: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _ControlInputError(f"{name}_must_be_a_finite_number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise _ControlInputError(f"{name}_must_be_a_finite_number") from exc
    if not math.isfinite(result) or positive and result <= 0:
        raise _ControlInputError(f"{name}_must_be_finite_and_positive")
    if result > KV_RETURN_MAX_INTEGER:
        raise _ControlInputError(f"{name}_exceeds_protocol_limit")
    return result


def _string(value: Any, name: str, *, max_length: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > max_length
    ):
        raise _ControlInputError(
            f"{name}_must_be_a_nonempty_bounded_string"
        )
    return value


def _optional_string(
    value: Any, name: str, *, max_length: int,
) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > max_length:
        raise _ControlInputError(f"{name}_must_be_null_or_a_bounded_string")
    return value


def _hex(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != KV_RETURN_FINGERPRINT_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _ControlInputError(f"{name}_must_be_32_lowercase_hex_characters")
    return value


def _control_error(error: _ControlInputError) -> web.Response:
    return _control_response(
        "invalid_request", error.reason, error.http_status,
    )


def _manager_response(result: KVReturnResult) -> web.Response:
    status = (
        "ready"
        if result.status == KVReturnStatus.PREPARED
        else result.status.value
    )
    reason = {
        KVReturnStatus.ISSUED: "ticket_issued",
        KVReturnStatus.DISABLED: "kv_return_disabled",
        KVReturnStatus.PREPARED: "reservation_ready",
        KVReturnStatus.COMMITTED: "prefix_committed",
        KVReturnStatus.ABORTED: "attempt_aborted",
        KVReturnStatus.EXPIRED: "attempt_expired",
        KVReturnStatus.CAPACITY_REJECTED: "insufficient_clean_capacity",
        KVReturnStatus.INCOMPATIBLE: "incompatible_control_contract",
        KVReturnStatus.INVALID_TOKEN_IDS: "invalid_generated_token_ids",
        KVReturnStatus.CHECKSUM_MISMATCH: "token_ids_checksum_mismatch",
        KVReturnStatus.NOT_PREPARED: "attempt_not_prepared",
        KVReturnStatus.IMPORT_FAILED: "atomic_prefix_import_failed",
        KVReturnStatus.NOT_FOUND: "ticket_not_found",
    }[result.status]
    http_status = {
        KVReturnStatus.ISSUED: 409,
        KVReturnStatus.DISABLED: 403,
        KVReturnStatus.PREPARED: 200,
        KVReturnStatus.COMMITTED: 200,
        KVReturnStatus.ABORTED: 200,
        KVReturnStatus.EXPIRED: 410,
        KVReturnStatus.CAPACITY_REJECTED: 409,
        KVReturnStatus.INCOMPATIBLE: 409,
        KVReturnStatus.INVALID_TOKEN_IDS: 422,
        KVReturnStatus.CHECKSUM_MISMATCH: 422,
        KVReturnStatus.NOT_PREPARED: 409,
        KVReturnStatus.IMPORT_FAILED: 409,
        KVReturnStatus.NOT_FOUND: 404,
    }[result.status]
    return _control_response(
        status,
        reason,
        http_status,
        idempotent=result.idempotent,
        reserved_blocks=result.reserved_blocks,
    )


def _control_response(
    status: str,
    reason: str,
    http_status: int,
    *,
    idempotent: bool = False,
    reserved_blocks: int = 0,
) -> web.Response:
    payload = {
        "status": status,
        "reason": reason,
        "idempotent": idempotent,
        "reserved_blocks": reserved_blocks,
    }
    return web.json_response(payload, status=http_status)


class BootstrapClient:
    """Runs in the decode worker. Polls a prefill peer for one room."""

    def __init__(self, poll_interval_s: float, timeout_s: float) -> None:
        if not math.isfinite(poll_interval_s) or poll_interval_s < 0:
            raise ValueError("poll_interval_s must be finite and nonnegative")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._session: Any = None

    async def _get_session(self):
        import aiohttp

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_s + 5)
            )
        return self._session

    async def await_room(self, host: str, port: int, room: int) -> Dict[str, Any]:
        """Block until the prefill peer publishes ``room``.

        Raises ``TimeoutError`` rather than hanging, so a dead or misaddressed
        prefill worker surfaces as a failed request instead of a stuck one.
        """
        import aiohttp

        session = await self._get_session()
        url = f"http://{_bracket(host)}:{port}/room/{room}"
        deadline = time.monotonic() + self.timeout_s

        while True:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    if response.status == 410:
                        payload = await response.json()
                        raise RuntimeError(
                            f"prefill reported failure for room {room}: "
                            f"{payload.get('error')}"
                        )
            except aiohttp.ClientError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not reach prefill bootstrap at {url}: {exc}"
                    ) from exc

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {self.timeout_s:.0f}s waiting for KV of "
                    f"bootstrap_room {room} from {host}:{port}"
                )
            await asyncio.sleep(self.poll_interval_s)

    async def release_room(self, host: str, port: int, room: int) -> None:
        try:
            session = await self._get_session()
            url = f"http://{_bracket(host)}:{port}/room/{room}"
            async with session.delete(url):
                pass
        except Exception as exc:  # noqa: BLE001 - cleanup must never fail a request
            logger.debug("fake-sglang: room %s release failed: %s", room, exc)

    async def prepare_kv_return(
        self,
        host: str,
        port: int,
        prepare: KVReturnPrepare,
        *,
        deadline: Optional[float] = None,
    ) -> KVReturnControlResponse:
        payload = asdict(prepare)
        try:
            if _parse_prepare(payload) != prepare:
                raise _ControlInputError("prepare_did_not_round_trip")
        except _ControlInputError as exc:
            raise KVReturnProtocolError(
                f"invalid KV return prepare request: {exc.reason}"
            ) from exc
        return await self._post_kv_return(
            host,
            port,
            "prepare",
            prepare.ticket.compatibility.protocol_version,
            payload,
            deadline=deadline,
        )

    async def commit_kv_return(
        self,
        host: str,
        port: int,
        commit: KVReturnCommit,
        *,
        deadline: Optional[float] = None,
    ) -> KVReturnControlResponse:
        payload = {
            "prepare": asdict(commit.prepare),
            "generated_token_ids": list(commit.generated_token_ids),
        }
        try:
            if _parse_commit(payload) != commit:
                raise _ControlInputError("commit_did_not_round_trip")
        except _ControlInputError as exc:
            raise KVReturnProtocolError(
                f"invalid KV return commit request: {exc.reason}"
            ) from exc
        return await self._post_kv_return(
            host,
            port,
            "commit",
            commit.prepare.ticket.compatibility.protocol_version,
            payload,
            deadline=deadline,
        )

    async def abort_kv_return(
        self,
        host: str,
        port: int,
        abort: KVReturnAbort,
        *,
        deadline: Optional[float] = None,
    ) -> KVReturnControlResponse:
        payload = asdict(abort)
        try:
            if _parse_abort(payload) != abort:
                raise _ControlInputError("abort_did_not_round_trip")
        except _ControlInputError as exc:
            raise KVReturnProtocolError(
                f"invalid KV return abort request: {exc.reason}"
            ) from exc
        return await self._post_kv_return(
            host,
            port,
            "abort",
            abort.ticket.compatibility.protocol_version,
            payload,
            deadline=deadline,
        )

    async def _post_kv_return(
        self,
        host: str,
        port: int,
        operation: str,
        protocol_version: int,
        payload: Mapping[str, Any],
        *,
        deadline: Optional[float],
    ) -> KVReturnControlResponse:
        import aiohttp

        _validate_peer(host, port)
        if (
            isinstance(protocol_version, bool)
            or not isinstance(protocol_version, int)
            or not 0 <= protocol_version <= KV_RETURN_MAX_INTEGER
        ):
            raise KVReturnProtocolError(
                "KV return protocol version must be a nonnegative bounded integer"
            )
        request_deadline = (
            time.monotonic() + self.timeout_s
            if deadline is None
            else deadline
        )
        remaining = request_deadline - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError(
                f"KV return {operation} deadline expired before the request"
            )

        session = await self._get_session()
        url = (
            f"http://{_bracket(host)}:{port}"
            f"/kv-return/v{protocol_version}/{operation}"
        )

        async def exchange() -> Tuple[int, str, bytes]:
            async with session.post(url, json=payload) as response:
                if (
                    response.content_length is not None
                    and response.content_length > _KV_RETURN_MAX_RESPONSE_BYTES
                ):
                    raise KVReturnProtocolError(
                        f"KV return {operation} response is too large"
                    )
                raw = await response.content.read(
                    _KV_RETURN_MAX_RESPONSE_BYTES + 1
                )
                if len(raw) > _KV_RETURN_MAX_RESPONSE_BYTES:
                    raise KVReturnProtocolError(
                        f"KV return {operation} response is too large"
                    )
                return response.status, response.content_type, raw

        try:
            http_status, content_type, raw = await asyncio.wait_for(
                exchange(),
                timeout=remaining,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"timed out during KV return {operation}"
            ) from exc
        except KVReturnProtocolError:
            raise
        except aiohttp.ClientError as exc:
            raise KVReturnTransportError(
                f"KV return {operation} request to {host}:{port} failed: {exc}"
            ) from exc

        return _parse_kv_return_response(
            operation,
            protocol_version,
            http_status,
            content_type,
            raw,
        )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


_KV_RETURN_HTTP_STATUSES = {
    KVReturnControlStatus.READY: {200},
    KVReturnControlStatus.COMMITTED: {200},
    KVReturnControlStatus.ABORTED: {200},
    KVReturnControlStatus.DISABLED: {403},
    KVReturnControlStatus.EXPIRED: {410},
    KVReturnControlStatus.CAPACITY_REJECTED: {409},
    KVReturnControlStatus.INCOMPATIBLE: {409},
    KVReturnControlStatus.INVALID_TOKEN_IDS: {422},
    KVReturnControlStatus.CHECKSUM_MISMATCH: {422},
    KVReturnControlStatus.NOT_PREPARED: {409},
    KVReturnControlStatus.IMPORT_FAILED: {409},
    KVReturnControlStatus.NOT_FOUND: {404},
    KVReturnControlStatus.INVALID_REQUEST: {400, 413},
}

_KV_RETURN_FIXED_REASONS = {
    KVReturnControlStatus.READY: "reservation_ready",
    KVReturnControlStatus.COMMITTED: "prefix_committed",
    KVReturnControlStatus.ABORTED: "attempt_aborted",
    KVReturnControlStatus.DISABLED: "kv_return_disabled",
    KVReturnControlStatus.EXPIRED: "attempt_expired",
    KVReturnControlStatus.CAPACITY_REJECTED: "insufficient_clean_capacity",
    KVReturnControlStatus.INVALID_TOKEN_IDS: "invalid_generated_token_ids",
    KVReturnControlStatus.CHECKSUM_MISMATCH: "token_ids_checksum_mismatch",
    KVReturnControlStatus.NOT_PREPARED: "attempt_not_prepared",
    KVReturnControlStatus.IMPORT_FAILED: "atomic_prefix_import_failed",
    KVReturnControlStatus.NOT_FOUND: "ticket_not_found",
}

_KV_RETURN_INCOMPATIBLE_REASONS = {
    "incompatible_control_contract",
    "prefill_dp_rank_mismatch",
    "prefill_dp_rank_not_hosted",
    "unsupported_protocol_version",
}


def _parse_kv_return_response(
    operation: str,
    protocol_version: int,
    http_status: int,
    content_type: str,
    raw: bytes,
) -> KVReturnControlResponse:
    if content_type != "application/json":
        raise KVReturnProtocolError(
            f"KV return {operation} response must be application/json"
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError, OverflowError, RecursionError) as exc:
        raise KVReturnProtocolError(
            f"KV return {operation} response is not valid JSON"
        ) from exc
    expected_fields = {"status", "reason", "idempotent", "reserved_blocks"}
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise KVReturnProtocolError(
            f"KV return {operation} response has the wrong fields"
        )
    try:
        status = KVReturnControlStatus(payload["status"])
    except (TypeError, ValueError) as exc:
        raise KVReturnProtocolError(
            f"KV return {operation} response has an unknown status"
        ) from exc
    if http_status not in _KV_RETURN_HTTP_STATUSES[status]:
        raise KVReturnProtocolError(
            f"KV return {operation} response status disagrees with HTTP status"
        )
    reason = payload["reason"]
    if (
        not isinstance(reason, str)
        or not reason
        or len(reason) > KV_RETURN_MAX_METADATA_LENGTH
    ):
        raise KVReturnProtocolError(
            f"KV return {operation} response reason has the wrong type"
        )
    fixed_reason = _KV_RETURN_FIXED_REASONS.get(status)
    if fixed_reason is not None and reason != fixed_reason:
        raise KVReturnProtocolError(
            f"KV return {operation} response has an invalid reason"
        )
    if (
        status == KVReturnControlStatus.INCOMPATIBLE
        and reason not in _KV_RETURN_INCOMPATIBLE_REASONS
    ):
        raise KVReturnProtocolError(
            f"KV return {operation} response has an invalid reason"
        )
    idempotent = payload["idempotent"]
    reserved_blocks = payload["reserved_blocks"]
    if not isinstance(idempotent, bool):
        raise KVReturnProtocolError(
            f"KV return {operation} response idempotent has the wrong type"
        )
    if (
        isinstance(reserved_blocks, bool)
        or not isinstance(reserved_blocks, int)
        or not 0 <= reserved_blocks <= KV_RETURN_MAX_INTEGER
    ):
        raise KVReturnProtocolError(
            f"KV return {operation} response reserved_blocks has the wrong type"
        )
    if status != KVReturnControlStatus.READY and reserved_blocks != 0:
        raise KVReturnProtocolError(
            f"KV return {operation} rejection reserved unexpected blocks"
        )
    return KVReturnControlResponse(
        operation=operation,
        protocol_version=protocol_version,
        http_status=http_status,
        status=status,
        reason=reason,
        idempotent=idempotent,
        reserved_blocks=reserved_blocks,
    )


def _validate_peer(host: str, port: int) -> None:
    if not isinstance(host, str) or not host:
        raise KVReturnProtocolError("KV return host must be a nonempty string")
    if (
        isinstance(port, bool)
        or not isinstance(port, int)
        or not 1 <= port <= 65535
    ):
        raise KVReturnProtocolError("KV return port must be between 1 and 65535")


def _bracket(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host
