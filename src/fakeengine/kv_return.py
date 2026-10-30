"""Local state management for reverse KV-cache return attempts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
import math
import secrets
import time
from typing import Callable, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from fakeengine.scheduler import PrefixImportReservation, Scheduler


class KVReturnStatus(str, Enum):
    ISSUED = "issued"
    DISABLED = "disabled"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ABORTED = "aborted"
    EXPIRED = "expired"
    CAPACITY_REJECTED = "capacity_rejected"
    INCOMPATIBLE = "incompatible"
    INVALID_TOKEN_IDS = "invalid_token_ids"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    NOT_PREPARED = "not_prepared"
    IMPORT_FAILED = "import_failed"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class KVReturnCompatibility:
    protocol_version: int
    prefill_dp_rank: int
    served_model_name: str
    model_identity: str
    vocab_size: int
    block_size: int
    kv_bytes_per_token: int
    kv_layout_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return _fingerprint(asdict(self))


@dataclass(frozen=True)
class KVReturnTicket:
    nonce: str
    room_id: int
    attempt_id: str
    compatibility: KVReturnCompatibility
    cache_salt: Optional[str]
    max_generated_tokens: int
    max_payload_bytes: int
    deadline_ms: float


@dataclass(frozen=True)
class KVReturnPrepare:
    ticket: KVReturnTicket
    attempt_id: str
    prefill_dp_rank: int
    compatibility_fingerprint: str
    cache_salt: Optional[str]
    generated_token_count: int
    claimed_payload_bytes: int
    token_ids_checksum: str


@dataclass(frozen=True)
class KVReturnCommit:
    prepare: KVReturnPrepare
    generated_token_ids: Tuple[int, ...]


@dataclass(frozen=True)
class KVReturnAbort:
    ticket: KVReturnTicket
    attempt_id: str


@dataclass(frozen=True)
class KVReturnResult:
    status: KVReturnStatus
    ticket: Optional[KVReturnTicket] = None
    reserved_blocks: int = 0
    idempotent: bool = False


@dataclass
class _Attempt:
    ticket: KVReturnTicket
    prompt_ids: Tuple[int, ...]
    expires_at: float
    prepare: Optional[KVReturnPrepare] = None
    reservation: Optional["PrefixImportReservation"] = None


@dataclass(frozen=True)
class _Terminal:
    status: KVReturnStatus
    ticket: KVReturnTicket
    prepare_key: Optional[str]
    commit_key: Optional[str]
    expires_at: float


class KVReturnManager:
    """Own nonce-bound reverse-return attempts for one prefill DP rank."""

    DEFAULT_TERMINAL_TTL_S = 300.0
    DEFAULT_MAX_TERMINAL_RECORDS = 1024
    HARD_MAX_TERMINAL_RECORDS = 4096

    def __init__(
        self,
        scheduler: "Scheduler",
        dp_rank: int,
        *,
        kv_layout_fingerprint: Optional[str] = None,
        terminal_ttl_s: float = DEFAULT_TERMINAL_TTL_S,
        max_terminal_records: int = DEFAULT_MAX_TERMINAL_RECORDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(dp_rank, bool) or not isinstance(dp_rank, int) or dp_rank < 0:
            raise ValueError("dp_rank must be a nonnegative integer")
        if not math.isfinite(terminal_ttl_s) or terminal_ttl_s <= 0:
            raise ValueError("terminal_ttl_s must be finite and positive")
        if (
            isinstance(max_terminal_records, bool)
            or not isinstance(max_terminal_records, int)
            or not 1 <= max_terminal_records <= self.HARD_MAX_TERMINAL_RECORDS
        ):
            raise ValueError("max_terminal_records is outside the hard limit")

        self.scheduler = scheduler
        self.config = scheduler.config
        self.dp_rank = dp_rank
        self.terminal_ttl_s = terminal_ttl_s
        self.max_terminal_records = max_terminal_records
        self._clock = clock
        self.compatibility = KVReturnCompatibility(
            protocol_version=self.config.kv_return_protocol_version,
            prefill_dp_rank=dp_rank,
            served_model_name=self.config.served_model_name,
            model_identity=self.config.model_path,
            vocab_size=self.config.vocab_size,
            block_size=self.config.block_size,
            kv_bytes_per_token=self.config.kv_return_bytes_per_token,
            kv_layout_fingerprint=kv_layout_fingerprint or _fingerprint({
                "layout": "fake-sglang-kv-v1",
                "block_size": self.config.block_size,
                "kv_bytes_per_token": self.config.kv_return_bytes_per_token,
            }),
        )
        self._attempts: Dict[str, _Attempt] = {}
        self._terminal: "OrderedDict[str, _Terminal]" = OrderedDict()

    def issue_ticket(
        self,
        room_id: int,
        prompt_ids: Sequence[int],
        cache_salt: Optional[str] = None,
        *,
        attempt_id: Optional[str] = None,
    ) -> KVReturnResult:
        """Issue a local ticket for one eligible prefill room."""
        self.expire()
        if not self._enabled():
            return KVReturnResult(KVReturnStatus.DISABLED)
        if isinstance(room_id, bool) or not isinstance(room_id, int) or room_id < 0:
            return KVReturnResult(KVReturnStatus.INCOMPATIBLE)
        prompt = tuple(prompt_ids)
        if len(prompt) > self.config.context_len or not self._valid_tokens(prompt):
            return KVReturnResult(KVReturnStatus.INCOMPATIBLE)
        if cache_salt is not None and not isinstance(cache_salt, str):
            return KVReturnResult(KVReturnStatus.INCOMPATIBLE)
        salt = cache_salt or None
        attempt = secrets.token_urlsafe(18) if attempt_id is None else attempt_id
        if not isinstance(attempt, str) or not attempt or len(attempt) > 256:
            return KVReturnResult(KVReturnStatus.INCOMPATIBLE)

        nonce = self._new_nonce()
        ticket = KVReturnTicket(
            nonce=nonce,
            room_id=room_id,
            attempt_id=attempt,
            compatibility=self.compatibility,
            cache_salt=salt,
            max_generated_tokens=self.config.kv_return_max_tokens,
            max_payload_bytes=(
                self.config.kv_return_max_tokens
                * self.compatibility.kv_bytes_per_token
            ),
            deadline_ms=self.config.kv_return_deadline_ms,
        )
        expires_at = self._clock() + ticket.deadline_ms / 1000.0
        self._attempts[nonce] = _Attempt(ticket, prompt, expires_at)
        return KVReturnResult(
            KVReturnStatus.ISSUED,
            ticket=ticket,
        )

    def prepare(self, prepare: KVReturnPrepare) -> KVReturnResult:
        """Validate controls and reserve capacity before any transfer."""
        now = self._clock()
        self.expire(now)
        nonce = prepare.ticket.nonce
        terminal = self._terminal.get(nonce)
        if terminal is not None:
            return self._terminal_result(terminal, prepare=prepare)
        attempt = self._attempts.get(nonce)
        if attempt is None:
            return KVReturnResult(KVReturnStatus.NOT_FOUND)
        if attempt.prepare is not None:
            if attempt.prepare == prepare:
                reservation = attempt.reservation
                assert reservation is not None
                return KVReturnResult(
                    KVReturnStatus.PREPARED,
                    reserved_blocks=reservation.reserved_blocks,
                    idempotent=True,
                )
            return self._finish(
                attempt,
                KVReturnStatus.INCOMPATIBLE,
            )
        if not self._valid_prepare(attempt, prepare, now):
            return self._finish(
                attempt, KVReturnStatus.INCOMPATIBLE,
                prepare=prepare,
            )

        prompt_blocks = len(attempt.prompt_ids) // self.config.block_size
        total_blocks = (
            len(attempt.prompt_ids) + prepare.generated_token_count
        ) // self.config.block_size
        if total_blocks - prompt_blocks > self.scheduler.pool.free_blocks:
            return self._finish(
                attempt,
                KVReturnStatus.CAPACITY_REJECTED,
                prepare=prepare,
            )
        reservation = self.scheduler.reserve_prefix_import(
            attempt.prompt_ids,
            prepare.generated_token_count,
            prepare.cache_salt,
        )
        if reservation is None:
            return self._finish(
                attempt,
                KVReturnStatus.INCOMPATIBLE,
                prepare=prepare,
            )
        attempt.prepare = prepare
        attempt.reservation = reservation
        return KVReturnResult(
            KVReturnStatus.PREPARED,
            reserved_blocks=reservation.reserved_blocks,
        )

    def commit(self, commit: KVReturnCommit) -> KVReturnResult:
        """Validate a prepared attempt and atomically import its token tail."""
        prepare = commit.prepare
        nonce = prepare.ticket.nonce
        now = self._clock()
        self.expire(now)
        terminal = self._terminal.get(nonce)
        if terminal is not None:
            return self._terminal_result(terminal, prepare=prepare, commit=commit)
        attempt = self._attempts.get(nonce)
        if attempt is None:
            return KVReturnResult(KVReturnStatus.NOT_FOUND)
        if attempt.prepare is None:
            return self._finish(
                attempt,
                (
                    KVReturnStatus.NOT_PREPARED
                    if self._valid_prepare(attempt, prepare, now)
                    else KVReturnStatus.INCOMPATIBLE
                ),
                prepare=prepare,
                commit=commit,
            )
        if prepare != attempt.prepare:
            return self._finish(
                attempt,
                KVReturnStatus.INCOMPATIBLE,
                commit=commit,
            )
        actual_token_count = len(commit.generated_token_ids)
        actual_payload_bytes = (
            actual_token_count * self.compatibility.kv_bytes_per_token
        )
        if (
            actual_token_count != prepare.generated_token_count
            or actual_payload_bytes != prepare.claimed_payload_bytes
        ):
            return self._finish(
                attempt,
                KVReturnStatus.INCOMPATIBLE,
                commit=commit,
            )
        if not self._valid_tokens(commit.generated_token_ids):
            return self._finish(
                attempt,
                KVReturnStatus.INVALID_TOKEN_IDS,
                commit=commit,
            )
        if prepare.token_ids_checksum != kv_return_checksum(
            prepare.ticket, commit.generated_token_ids,
        ):
            return self._finish(
                attempt,
                KVReturnStatus.CHECKSUM_MISMATCH,
                commit=commit,
            )

        reservation = attempt.reservation
        assert reservation is not None
        imported = self.scheduler.commit_prefix_import(
            reservation, commit.generated_token_ids,
        )
        return self._finish(
            attempt,
            KVReturnStatus.COMMITTED if imported else KVReturnStatus.IMPORT_FAILED,
            commit=commit,
            release=False,
        )

    def abort(self, abort: KVReturnAbort) -> KVReturnResult:
        """Abort an issued or prepared attempt and release its reservation."""
        self.expire()
        nonce = abort.ticket.nonce
        terminal = self._terminal.get(nonce)
        if terminal is not None:
            return self._terminal_result(terminal, abort=abort)
        attempt = self._attempts.get(nonce)
        if attempt is None:
            return KVReturnResult(KVReturnStatus.NOT_FOUND)
        if abort.ticket != attempt.ticket or abort.attempt_id != attempt.ticket.attempt_id:
            return self._finish(
                attempt,
                KVReturnStatus.INCOMPATIBLE,
            )
        return self._finish(attempt, KVReturnStatus.ABORTED)

    def expire(self, now: Optional[float] = None) -> int:
        """Expire active attempts and purge bounded idempotency records."""
        current = self._clock() if now is None else now
        self._purge_terminals(current)
        expired = [
            attempt for attempt in self._attempts.values()
            if current >= attempt.expires_at
        ]
        for attempt in sorted(expired, key=lambda item: (item.expires_at, item.ticket.nonce)):
            self._finish(
                attempt,
                KVReturnStatus.EXPIRED,
                now=current,
            )
        return len(expired)

    def abort_all(self) -> None:
        """Release all reservations and clear active and terminal state."""
        for attempt in sorted(
            self._attempts.values(), key=lambda item: item.ticket.nonce,
        ):
            if attempt.reservation is not None:
                self.scheduler.release_prefix_import(attempt.reservation)
        self._attempts.clear()
        self._terminal.clear()

    def _valid_prepare(
        self,
        attempt: _Attempt,
        prepare: KVReturnPrepare,
        now: float,
    ) -> bool:
        ticket = prepare.ticket
        count = prepare.generated_token_count
        valid_count = (
            isinstance(count, int)
            and not isinstance(count, bool)
            and 0 <= count <= ticket.max_generated_tokens
        )
        expected_bytes = (
            count * self.compatibility.kv_bytes_per_token if valid_count else -1
        )
        return (
            ticket.compatibility == self.compatibility
            and ticket == attempt.ticket
            and prepare.attempt_id == ticket.attempt_id
            and prepare.prefill_dp_rank == self.dp_rank
            and prepare.compatibility_fingerprint == self.compatibility.fingerprint
            and prepare.cache_salt == ticket.cache_salt
            and valid_count
            and len(attempt.prompt_ids) + count <= self.config.context_len
            and prepare.claimed_payload_bytes == expected_bytes
            and expected_bytes <= ticket.max_payload_bytes
            and _valid_checksum(prepare.token_ids_checksum)
            and now < attempt.expires_at
        )

    def _finish(
        self,
        attempt: _Attempt,
        status: KVReturnStatus,
        *,
        prepare: Optional[KVReturnPrepare] = None,
        commit: Optional[KVReturnCommit] = None,
        release: bool = True,
        now: Optional[float] = None,
    ) -> KVReturnResult:
        self._attempts.pop(attempt.ticket.nonce, None)
        if release and attempt.reservation is not None:
            self.scheduler.release_prefix_import(attempt.reservation)
        prepare = prepare or attempt.prepare
        current = self._clock() if now is None else now
        terminal_expires_at = current + self.terminal_ttl_s
        self._terminal[attempt.ticket.nonce] = _Terminal(
            status=status,
            ticket=attempt.ticket,
            prepare_key=_fingerprint(prepare) if prepare is not None else None,
            commit_key=_fingerprint(commit) if commit is not None else None,
            expires_at=terminal_expires_at,
        )
        self._terminal.move_to_end(attempt.ticket.nonce)
        while len(self._terminal) > self.max_terminal_records:
            self._terminal.popitem(last=False)
        return KVReturnResult(status)

    @staticmethod
    def _terminal_result(
        terminal: _Terminal,
        *,
        prepare: Optional[KVReturnPrepare] = None,
        commit: Optional[KVReturnCommit] = None,
        abort: Optional[KVReturnAbort] = None,
    ) -> KVReturnResult:
        ticket_matches = (
            prepare is not None
            and prepare.ticket == terminal.ticket
            and prepare.attempt_id == terminal.ticket.attempt_id
        )
        if commit is not None:
            if terminal.commit_key is not None:
                compatible = _fingerprint(commit) == terminal.commit_key
            elif terminal.prepare_key is not None:
                compatible = _fingerprint(prepare) == terminal.prepare_key
            else:
                compatible = ticket_matches
        elif prepare is not None:
            compatible = (
                _fingerprint(prepare) == terminal.prepare_key
                if terminal.prepare_key is not None
                else ticket_matches
            )
        elif abort is not None:
            compatible = (
                abort.ticket == terminal.ticket
                and abort.attempt_id == terminal.ticket.attempt_id
            )
        else:
            compatible = False
        if compatible:
            return KVReturnResult(terminal.status, idempotent=True)
        return KVReturnResult(
            KVReturnStatus.INCOMPATIBLE,
            idempotent=True,
        )

    def _enabled(self) -> bool:
        return (
            self.config.enable_kv_return
            and self.config.disaggregation_mode == "prefill"
            and self.config.prefix_caching
        )

    def _purge_terminals(self, now: float) -> int:
        expired = [
            nonce for nonce, terminal in self._terminal.items()
            if now >= terminal.expires_at
        ]
        for nonce in expired:
            self._terminal.pop(nonce, None)
        return len(expired)

    def _new_nonce(self) -> str:
        while True:
            nonce = secrets.token_urlsafe(24)
            if nonce not in self._attempts and nonce not in self._terminal:
                return nonce

    def _valid_tokens(self, token_ids: Sequence[int]) -> bool:
        return all(
            isinstance(token, int)
            and not isinstance(token, bool)
            and 0 <= token < self.compatibility.vocab_size
            for token in token_ids
        )


def kv_return_checksum(
    ticket: KVReturnTicket, generated_token_ids: Sequence[int],
) -> str:
    """Checksum token IDs together with their immutable control contract."""
    return _fingerprint({
        "nonce": ticket.nonce,
        "attempt_id": ticket.attempt_id,
        "prefill_dp_rank": ticket.compatibility.prefill_dp_rank,
        "compatibility_fingerprint": ticket.compatibility.fingerprint,
        "cache_salt": ticket.cache_salt,
        "generated_token_count": len(generated_token_ids),
        "claimed_payload_bytes": (
            len(generated_token_ids) * ticket.compatibility.kv_bytes_per_token
        ),
        "generated_token_ids": tuple(generated_token_ids),
    })


def _valid_checksum(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )


def _fingerprint(value: object) -> str:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()
