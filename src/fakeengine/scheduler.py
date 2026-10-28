"""Continuous-batching scheduler.

Models the parts of an LLM scheduler that change observable behavior:

* chunked prefill against a token budget, so a long prompt does not stall the
  whole batch for one step;
* a shared decode step whose cost grows with batch size, so queueing shows up
  as ITL growth the way it does on a GPU;
* a finite KV pool with LIFO preemption, so memory pressure is reachable;
* prefix reuse, so a repeated prompt reports non-zero ``cached_tokens``.
* scheduler-wide output batches separated into coalesced request streams.

"Prebuilt" requests are the decode side of disaggregation: their KV already
arrived over the (simulated) transfer, so they skip prefill compute but still
occupy blocks. SGLang calls the same thing ``prepare_for_prebuilt`` /
``process_prebuilt`` in ``disaggregation/decode_schedule_batch_mixin.py``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import math
import time
import uuid
from statistics import pvariance
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.observability.forward_pass_metrics import (
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)

from fakeengine import tokens as token_gen
from fakeengine.cache_metrics import PrefixCacheMetrics
from fakeengine.config import SimConfig
from fakeengine.fpm import ForwardPassPublisher
from fakeengine.kv_events import NullKvEventEmitter
from fakeengine.kv_return import KVReturnManager
from fakeengine.kvcache import BlockPool, block_hashes
from fakeengine.timing import TimingModel

logger = logging.getLogger(__name__)

FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_ABORT = "abort"


@dataclass
class PrefixImportReservation:
    prompt_ids: Tuple[int, ...]
    max_generated_tokens: int
    cache_salt: Optional[str]
    _owner: object
    _prefix_blocks: List[int]
    _reserved_blocks: List[int]
    _active: bool = True

    @property
    def reserved_blocks(self) -> int:
        return len(self._reserved_blocks)


@dataclass
class _KVRetentionState:
    released: bool = False
    released_at: Optional[float] = None
    release_reason: Optional[str] = None


@dataclass(frozen=True)
class KVRetentionLease:
    lease_id: str
    request_id: str
    prompt_token_count: int
    generated_token_count: int
    generated_token_ids: Tuple[int, ...] = field(repr=False)
    block_ids: Tuple[int, ...]
    block_count: int
    retained_bytes: int
    cache_salt: Optional[str]
    created_at: float
    deadline: float
    _owner: object = field(repr=False, compare=False)
    _state: _KVRetentionState = field(
        default_factory=_KVRetentionState,
        repr=False,
        compare=False,
    )

    @property
    def released(self) -> bool:
        return self._state.released

    @property
    def released_at(self) -> Optional[float]:
        return self._state.released_at

    @property
    def release_reason(self) -> Optional[str]:
        return self._state.release_reason


class Request:
    __slots__ = (
        "_cache_observed", "_hashes", "_output_event", "_output_ready",
        "_output_reason", "_output_started", "_output_tokens", "_stream",
        "_original_prompt_token_count", "_generated_token_ids",
        "arrival", "blocks", "cache_salt", "cached_tokens", "finished",
        "first_token", "generated", "max_new_tokens", "observe_cache", "prebuilt",
        "prefilled", "prompt_ids", "retain_response_kv", "retention_lease",
        "retention_rejection_reason", "rid",
    )

    def __init__(
        self,
        rid: str,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        prebuilt: bool = False,
        first_token: Optional[int] = None,
        cache_salt: Optional[str] = None,
        observe_cache: bool = True,
        retain_response_kv: bool = False,
    ) -> None:
        if not isinstance(retain_response_kv, bool):
            raise TypeError("retain_response_kv must be a boolean")
        self.rid = rid
        # Prefix-cache namespace: only requests with the same salt share KV.
        self.cache_salt = cache_salt or None
        self.prompt_ids = list(prompt_ids)
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.prebuilt = prebuilt
        self.first_token = first_token
        self.blocks: List[int] = []
        self.prefilled = 0
        self.generated: List[int] = []
        self.cached_tokens = 0
        self._hashes: Optional[List[bytes]] = None
        self._stream = None
        self.finished = False
        self.arrival = time.monotonic()
        self._cache_observed = False
        self.observe_cache = observe_cache
        self._output_tokens: list[int] = []
        self._output_ready = 0
        self._output_reason: str | None = None
        self._output_started = False
        self._output_event = asyncio.Event()
        self.retain_response_kv = retain_response_kv
        self.retention_lease: Optional[KVRetentionLease] = None
        self.retention_rejection_reason: Optional[str] = None
        self._original_prompt_token_count = len(self.prompt_ids)
        self._generated_token_ids: List[int] = []

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.generated)

    @property
    def pending_output_tokens(self) -> int:
        return len(self._output_tokens)

    @property
    def output_notification_pending(self) -> bool:
        return self._output_event.is_set()

    @property
    def output_finished(self) -> bool:
        return self._output_reason is not None

    def publish_output(
        self,
        token_ids: Sequence[int],
        finish_reason: str | None,
        stream_interval: int,
    ) -> None:
        """Add separated output without allocating one queue item per token."""
        self._output_tokens.extend(token_ids)
        unsent = len(self._output_tokens) - self._output_ready
        if finish_reason is not None:
            self._output_ready = len(self._output_tokens)
            self._output_reason = finish_reason
        elif not self._output_started and unsent > 0:
            self._output_ready = len(self._output_tokens)
            self._output_started = True
        elif unsent >= stream_interval:
            self._output_ready = len(self._output_tokens)
        else:
            return
        self._output_event.set()

    async def next_output(self) -> tuple[list[int], str | None]:
        """Wait for the next stream-interval boundary or terminal flush."""
        while True:
            if self._output_ready:
                token_ids = self._output_tokens[:self._output_ready]
                del self._output_tokens[:self._output_ready]
                self._output_ready = 0
                finish_reason = None
                if self._output_reason is not None and not self._output_tokens:
                    finish_reason = self._output_reason
                    self._output_reason = None
                self._output_event.clear()
                return token_ids, finish_reason
            if self._output_reason is not None:
                finish_reason = self._output_reason
                self._output_reason = None
                self._output_event.clear()
                return [], finish_reason
            self._output_event.clear()
            await self._output_event.wait()


@dataclass(frozen=True)
class RequestOutput:
    request: Request
    token_ids: tuple[int, ...] = ()
    finish_reason: str | None = None


@dataclass(frozen=True)
class SchedulerOutputBatch:
    outputs: tuple[RequestOutput, ...]


class Scheduler:
    def __init__(
        self, config: SimConfig, kv_events=None,
        fpm: Optional[ForwardPassPublisher] = None,
        cache_metrics: Optional[PrefixCacheMetrics] = None,
        *,
        dp_rank: int = 0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.config = config
        self.dp_rank = dp_rank
        self.timing = TimingModel(config)
        # The emitter is wired into the pool itself so an eviction cannot
        # happen without the router hearing about it, whichever code path
        # triggered it.
        self.kv_events = kv_events or NullKvEventEmitter()
        self.cache_metrics = cache_metrics
        self.pool = BlockPool(
            config.num_kv_blocks,
            config.block_size,
            on_evict=self.kv_events.publish_removed,
            on_change=self._publish_cache_state,
        )
        self._publish_cache_state()
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.by_rid: Dict[str, Request] = {}
        self._wakeup = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._output_queue: asyncio.Queue[SchedulerOutputBatch] = asyncio.Queue()
        self._output_task: asyncio.Task | None = None
        self._stopping = False
        self.last_step_ms = 0.0
        self.fpm = fpm
        self.prefix_cache_query_tokens = 0
        self.prefix_cache_hit_tokens = 0
        self.prefix_cache_queries = 0
        self.kv_generation_seconds = 0.0
        self._prefix_import_owner = object()
        self.kv_return = KVReturnManager(self, dp_rank)
        self._retention_owner = object()
        self._retention_leases: Dict[str, KVRetentionLease] = {}
        self._retention_by_request: Dict[str, str] = {}
        self._retained_blocks = 0
        self._retained_bytes = 0
        self._retention_rejections = 0
        self._clock = clock or time.monotonic

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            loop = asyncio.get_running_loop()
            self._output_task = loop.create_task(self._run_output_processor())
            self._task = loop.create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        self._wakeup.set()
        try:
            if self._task is not None:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        finally:
            self._task = None
            self.kv_return.abort_all()
            self.abort_all()
            if self._output_task is not None:
                await self._output_queue.join()
                self._output_task.cancel()
                try:
                    await self._output_task
                except asyncio.CancelledError:
                    pass
                self._output_task = None
            if self.fpm is not None:
                self.fpm.close()
                self.fpm = None

    # -- submission --------------------------------------------------------

    def submit(self, request: Request) -> Request:
        limit = min(self.config.total_kv_tokens, self.config.context_len)
        if len(request.prompt_ids) > limit:
            raise ValueError(
                f"Prompt has {len(request.prompt_ids)} tokens; worker capacity is {limit}"
            )
        self.by_rid[request.rid] = request
        self.waiting.append(request)
        self._wakeup.set()
        return request

    def abort(self, rid: str) -> bool:
        request = self.by_rid.get(rid)
        if request is None or request.finished:
            return False
        self._finish(request, FINISH_ABORT)
        return True

    def abort_all(self) -> None:
        for rid in list(self.by_rid):
            self.abort(rid)
        for lease in list(self._retention_leases.values()):
            self.release_retention(lease, reason="abort_all")

    def flush_cache(self) -> None:
        """Drop cached prefixes. Only unreferenced blocks can go, so this is a
        best-effort hint exactly as it is in a real engine."""
        self.pool.clear_cache()

    # -- completed decode retention ---------------------------------------

    @property
    def active_retention_leases(self) -> Tuple[KVRetentionLease, ...]:
        self.expire_retention_leases()
        return tuple(self._retention_leases.values())

    def get_retention_lease(
        self, request_id: str,
    ) -> Optional[KVRetentionLease]:
        self.expire_retention_leases()
        lease_id = self._retention_by_request.get(request_id)
        if lease_id is None:
            return None
        return self._retention_leases.get(lease_id)

    def release_retention(
        self,
        lease: KVRetentionLease,
        *,
        reason: str = "released",
        now: Optional[float] = None,
    ) -> bool:
        if (
            not isinstance(lease, KVRetentionLease)
            or lease._owner is not self._retention_owner
        ):
            return False
        if lease.released:
            return True

        active = self._retention_leases.get(lease.lease_id)
        if active is not lease:
            return False

        lease._state.released = True                        # noqa: SLF001
        lease._state.released_at = (                        # noqa: SLF001
            self._clock() if now is None else now
        )
        lease._state.release_reason = reason                 # noqa: SLF001
        self._retention_leases.pop(lease.lease_id, None)
        if self._retention_by_request.get(lease.request_id) == lease.lease_id:
            self._retention_by_request.pop(lease.request_id, None)
        self._retained_blocks -= lease.block_count
        self._retained_bytes -= lease.retained_bytes
        self.pool.free(lease.block_ids)
        return True

    def expire_retention_leases(self, now: Optional[float] = None) -> int:
        current = self._clock() if now is None else now
        expired = [
            lease
            for lease in self._retention_leases.values()
            if lease.deadline <= current
        ]
        for lease in sorted(expired, key=lambda item: (item.deadline, item.lease_id)):
            self.release_retention(lease, reason="expired", now=current)
        return len(expired)

    # -- local prefix import ----------------------------------------------

    def reserve_prefix_import(
        self,
        prompt_ids: Sequence[int],
        max_generated_tokens: int,
        cache_salt: Optional[str] = None,
    ) -> Optional[PrefixImportReservation]:
        """Hold clean blocks for a future prompt-plus-response prefix."""
        self.expire_retention_leases()
        prompt = tuple(prompt_ids)
        if (
            not self.config.prefix_caching
            or isinstance(max_generated_tokens, bool)
            or not isinstance(max_generated_tokens, int)
            or max_generated_tokens < 0
            or len(prompt) + max_generated_tokens > self.config.context_len
            or not self._valid_token_ids(prompt)
            or cache_salt is not None and not isinstance(cache_salt, str)
        ):
            return None

        salt = cache_salt or None
        prompt_hashes = block_hashes(prompt, self.config.block_size, salt)
        prefix_blocks = self.pool.acquire_prefix(prompt_hashes, touch=False)
        if len(prefix_blocks) != len(prompt_hashes):
            self.pool.free(prefix_blocks)
            return None

        target_blocks = (
            len(prompt) + max_generated_tokens
        ) // self.config.block_size
        reserve_count = target_blocks - len(prompt_hashes)
        reserved_blocks = self.pool.allocate_clean(reserve_count)
        if reserved_blocks is None:
            self.pool.free(prefix_blocks)
            return None

        return PrefixImportReservation(
            prompt,
            max_generated_tokens,
            salt,
            self._prefix_import_owner,
            prefix_blocks,
            reserved_blocks,
        )

    def commit_prefix_import(
        self,
        reservation: PrefixImportReservation,
        generated_token_ids: Sequence[int],
    ) -> bool:
        """Atomically install all full blocks covered by a reservation."""
        generated = tuple(generated_token_ids)
        if (
            not self._owns_prefix_import(reservation)
            or len(generated) > reservation.max_generated_tokens
            or not self._valid_token_ids(generated)
        ):
            if self._owns_prefix_import(reservation):
                self.release_prefix_import(reservation)
            return False

        token_ids = reservation.prompt_ids + generated
        hashes = block_hashes(
            token_ids, self.config.block_size, reservation.cache_salt,
        )
        needed = len(hashes) - len(reservation._prefix_blocks)
        if needed < 0 or needed > len(reservation._reserved_blocks):
            self.release_prefix_import(reservation)
            return False

        blocks = (
            reservation._prefix_blocks
            + reservation._reserved_blocks[:needed]
        )
        added = self.pool.insert_prefix_atomic(hashes, blocks)
        if added is None:
            self.release_prefix_import(reservation)
            return False
        if added:
            self.kv_events.publish_stored(
                token_ids=token_ids,
                digests=hashes,
                new_indices=[index for index, _ in added],
                cache_salt=reservation.cache_salt,
            )
        self.release_prefix_import(reservation)
        return True

    def release_prefix_import(
        self, reservation: PrefixImportReservation,
    ) -> bool:
        if not self._owns_prefix_import(reservation):
            return False
        reservation._active = False
        self.pool.free(
            reservation._prefix_blocks + reservation._reserved_blocks
        )
        return True

    def _owns_prefix_import(
        self, reservation: PrefixImportReservation,
    ) -> bool:
        return (
            isinstance(reservation, PrefixImportReservation)
            and reservation._owner is self._prefix_import_owner
            and reservation._active
        )

    @staticmethod
    def _valid_token_ids(token_ids: Sequence[int]) -> bool:
        return all(
            isinstance(token, int) and not isinstance(token, bool) and token >= 0
            for token in token_ids
        )

    # -- stats -------------------------------------------------------------

    def _publish_cache_state(self) -> None:
        if self.cache_metrics is not None:
            self.cache_metrics.update_pool(self.pool)

    def stats(self) -> dict:
        self.expire_retention_leases()
        return {
            "num_running_reqs": len(self.running),
            "num_waiting_reqs": len(self.waiting),
            "num_used_tokens": self.pool.used_blocks * self.config.block_size,
            "token_usage": self.pool.usage,
            "kv_used_blocks": self.pool.used_blocks,
            "kv_total_blocks": self.pool.num_blocks,
            "kv_cached_blocks": self.pool.cached_blocks,
            "kv_evicted_blocks": self.pool.evicted_blocks,
            "kv_cleared_blocks": self.pool.cleared_blocks,
            "last_step_ms": self.last_step_ms,
            "prefix_cache_query_tokens": self.prefix_cache_query_tokens,
            "prefix_cache_hit_tokens": self.prefix_cache_hit_tokens,
            "prefix_cache_queries": self.prefix_cache_queries,
            "kv_generation_seconds": self.kv_generation_seconds,
            "kv_retention_active_leases": len(self._retention_leases),
            "kv_retention_blocks": self._retained_blocks,
            "kv_retention_bytes": self._retained_bytes,
            "kv_retention_rejections": self._retention_rejections,
        }

    # -- the loop ----------------------------------------------------------

    def _queued_metrics(self, scheduled_ids=frozenset()) -> QueuedRequestMetrics:
        waiting = [r for r in self.waiting if r.rid not in scheduled_ids]
        prefill = [
            max(len(r.prompt_ids) - max(r.prefilled, r.cached_tokens), 0)
            for r in waiting if not r.prebuilt
        ]
        decode = [r.total_tokens for r in waiting if r.prebuilt]
        decode += [r.total_tokens for r in self.running if r.rid not in scheduled_ids]
        return QueuedRequestMetrics(
            num_prefill_requests=len(prefill),
            sum_prefill_tokens=sum(prefill),
            var_prefill_length=float(pvariance(prefill)) if prefill else 0.0,
            num_decode_requests=len(decode),
            sum_decode_kv_tokens=sum(decode),
            var_decode_kv_tokens=float(pvariance(decode)) if decode else 0.0,
        )

    def _publish_idle_fpm(self) -> None:
        if self.fpm is not None and time.monotonic() - self.fpm.last_publish >= 1.0:
            self.fpm.publish(ScheduledRequestMetrics(), self._queued_metrics(), 0.0)

    async def _run(self) -> None:
        while not self._stopping:
            self.expire_retention_leases()
            if not self.waiting and not self.running:
                self._wakeup.clear()
                timeout = self._idle_wait_timeout()
                if timeout is None:
                    await self._wakeup.wait()
                else:
                    self._publish_idle_fpm()
                    try:
                        await asyncio.wait_for(self._wakeup.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        pass
                continue
            if await self._prefill_step():
                continue
            if self.running:
                await self._decode_step()
                continue
            # Waiting requests that no step can admit: the KV pool is full of
            # referenced blocks. Yield rather than spin; a completion frees them.
            self._publish_idle_fpm()
            await asyncio.sleep(0.001)

    async def _run_output_processor(self) -> None:
        while True:
            batch = await self._output_queue.get()
            try:
                self._separate_outputs(batch)
            finally:
                self._output_queue.task_done()

    def _publish_output_batch(self, outputs: Sequence[RequestOutput]) -> None:
        if not outputs:
            return
        batch = SchedulerOutputBatch(tuple(outputs))
        if self._output_task is None or self._output_task.done():
            self._separate_outputs(batch)
        else:
            self._output_queue.put_nowait(batch)

    def _separate_outputs(self, batch: SchedulerOutputBatch) -> None:
        for output in batch.outputs:
            output.request.publish_output(
                output.token_ids,
                output.finish_reason,
                self.config.stream_interval,
            )

    def _idle_wait_timeout(self) -> Optional[float]:
        timeouts: List[float] = []
        if self.fpm is not None:
            timeouts.append(1.0)
        if self._retention_leases:
            next_deadline = min(
                lease.deadline for lease in self._retention_leases.values()
            )
            timeouts.append(max(next_deadline - self._clock(), 0.0))
        return min(timeouts) if timeouts else None

    async def _prefill_step(self) -> bool:
        """Advance prefill for as many waiting requests as the budget allows.

        Returns True when this step was spent on prefill, so the caller skips
        decode -- a real scheduler cannot run both in one forward pass.
        """
        if not self.waiting:
            return False

        budget = self.config.max_batched_tokens
        admitted: List[Request] = []
        computed_tokens = 0     # tokens actually recomputed (billed for time)
        observed_computed_tokens = 0
        moved_tokens = 0        # tokens advanced, cached or not (progress check)
        scheduled_ids = set()
        prefill_lengths = []
        prefill_kv_tokens = 0

        while self.waiting and budget > 0:
            if len(self.running) + len(admitted) >= self.config.max_running_requests:
                break
            request = self.waiting[0]

            if request.prefilled == 0 and not self._reserve_prefix(request):
                break                       # out of KV; try again after a decode frees some

            remaining = len(request.prompt_ids) - request.prefilled
            chunk = min(remaining, budget)
            if chunk <= 0:
                break
            if not self._grow_blocks(request, request.prefilled + chunk):
                break

            if request.observe_cache and not request.prebuilt and not request._cache_observed:
                prompt_tokens = len(request.prompt_ids)
                self.prefix_cache_query_tokens += prompt_tokens
                self.prefix_cache_hit_tokens += request.cached_tokens
                self.prefix_cache_queries += 1
                if self.cache_metrics is not None:
                    self.cache_metrics.observe(prompt_tokens, request.cached_tokens)
                request._cache_observed = True

            # Cached prefix blocks and KV that arrived over the transfer are
            # occupancy, not compute: they must not be billed as prefill time.
            free_tokens = max(request.cached_tokens - request.prefilled, 0)
            computed = max(chunk - free_tokens, 0)
            computed_tokens += computed
            if request.observe_cache:
                observed_computed_tokens += computed
            scheduled_ids.add(request.rid)
            if computed:
                prefill_lengths.append(computed)
                prefill_kv_tokens += max(request.prefilled, request.cached_tokens)
            request.prefilled += chunk
            moved_tokens += chunk
            budget -= chunk

            if request.prefilled >= len(request.prompt_ids):
                self.waiting.pop(0)
                admitted.append(request)
            else:
                break                       # chunked: finish this prompt next step

        if moved_tokens == 0 and not admitted:
            return False

        kv_seconds = self.timing.kv_generation_ms(observed_computed_tokens) / (
            1000.0 * self.config.speedup_ratio
        )
        self.kv_generation_seconds += kv_seconds
        if self.cache_metrics is not None:
            self.cache_metrics.record_kv_generation(kv_seconds)
        await self._sleep_step(
            self.timing.prefill_ms(computed_tokens),
            ScheduledRequestMetrics(
                num_prefill_requests=len(prefill_lengths),
                sum_prefill_tokens=computed_tokens,
                var_prefill_length=float(pvariance(prefill_lengths)) if prefill_lengths else 0.0,
                sum_prefill_kv_tokens=prefill_kv_tokens,
            ),
            scheduled_ids,
        )

        outputs: list[RequestOutput] = []
        for request in admitted:
            if request.finished:
                continue
            self._cache_prefix(request)
            self.running.append(request)
            outputs.append(self._emit_first_token(request))
        self._publish_output_batch(outputs)
        return True

    async def _decode_step(self) -> None:
        if not self.running:
            return
        decode_lengths = [r.total_tokens for r in self.running]
        await self._sleep_step(
            self.timing.decode_step_ms(len(self.running)),
            ScheduledRequestMetrics(
                num_decode_requests=len(decode_lengths),
                sum_decode_kv_tokens=sum(decode_lengths),
                var_decode_kv_tokens=float(pvariance(decode_lengths)),
            ),
            {r.rid for r in self.running},
        )

        outputs: list[RequestOutput] = []
        for request in list(self.running):
            if request.finished:
                continue
            if not self._grow_blocks(request, request.total_tokens + 1):
                if not self._preempt_one(protect=request):
                    self._complete(request, FINISH_ABORT)
                    outputs.append(RequestOutput(request, finish_reason=FINISH_ABORT))
                    continue
                if not self._grow_blocks(request, request.total_tokens + 1):
                    self._complete(request, FINISH_ABORT)
                    outputs.append(RequestOutput(request, finish_reason=FINISH_ABORT))
                    continue

            token = self._next_token(request)
            request.generated.append(token)
            if request.retain_response_kv:
                request._generated_token_ids.append(token)      # noqa: SLF001

            finish_reason = None
            if (
                len(request.generated) >= request.max_new_tokens
                or request.total_tokens >= self.config.context_len
            ):
                finish_reason = FINISH_LENGTH
            if finish_reason is not None:
                self._complete(request, finish_reason)
            outputs.append(RequestOutput(request, (token,), finish_reason))
        self._publish_output_batch(outputs)

    async def _sleep_step(
        self, milliseconds: float, scheduled: ScheduledRequestMetrics, scheduled_ids: set,
    ) -> None:
        self.last_step_ms = milliseconds
        await self.timing.sleep_ms(milliseconds)
        self.expire_retention_leases()
        if self.fpm is not None:
            self.fpm.publish(
                scheduled, self._queued_metrics(scheduled_ids),
                milliseconds / (1000.0 * self.config.speedup_ratio),
            )

    # -- helpers -----------------------------------------------------------

    def _hashes_for(self, request: Request) -> List[bytes]:
        if request._hashes is None:                       # noqa: SLF001
            request._hashes = block_hashes(               # noqa: SLF001
                request.prompt_ids, self.config.block_size, request.cache_salt
            )
        return request._hashes                            # noqa: SLF001

    def _reserve_prefix(self, request: Request) -> bool:
        """Take the cached prefix, then confirm the rest is affordable."""
        self.expire_retention_leases()
        if request.prebuilt:
            # KV arrived over the transfer, so from this worker's point of view
            # the whole prompt is already "cached" -- it is never recomputed.
            request.cached_tokens = len(request.prompt_ids)
            return self.pool.can_allocate(1)

        if not self.config.prefix_caching:
            return self.pool.can_allocate(1)

        if request.blocks:
            # A blocked first chunk already owns its cached references.
            return request.cached_tokens == len(request.prompt_ids) or self.pool.can_allocate(1)

        # Acquire rather than merely match: the matched blocks become this
        # request's, shared with whoever else holds them, so the prefix costs
        # no new memory and cannot be evicted while in use.
        shared = self.pool.acquire_prefix(self._hashes_for(request))
        request.blocks.extend(shared)
        request.cached_tokens = len(shared) * self.config.block_size
        return request.cached_tokens == len(request.prompt_ids) or self.pool.can_allocate(1)

    def _grow_blocks(self, request: Request, token_capacity: int) -> bool:
        self.expire_retention_leases()
        needed = math.ceil(token_capacity / self.config.block_size)
        deficit = needed - len(request.blocks)
        if deficit <= 0:
            return True
        blocks = self.pool.allocate(deficit)
        if blocks is None:
            return False
        request.blocks.extend(blocks)
        return True

    def _cache_prefix(self, request: Request) -> None:
        """Publish this request's prompt blocks into the cache and the router.

        A disaggregated decode is excluded: its KV arrived over the transfer
        rather than being computed here, and the prefill worker already
        announced those blocks. Publishing them again would tell the router two
        workers hold the same prefix when only one computed it.
        """
        if request.prebuilt or not self.config.prefix_caching:
            return
        hashes = self._hashes_for(request)
        added = self.pool.insert_prefix(hashes, request.blocks[: len(hashes)])
        if added:
            self.kv_events.publish_stored(
                token_ids=request.prompt_ids,
                digests=hashes,
                new_indices=[index for index, _ in added],
                cache_salt=request.cache_salt,
            )

    def _emit_first_token(self, request: Request) -> RequestOutput:
        # Always draw, even when the caller supplied the token: a disaggregated
        # decode inherits prefill's first token, and the stream must still
        # advance past position 0 so the rest of the answer continues where
        # prefill left off instead of repeating it.
        drawn = self._next_token(request)
        token = request.first_token if request.first_token is not None else drawn
        will_finish = (
            len(request.generated) + 1 >= request.max_new_tokens
            or request.total_tokens + 1 >= self.config.context_len
        )
        if will_finish:
            self._prepare_retention_capacity(
                request,
                len(request._generated_token_ids) + 1,          # noqa: SLF001
            )
        request.generated.append(token)
        if request.retain_response_kv:
            request._generated_token_ids.append(token)          # noqa: SLF001
        finish_reason = None
        if len(request.generated) >= request.max_new_tokens:
            finish_reason = FINISH_LENGTH
            self._complete(request, finish_reason)
        return RequestOutput(request, (token,), finish_reason)

    def _next_token(self, request: Request) -> int:
        if request._stream is None:                       # noqa: SLF001
            request._stream = token_gen.token_stream(     # noqa: SLF001
                request.prompt_ids, self.config.vocab_size
            )
        return next(request._stream)                      # noqa: SLF001

    def _preempt_one(self, protect: Request) -> bool:
        """LIFO preemption: the newest running request yields its blocks and
        re-queues, keeping what it has generated. Matches the Dynamo mocker's
        default ``preemption_mode=lifo``."""
        for request in reversed(self.running):
            if request is protect or request.finished:
                continue
            self.running.remove(request)
            self.pool.free(request.blocks)
            request.blocks = []
            request.prefilled = 0
            request._hashes = None                        # noqa: SLF001
            request._stream = None                        # noqa: SLF001
            request.prompt_ids = request.prompt_ids + request.generated
            request.max_new_tokens -= len(request.generated)
            request.generated = []
            if request.max_new_tokens <= 0:
                self._finish(request, FINISH_LENGTH)
                return True
            self.waiting.insert(0, request)
            logger.debug("fake-sglang: preempted %s under KV pressure", request.rid)
            return True
        return False

    def _prepare_retention_capacity(
        self, request: Request, generated_token_count: int,
    ) -> None:
        if not request.prebuilt or not request.retain_response_kv:
            return
        rejection, _start, stop, _blocks, _bytes = self._retention_shape(
            request, generated_token_count,
        )
        if rejection is not None:
            request.retention_rejection_reason = rejection
            return
        if stop > len(request.blocks) and not self._grow_blocks(
            request,
            request._original_prompt_token_count + generated_token_count,  # noqa: SLF001
        ):
            request.retention_rejection_reason = "capacity"

    def _retention_shape(
        self, request: Request, generated_token_count: int,
    ) -> Tuple[Optional[str], int, int, int, int]:
        self.expire_retention_leases()
        block_size = self.config.block_size
        start = request._original_prompt_token_count // block_size  # noqa: SLF001
        stop = math.ceil(
            (
                request._original_prompt_token_count               # noqa: SLF001
                + generated_token_count
            )
            / block_size
        )
        block_count = max(stop - start, 0)
        retained_bytes = (
            block_count * block_size * self.config.kv_bytes_per_token
        )

        if request.rid in self._retention_by_request:
            return "duplicate_request", start, stop, block_count, retained_bytes
        if len(self._retention_leases) >= self.config.kv_return_retention_max_leases:
            return "lease_budget", start, stop, block_count, retained_bytes
        if (
            self._retained_blocks + block_count
            > self.config.kv_return_retention_max_blocks
        ):
            return "block_budget", start, stop, block_count, retained_bytes
        if (
            self._retained_bytes + retained_bytes
            > self.config.kv_return_retention_max_bytes
        ):
            return "byte_budget", start, stop, block_count, retained_bytes
        return None, start, stop, block_count, retained_bytes

    def _retain_completed_response(
        self, request: Request,
    ) -> Optional[KVRetentionLease]:
        if (
            not request.prebuilt
            or not request.retain_response_kv
            or request.retention_rejection_reason is not None
        ):
            if request.retention_rejection_reason is not None:
                self._retention_rejections += 1
            return None

        generated_token_ids = tuple(request._generated_token_ids)  # noqa: SLF001
        rejection, start, stop, block_count, retained_bytes = (
            self._retention_shape(request, len(generated_token_ids))
        )
        if rejection is None and stop > len(request.blocks):
            rejection = "capacity"
        if rejection is not None:
            request.retention_rejection_reason = rejection
            self._retention_rejections += 1
            return None

        block_ids = tuple(request.blocks[start:stop])
        if len(block_ids) != block_count:
            request.retention_rejection_reason = "capacity"
            self._retention_rejections += 1
            return None

        created_at = self._clock()
        lease = KVRetentionLease(
            lease_id=uuid.uuid4().hex,
            request_id=request.rid,
            prompt_token_count=request._original_prompt_token_count,  # noqa: SLF001
            generated_token_count=len(generated_token_ids),
            generated_token_ids=generated_token_ids,
            block_ids=block_ids,
            block_count=block_count,
            retained_bytes=retained_bytes,
            cache_salt=request.cache_salt,
            created_at=created_at,
            deadline=created_at
            + self.config.kv_return_retention_timeout_ms / 1000.0,
            _owner=self._retention_owner,
        )
        self._retention_leases[lease.lease_id] = lease
        self._retention_by_request[request.rid] = lease.lease_id
        self._retained_blocks += block_count
        self._retained_bytes += retained_bytes
        request.retention_lease = lease
        self._wakeup.set()
        return lease

    def _finish(self, request: Request, reason: str) -> None:
        if not self._complete(request, reason):
            return
        self._publish_output_batch((RequestOutput(request, finish_reason=reason),))

    def _complete(self, request: Request, reason: str) -> bool:
        if request.finished:
            return False
        request.finished = True
        if request in self.running:
            self.running.remove(request)
        if request in self.waiting:
            self.waiting.remove(request)
        lease = (
            self._retain_completed_response(request)
            if reason in (FINISH_STOP, FINISH_LENGTH)
            else None
        )
        retained = set(lease.block_ids) if lease is not None else set()
        self.pool.free(
            [block for block in request.blocks if block not in retained]
        )
        request.blocks = []
        self.by_rid.pop(request.rid, None)
        return True
