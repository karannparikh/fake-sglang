"""Continuous-batching scheduler.

Models the parts of an LLM scheduler that change observable behavior:

* chunked prefill against a token budget, so a long prompt does not stall the
  whole batch for one step;
* a shared decode step whose cost grows with batch size, so queueing shows up
  as ITL growth the way it does on a GPU;
* a finite KV pool with LIFO preemption, so memory pressure is reachable;
* prefix reuse, so a repeated prompt reports non-zero ``cached_tokens``.

"Prebuilt" requests are the decode side of disaggregation: their KV already
arrived over the (simulated) transfer, so they skip prefill compute but still
occupy blocks. SGLang calls the same thing ``prepare_for_prebuilt`` /
``process_prebuilt`` in ``disaggregation/decode_schedule_batch_mixin.py``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Sequence

from fakeengine import tokens as token_gen
from fakeengine.config import SimConfig
from fakeengine.kv_events import NullKvEventEmitter
from fakeengine.kvcache import BlockPool, block_hashes
from fakeengine.timing import TimingModel

logger = logging.getLogger(__name__)

FINISH_STOP = "stop"
FINISH_LENGTH = "length"
FINISH_ABORT = "abort"


class Request:
    __slots__ = (
        "rid", "prompt_ids", "max_new_tokens", "prebuilt", "first_token",
        "queue", "blocks", "prefilled", "generated", "cached_tokens",
        "_stream", "_hashes", "finished", "arrival", "cache_salt",
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
    ) -> None:
        self.rid = rid
        # Prefix-cache namespace: only requests with the same salt share KV.
        self.cache_salt = cache_salt or None
        self.prompt_ids = list(prompt_ids)
        self.max_new_tokens = max(int(max_new_tokens), 1)
        self.prebuilt = prebuilt
        self.first_token = first_token
        self.queue: asyncio.Queue = asyncio.Queue()
        self.blocks: List[int] = []
        self.prefilled = 0
        self.generated: List[int] = []
        self.cached_tokens = 0
        self._hashes: Optional[List[bytes]] = None
        self._stream = None
        self.finished = False
        self.arrival = time.monotonic()

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.generated)


class Scheduler:
    def __init__(self, config: SimConfig, kv_events=None) -> None:
        self.config = config
        self.timing = TimingModel(config)
        # The emitter is wired into the pool itself so an eviction cannot
        # happen without the router hearing about it, whichever code path
        # triggered it.
        self.kv_events = kv_events or NullKvEventEmitter()
        self.pool = BlockPool(
            config.num_kv_blocks,
            config.block_size,
            on_evict=self.kv_events.publish_removed,
        )
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.by_rid: Dict[str, Request] = {}
        self._wakeup = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self.last_step_ms = 0.0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        self._stopping = True
        self._wakeup.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

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

    def flush_cache(self) -> None:
        """Drop cached prefixes. Only unreferenced blocks can go, so this is a
        best-effort hint exactly as it is in a real engine."""
        self.pool.clear_cache()

    # -- stats -------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "num_running_reqs": len(self.running),
            "num_waiting_reqs": len(self.waiting),
            "num_used_tokens": self.pool.used_blocks * self.config.block_size,
            "token_usage": self.pool.usage,
            "kv_used_blocks": self.pool.used_blocks,
            "kv_total_blocks": self.pool.num_blocks,
            "kv_cached_blocks": self.pool.cached_blocks,
            "last_step_ms": self.last_step_ms,
        }

    # -- the loop ----------------------------------------------------------

    async def _run(self) -> None:
        while not self._stopping:
            if not self.waiting and not self.running:
                self._wakeup.clear()
                await self._wakeup.wait()
                continue
            if await self._prefill_step():
                continue
            if self.running:
                await self._decode_step()
                continue
            # Waiting requests that no step can admit: the KV pool is full of
            # referenced blocks. Yield rather than spin; a completion frees them.
            await asyncio.sleep(0.001)

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
        moved_tokens = 0        # tokens advanced, cached or not (progress check)

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

            # Cached prefix blocks and KV that arrived over the transfer are
            # occupancy, not compute: they must not be billed as prefill time.
            free_tokens = max(request.cached_tokens - request.prefilled, 0)
            computed_tokens += max(chunk - free_tokens, 0)
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

        await self._sleep_step(self.timing.prefill_ms(computed_tokens))

        for request in admitted:
            self._cache_prefix(request)
            self.running.append(request)
            self._emit_first_token(request)
        return True

    async def _decode_step(self) -> None:
        if not self.running:
            return
        await self._sleep_step(self.timing.decode_step_ms(len(self.running)))

        for request in list(self.running):
            if request.finished:
                continue
            if not self._grow_blocks(request, request.total_tokens + 1):
                if not self._preempt_one(protect=request):
                    self._finish(request, FINISH_ABORT)
                    continue
                if not self._grow_blocks(request, request.total_tokens + 1):
                    self._finish(request, FINISH_ABORT)
                    continue

            token = self._next_token(request)
            request.generated.append(token)
            request.queue.put_nowait(("token", token))

            if len(request.generated) >= request.max_new_tokens:
                self._finish(request, FINISH_LENGTH)
            elif request.total_tokens >= self.config.context_len:
                self._finish(request, FINISH_LENGTH)

    async def _sleep_step(self, milliseconds: float) -> None:
        self.last_step_ms = milliseconds
        await self.timing.sleep_ms(milliseconds)

    # -- helpers -----------------------------------------------------------

    def _hashes_for(self, request: Request) -> List[bytes]:
        if request._hashes is None:                       # noqa: SLF001
            request._hashes = block_hashes(               # noqa: SLF001
                request.prompt_ids, self.config.block_size, request.cache_salt
            )
        return request._hashes                            # noqa: SLF001

    def _reserve_prefix(self, request: Request) -> bool:
        """Take the cached prefix, then confirm the rest is affordable."""
        if request.prebuilt:
            # KV arrived over the transfer, so from this worker's point of view
            # the whole prompt is already "cached" -- it is never recomputed.
            request.cached_tokens = len(request.prompt_ids)
            return self.pool.can_allocate(1)

        if not self.config.prefix_caching:
            return self.pool.can_allocate(1)

        # Acquire rather than merely match: the matched blocks become this
        # request's, shared with whoever else holds them, so the prefix costs
        # no new memory and cannot be evicted while in use.
        shared = self.pool.acquire_prefix(self._hashes_for(request))
        request.blocks.extend(shared)
        request.cached_tokens = len(shared) * self.config.block_size
        return self.pool.can_allocate(1)

    def _grow_blocks(self, request: Request, token_capacity: int) -> bool:
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

    def _emit_first_token(self, request: Request) -> None:
        # Always draw, even when the caller supplied the token: a disaggregated
        # decode inherits prefill's first token, and the stream must still
        # advance past position 0 so the rest of the answer continues where
        # prefill left off instead of repeating it.
        drawn = self._next_token(request)
        token = request.first_token if request.first_token is not None else drawn
        request.generated.append(token)
        request.queue.put_nowait(("token", token))
        if len(request.generated) >= request.max_new_tokens:
            self._finish(request, FINISH_LENGTH)

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

    def _finish(self, request: Request, reason: str) -> None:
        if request.finished:
            return
        request.finished = True
        if request in self.running:
            self.running.remove(request)
        if request in self.waiting:
            self.waiting.remove(request)
        self.pool.free(request.blocks)
        request.blocks = []
        self.by_rid.pop(request.rid, None)
        request.queue.put_nowait(("done", reason))
