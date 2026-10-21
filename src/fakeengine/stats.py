"""Scheduler-metrics publishing.

Dynamo's ``publisher.py`` binds a ZMQ PULL socket on
``engine.port_args.metrics_ipc_name`` the moment a worker starts, whether or
not the engine ever writes to it. Leaving it silent would make every fake
worker report zero load, which quietly breaks load-based routing and the
Planner -- so the scheduler pushes its real counters at SGLang's own cadence.

KV events (the prefix-tree feed) and observed prefix-cache metrics are
published separately; see the README.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from sglang.srt.observability.scheduler_metrics_mixin import KvMetrics

from fakeengine.scheduler import Scheduler

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 0.1


class MetricsPusher:
    def __init__(
        self,
        scheduler: Scheduler,
        ipc_name: str,
        *,
        dp_rank: int = 0,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        self.scheduler = scheduler
        self.ipc_name = ipc_name
        self.dp_rank = dp_rank
        self.interval_s = interval_s
        self._task: Optional[asyncio.Task] = None
        self._socket = None
        self._context = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._close()

    async def _run(self) -> None:
        try:
            import zmq
        except ImportError:
            logger.warning("fake-sglang: pyzmq missing; scheduler metrics disabled")
            return

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUSH)
        # Connect, never bind: Dynamo's publisher owns the bind end, and it may
        # not exist yet when the engine starts. ZMQ reconnects on its own.
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 8)         # drop stale load, don't queue it
        self._socket.connect(self.ipc_name)

        while True:
            try:
                self._socket.send_pyobj(self._snapshot(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass                                   # nobody draining; skip this tick
            except Exception as exc:  # noqa: BLE001
                logger.debug("fake-sglang: metrics push failed: %s", exc)
            await asyncio.sleep(self.interval_s)

    def _snapshot(self) -> KvMetrics:
        stats = self.scheduler.stats()
        return KvMetrics(
            request_active_slots=stats["num_running_reqs"],
            request_total_slots=self.scheduler.config.max_running_requests,
            # Dynamo divides these by page_size, so they are token counts.
            kv_active_blocks=stats["num_used_tokens"],
            kv_total_blocks=self.scheduler.config.total_kv_tokens,
            num_requests_waiting=stats["num_waiting_reqs"],
            gpu_cache_usage_perc=stats["token_usage"],
            gpu_prefix_cache_hit_rate=(
                stats["prefix_cache_hit_tokens"] / stats["prefix_cache_query_tokens"]
                if stats["prefix_cache_query_tokens"] else 0.0
            ),
            data_parallel_rank=self.dp_rank,
        )

    def _close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
