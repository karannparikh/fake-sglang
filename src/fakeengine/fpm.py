"""Simulated forward-pass telemetry in SGLang's ZMQ wire format."""

from __future__ import annotations

import logging
import time

import zmq

from sglang.srt.observability.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
    encode,
)

logger = logging.getLogger(__name__)


class ForwardPassPublisher:
    def __init__(self, ipc_name: str, worker_id: str, dp_rank: int = 0) -> None:
        if not ipc_name or not worker_id:
            raise ValueError("FPM requires Dynamo's IPC endpoint and worker identity")
        self.worker_id = worker_id
        self.dp_rank = dp_rank
        self.counter = 0
        self.last_publish = 0.0
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDHWM, 1024)
        try:
            self._socket.bind(f"{ipc_name}.{dp_rank}")
        except zmq.ZMQError:
            self.close()
            raise

    def publish(
        self,
        scheduled: ScheduledRequestMetrics,
        queued: QueuedRequestMetrics,
        wall_time: float,
    ) -> None:
        message = ForwardPassMetrics(
            worker_id=self.worker_id,
            dp_rank=self.dp_rank,
            counter_id=self.counter,
            wall_time=wall_time,
            scheduled_requests=scheduled,
            queued_requests=queued,
        )
        try:
            self._socket.send_multipart(
                [b"", self.counter.to_bytes(8, "big"), encode(message)],
                flags=zmq.NOBLOCK,
            )
        except zmq.ZMQError:
            logger.exception("fake-sglang: failed to publish forward-pass telemetry")
        self.counter += 1
        self.last_publish = time.monotonic()

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()
