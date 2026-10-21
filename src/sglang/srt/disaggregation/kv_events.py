"""KV-cache event publishing over ZMQ.

A faithful reimplementation of SGLang's
``python/sglang/srt/disaggregation/kv_events.py`` wire format, because Dynamo's
Rust router parses these bytes directly: ``publisher.py`` hands the ZMQ
endpoint to the Rust ``KvEventPublisher``, which subscribes and decodes without
any Python in the path (see ``dynamo/lib/kv-router/src/zmq_wire/``).

Wire format, which must match exactly:

* Three ZMQ frames per batch: ``(topic, seq_be64, msgpack_payload)``.
* The payload is a ``KVEventBatch``, an ``array_like`` struct encoding as
  ``[ts, events, attn_dp_rank]``.
* Each event is a *tagged map* -- msgspec writes the class name under a
  ``type`` key, then the field names. Dynamo parses both this and vLLM's
  positional form; the map form is what current SGLang emits.

What Dynamo does with the fields is the part worth knowing when reading
``fakeengine.kv_events``: ``block_hashes`` are treated as **opaque engine-side
ids** used only for identity and removal, while the router's prefix-matching
key is recomputed on the Rust side from ``token_ids`` via
``compute_block_hash_for_seq``. So the token payload has to be exactly right;
the hashes only have to be stable and collision-free.
"""

from __future__ import annotations

import atexit
import enum
import logging
import queue
import threading
from itertools import count
from queue import Queue
from typing import Any, List, Optional, Union

import msgspec

logger = logging.getLogger(__name__)


class EventBatch(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    ts: float
    events: List[Any]
    attn_dp_rank: Optional[int] = None


class KVCacheEvent(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
    tag=True,
):
    """Base for KV-cache events: a tagged msgpack map keyed by field name."""


class StorageMedium(str, enum.Enum):
    GPU = "GPU"
    CPU = "CPU_PINNED"
    DISK = "DISK"
    EXTERNAL = "EXTERNAL"


class BlockStored(KVCacheEvent):
    # One opaque id per block. Dynamo keys its tree by these for removal.
    block_hashes: List[int]
    # External hash of the block preceding the first one here; None starts a
    # new chain. This is how a partial store attaches to an existing prefix.
    parent_block_hash: Optional[int]
    # FLAT token ids covering every block in this event, in order. Dynamo
    # slices it block_size at a time, so it must be exactly
    # ``block_size * len(block_hashes)`` long.
    token_ids: List[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str] = None
    cache_salt: Optional[str] = None
    session_id: Optional[str] = None


class BlockRemoved(KVCacheEvent):
    block_hashes: List[int]
    medium: Optional[str] = None


class AllBlocksCleared(KVCacheEvent):
    pass


class KVEventBatch(EventBatch):
    events: List[Union[BlockStored, BlockRemoved, AllBlocksCleared]]


class EventPublisher:
    def publish(self, events: EventBatch) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class NullEventPublisher(EventPublisher):
    def publish(self, events) -> None:
        return None

    def shutdown(self) -> None:
        return None


class ZmqEventPublisher(EventPublisher):
    """PUB-socket publisher with a background sender thread.

    Binds when the endpoint names a wildcard or an IPC path and connects
    otherwise -- the same rule SGLang uses, and the reason Dynamo can take the
    advertised ``tcp://*:5557``, substitute the local IP, and connect to it.
    """

    SHUTDOWN_TIMEOUT: float = 1.0

    def __init__(
        self,
        attn_dp_rank: int = 0,
        endpoint: str = "tcp://*:5557",
        replay_endpoint: Optional[str] = None,
        buffer_steps: int = 10_000,
        hwm: int = 100_000,
        max_queue_size: int = 100_000,
        topic: str = "",
        **kwargs: Any,
    ) -> None:
        import zmq

        self._event_queue: "Queue[Optional[EventBatch]]" = Queue(maxsize=max_queue_size)
        self._dp_rank = attn_dp_rank
        self._endpoint = self.offset_endpoint_port(endpoint, attn_dp_rank) or endpoint
        self._topic_bytes = topic.encode("utf-8")
        self._seq_gen = count()

        self._ctx = zmq.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        try:
            self._pub.set_hwm(hwm)
            if (
                "*" in self._endpoint
                or "::" in self._endpoint
                or "0.0.0.0" in self._endpoint
                or self._endpoint.startswith("ipc://")
                or self._endpoint.startswith("inproc://")
            ):
                self._pub.bind(self._endpoint)
                logger.info("fake-sglang KV events publishing on %s", self._endpoint)
            else:
                self._pub.connect(self._endpoint)
                logger.info("fake-sglang KV events connecting to %s", self._endpoint)
        except Exception:
            self._pub.close(linger=0)
            raise

        self._running = True
        self._thread = threading.Thread(
            target=self._publisher_thread, daemon=True, name="fake-zmq-kv-publisher"
        )
        self._thread.start()
        atexit.register(self.shutdown)

    @staticmethod
    def offset_endpoint_port(endpoint: Optional[str], dp_rank: int) -> Optional[str]:
        """Give each DP rank its own port: ``tcp://host:5557`` + rank.

        Returns ``None`` for an endpoint with no parseable port, which is what
        Dynamo checks before it gives up on that rank.
        """
        if not endpoint or "://" not in endpoint:
            return None
        scheme, _, rest = endpoint.partition("://")
        host, sep, port = rest.rpartition(":")
        if not sep or not port.isdigit():
            return None
        return f"{scheme}://{host}:{int(port) + int(dp_rank)}"

    def publish(self, events: EventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")
        if events.attn_dp_rank is None:
            events.attn_dp_rank = self._dp_rank
        self._event_queue.put(events)

    def _publisher_thread(self) -> None:
        packer = msgspec.msgpack.Encoder()
        while self._running or not self._event_queue.empty():
            try:
                batch = self._event_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if batch is None:
                break
            try:
                seq = next(self._seq_gen)
                self._pub.send_multipart(
                    (self._topic_bytes, seq.to_bytes(8, "big"), packer.encode(batch))
                )
            except Exception as exc:  # noqa: BLE001 - never kill the sender thread
                logger.warning("fake-sglang: dropping KV event batch: %s", exc)

    def shutdown(self) -> None:
        if not self._running:
            return
        self._running = False
        self._event_queue.put(None)
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        try:
            self._pub.close(linger=0)
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass

    def close(self) -> None:
        self.shutdown()
