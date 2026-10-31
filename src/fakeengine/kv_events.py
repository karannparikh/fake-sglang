"""Turning KV block lifecycle into router-visible events.

The wire format lives in ``sglang.srt.disaggregation.kv_events``; this module
owns the *policy* -- which blocks get announced, when, and with what token
payload. Getting that payload right is the whole job: Dynamo's Rust router
recomputes its prefix-matching key from ``token_ids``
(``lib/kv-router/src/zmq_wire/convert.rs``), so a wrong or misaligned token
slice produces hashes that silently never match the frontend's, and KV-aware
routing degrades to round-robin with no error anywhere.

Three rules follow from how Dynamo parses a store:

1. **Only whole blocks.** A block whose token count is not exactly
   ``block_size`` is dropped with a warning on the Rust side, so a partial
   trailing block is never published.
2. **Flat, contiguous tokens.** ``token_ids`` covers every block in the event,
   in order, and is sliced ``block_size`` at a time.
3. **Chain to the parent.** ``parent_block_hash`` is the external id of the
   block just before the first one published here, so a store that extends an
   existing cached prefix attaches to it instead of starting a new root.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)

from fakeengine.kvcache import external_id

logger = logging.getLogger(__name__)


class KvEventEmitter:
    """Publishes block stores and removals for one DP rank."""

    def __init__(
        self,
        block_size: int,
        endpoint: str,
        *,
        dp_rank: int = 0,
        topic: str = "",
    ) -> None:
        self.block_size = block_size
        self.dp_rank = dp_rank
        self._publisher = ZmqEventPublisher(
            attn_dp_rank=dp_rank, endpoint=endpoint, topic=topic
        )

    # -- emission ----------------------------------------------------------

    def publish_stored(
        self,
        *,
        token_ids: Sequence[int],
        digests: Sequence[bytes],
        new_indices: Sequence[int],
        cache_salt: Optional[str] = None,
    ) -> None:
        """Announce newly cached blocks of one sequence.

        ``digests`` are the sequence's per-block cumulative hashes and
        ``new_indices`` selects the ones that were just added. ``cache_salt``
        goes out on the event, as in SGLang: Dynamo's router reads it as the
        block's cache namespace and folds it into the prefix key it
        recomputes from ``token_ids``, so a salted request only matches blocks
        stored under the same salt. Only a
        contiguous run starting at the lowest new index is published: the
        blocks form a chain, and a gap would leave later blocks parented to an
        id the router never saw.
        """
        if not new_indices:
            return

        start = min(new_indices)
        expected = set(range(start, start + len(new_indices)))
        if set(new_indices) != expected:
            # Shouldn't happen -- prefix matching is contiguous from the start
            # -- but publishing a broken chain is worse than publishing none.
            logger.warning(
                "fake-sglang: non-contiguous new blocks %s; skipping KV store event",
                sorted(new_indices),
            )
            return

        run = list(digests[start : start + len(new_indices)])
        token_start = start * self.block_size
        token_end = token_start + len(run) * self.block_size
        tokens = list(token_ids[token_start:token_end])
        if len(tokens) != len(run) * self.block_size:
            logger.warning(
                "fake-sglang: %d tokens for %d blocks of size %d; skipping KV store event",
                len(tokens),
                len(run),
                self.block_size,
            )
            return

        parent = external_id(digests[start - 1]) if start > 0 else None
        self._emit(
            BlockStored(
                block_hashes=[external_id(d) for d in run],
                parent_block_hash=parent,
                token_ids=tokens,
                block_size=self.block_size,
                lora_id=None,
                cache_salt=cache_salt or None,
            )
        )

    def publish_removed(self, block_ids: Sequence[int]) -> None:
        if block_ids:
            self._emit(BlockRemoved(block_hashes=list(block_ids)))

    def _emit(self, event) -> None:
        try:
            self._publisher.publish(KVEventBatch(ts=time.time(), events=[event]))
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail a request
            logger.warning("fake-sglang: failed to publish KV event: %s", exc)

    def shutdown(self) -> None:
        self._publisher.shutdown()


class NullKvEventEmitter:
    """Used when KV event publishing is off, so callers need no branches."""

    def publish_stored(self, **kwargs) -> None:
        return None

    def publish_removed(self, block_ids: Sequence[int]) -> None:
        return None

    def shutdown(self) -> None:
        return None


def build_emitter(server_args, block_size: int, dp_rank: int = 0):
    """Create an emitter from ``--kv-events-config``, or a null one.

    Dynamo derives ``use_kv_events`` from this same config
    (``dynamo/components/src/dynamo/sglang/args.py``), so an engine that
    publishes and a Dynamo that subscribes are enabled by one flag.
    """
    import json

    raw = getattr(server_args, "kv_events_config", None)
    if not raw:
        return NullKvEventEmitter()
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("fake-sglang: unparseable --kv-events-config %r", raw)
        return NullKvEventEmitter()
    if config.get("publisher", "null") == "null":
        return NullKvEventEmitter()

    endpoint = config.get("endpoint")
    if not endpoint:
        raise ValueError("--kv-events-config sets a publisher but no 'endpoint'")
    if endpoint.startswith("ipc://"):
        raise ValueError(
            "fake-sglang does not support IPC endpoints in --kv-events-config; "
            "use a TCP endpoint such as tcp://*:25557 for per-rank port offsets"
        )

    return KvEventEmitter(
        block_size=block_size,
        endpoint=endpoint,
        dp_rank=dp_rank,
        topic=config.get("topic", ""),
    )
