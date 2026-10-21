"""Forward-pass metrics (FPM) -- per-iteration scheduler telemetry.

This is a *wire contract*, not a convenience type: Dynamo encodes with
SGLang's structs and decodes with its own
(``dynamo/components/src/dynamo/common/forward_pass_metrics.py``), so the
msgspec field names, order, and types must match that module field for field.
``dynamo/components/src/dynamo/sglang/tests/test_fpm_contract.py`` asserts
exactly that, in both directions.

With ``--enable-forward-pass-metrics``, ``fakeengine.fpm`` publishes this
schema through Dynamo's injected per-rank IPC endpoint. Timings describe
simulated scheduler passes, not measured GPU execution.
"""

from __future__ import annotations

import logging

import msgspec

logger = logging.getLogger(__name__)

FPM_VERSION: int = 1


class ScheduledRequestMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,
):
    """Requests scheduled and executed in this iteration."""

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    sum_prefill_kv_tokens: int = 0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class QueuedRequestMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,
):
    """Requests waiting in the queue and not scheduled this iteration."""

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class ForwardPassMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,
):
    """One message per scheduler iteration."""

    version: int = FPM_VERSION
    worker_id: str = ""
    dp_rank: int = 0
    counter_id: int = 0
    wall_time: float = 0.0
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()


_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)


def encode(metrics: ForwardPassMetrics) -> bytes:
    return _encoder.encode(metrics)


def decode(data: bytes) -> "ForwardPassMetrics | None":
    try:
        metrics = _decoder.decode(data)
    except Exception:  # noqa: BLE001 - a bad message must not kill the consumer
        logger.warning("Failed to decode ForwardPassMetrics message, skipping")
        return None
    if metrics.version != FPM_VERSION:
        logger.warning(
            "Unsupported ForwardPassMetrics version %d (expected %d), skipping",
            metrics.version,
            FPM_VERSION,
        )
        return None
    return metrics
