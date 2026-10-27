#!/usr/bin/env bash
# Two aggregated workers behind Dynamo's KV-aware router.
#
# Each worker publishes KV-cache events on its own ZMQ port; the frontend's
# KvIndexer builds a prefix tree per worker from them and routes a request to
# whichever worker already holds the longest matching prefix. Repeat a prompt
# and it should land on the same worker both times.
#
# --page-size matters here: it is the block granularity for both the prefix
# tree and the published events. SGLang's default of 1 works but makes every
# token its own block, which is a lot of events for very little sharing.
#
#   ./launch/agg_kv_router.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_model

PAGE_SIZE="${PAGE_SIZE:-256}"
KV_PORT_A="${KV_PORT_A:-25557}"
KV_PORT_B="${KV_PORT_B:-25657}"

banner "fake-sglang KV-aware routing: frontend :8000 + 2 workers (page size $PAGE_SIZE)"

trap 'echo; echo "Cleaning up..."; kill 0' EXIT

"$PY" -m dynamo.frontend --http-port 8000 --router-mode kv &
FRONTEND=$!

DYN_SYSTEM_PORT=8081 "$PY" -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "${SERVED_NAME:-fake-model}" \
  --skip-tokenizer-init \
  --page-size "$PAGE_SIZE" \
  --kv-events-config "{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$KV_PORT_A\"}" \
  "$@" &
WORKER_A=$!

DYN_SYSTEM_PORT=8082 "$PY" -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "${SERVED_NAME:-fake-model}" \
  --skip-tokenizer-init \
  --page-size "$PAGE_SIZE" \
  --kv-events-config "{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$KV_PORT_B\"}" \
  "$@" &
WORKER_B=$!

wait_any_exit "$FRONTEND" "$WORKER_A" "$WORKER_B"
