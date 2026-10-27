#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE="$ROOT/.local"
DYNAMO_SOURCE="${DYNAMO_SOURCE:-$(dirname "$ROOT")/dynamo}"
DYNAMO_REVISION="701da25292ecca8f64d770cc0b17d389200019a1"
TOKENIZER_REVISION="c1899de289a04d12100db370d81485cdf75e47ca"
export KUBECONFIG="$STATE/kubeconfig"
export DOCKER_HOST="unix://$HOME/.colima/dynamo-sim/docker.sock"
export DOCKER_CONTEXT=colima-dynamo-sim
mkdir -p "$STATE"

fail() { echo "dynamo-sim: $*" >&2; exit 1; }

require_file() {
    [[ -f "$ROOT/$1" ]] || fail "Missing required file: $1"
}

source_fingerprint() {
    git -C "$ROOT" ls-files -c -o --exclude-standard -z |
        python3 -c '
import hashlib
import os
from pathlib import Path
import sys

root = Path(sys.argv[1])
digest = hashlib.sha256()
for raw in sorted(filter(None, sys.stdin.buffer.read().split(b"\0"))):
    path = root / os.fsdecode(raw)
    if path.is_file():
        digest.update(raw)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
print(digest.hexdigest()[:16])
' "$ROOT"
}

expected_image() {
    printf 'fake-sglang-dynamo:1.6.0-local-%s\n' "$(source_fingerprint)"
}

guard_cluster() {
    [[ "$(kubectl config current-context)" == kind-dynamo-sim ]] ||
        fail "Refusing to operate on a different Kubernetes context"
    [[ "$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')" == https://127.0.0.1:* ]] ||
        fail "The lab Kubernetes endpoint must be on localhost"
    [[ "$(docker inspect --format '{{index .Config.Labels "io.x-k8s.kind.cluster"}}' dynamo-sim-control-plane)" == dynamo-sim ]] ||
        fail "The target is not the dedicated kind cluster"
}

read_image() {
    [[ -s "$STATE/image" ]] || fail "No built image recorded; run 'up' first"
    export DYNAMO_SIM_IMAGE
    DYNAMO_SIM_IMAGE="$(cat "$STATE/image")"
}

build_image() {
    local image buildx
    image="$(expected_image)"
    buildx="$(brew --prefix docker-buildx)/bin/docker-buildx"
    [[ -x "$buildx" ]] || fail "Install Homebrew docker-buildx first"
    "$buildx" build --load -t "$image" -f "$ROOT/deploy/local/Dockerfile" "$ROOT"
    docker run --rm \
        -e DYNAMO_SGLANG_PATH=/usr/local/lib/python3.10/site-packages/dynamo/sglang \
        "$image" python -m pytest -q -p no:cacheprovider tests/
    kind load docker-image "$image" --name dynamo-sim
    printf '%s\n' "$image" > "$STATE/image"
    read_image
}

wait_graph() {
    local attempt
    for attempt in {1..120}; do
        if kubectl -n dynamo-sim get dgd fake-sglang -o json | python3 -c '
import json, sys
d = json.load(sys.stdin)
ready = any(c["type"] == "Ready" and c["status"] == "True"
            and c.get("observedGeneration") == d["metadata"]["generation"]
            for c in d.get("status", {}).get("conditions", []))
sys.exit(0 if ready else 1)
'; then
            return
        fi
        sleep 3
    done
    kubectl -n dynamo-sim get dgd,pods
    fail "Deployment did not become Ready within six minutes"
}

deploy_graph() {
    read_image
    python3 "$ROOT/deploy/local/render.py" dynamo | kubectl apply -f -
    wait_graph
}

prepare_benchmark() {
    local trace_dir="${1:-$STATE/mooncake-100rps}"
    for trace in conversation toolagent; do
        [[ -s "$trace_dir/${trace}_trace.jsonl" ]] ||
            fail "Missing Mooncake trace: $trace_dir/${trace}_trace.jsonl"
    done
    if [[ ! -x "$STATE/aiperf-venv/bin/aiperf" ]]; then
        python3 -m venv "$STATE/aiperf-venv"
        "$STATE/aiperf-venv/bin/python" -m pip install \
            --requirement "$ROOT/deploy/local/aiperf-requirements.txt"
    fi
    HF_HOME="$STATE/aiperf-hf" TOKENIZER_REVISION="$TOKENIZER_REVISION" \
        "$STATE/aiperf-venv/bin/python" - <<'PY'
import os
from transformers import AutoTokenizer

AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B",
    revision=os.environ["TOKENIZER_REVISION"],
)
PY
    printf 'AIPerf: %s\nTraces: %s\nTokenizer cache: %s\n' \
        "$STATE/aiperf-venv/bin/aiperf" "$trace_dir" "$STATE/aiperf-hf"
}

up() {
    local tool node attempt
    [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]] ||
        fail "This configuration targets Apple Silicon macOS"
    for tool in colima docker kind kubectl helm brew python3 git; do
        command -v "$tool" >/dev/null || fail "Missing tool: $tool"
    done
    [[ "$(git -C "$DYNAMO_SOURCE" rev-parse HEAD)" == "$DYNAMO_REVISION" ]] ||
        fail "DYNAMO_SOURCE must point at the pinned NVIDIA Dynamo v1.6.0 revision"
    if ! colima status --profile dynamo-sim >/dev/null 2>&1; then
        colima start --profile dynamo-sim --cpu 8 --memory 24 --disk 100 \
            --vm-type vz --arch aarch64 --activate=false --ssh-config=false \
            --mount none --binfmt=false --port-forwarder grpc
    fi
    if ! kind get clusters | grep -Fxq dynamo-sim; then
        kind create cluster --name dynamo-sim --image kindest/node:v1.35.0 \
            --config "$ROOT/deploy/local/kind.yaml" --kubeconfig "$KUBECONFIG"
    else
        kind export kubeconfig --name dynamo-sim --kubeconfig "$KUBECONFIG"
    fi
    guard_cluster
    for node in dynamo-sim-worker dynamo-sim-worker2; do
        kubectl label node "$node" dynamo-sim.local/gpu-capacity=simulated --overwrite
        kubectl patch node "$node" --subresource=status --type=merge \
            -p '{"status":{"capacity":{"nvidia.com/gpu":"10"}}}'
    done
    for attempt in {1..30}; do
        if kubectl get nodes -o json | python3 -c '
import json, sys
nodes = {n["metadata"]["name"]: n for n in json.load(sys.stdin)["items"]}
sys.exit(0 if all(nodes[n]["status"]["allocatable"].get("nvidia.com/gpu") == "10"
                  for n in ("dynamo-sim-worker", "dynamo-sim-worker2")) else 1)
'; then
            break
        fi
        [[ "$attempt" != 30 ]] || fail "Simulated GPU capacity was not advertised"
        sleep 2
    done
    python3 "$ROOT/deploy/local/render.py" infrastructure | kubectl apply -f -
    python3 "$ROOT/deploy/local/render.py" observability | kubectl apply --server-side -f -
    kubectl -n observability rollout status \
        deployment/otel-collector deployment/prometheus --timeout=180s
    helm upgrade --install dynamo-platform \
        "$DYNAMO_SOURCE/deploy/helm/charts/platform/components/operator" \
        --namespace dynamo-system --values "$ROOT/deploy/local/operator-values.yaml" \
        --wait --timeout 5m
    kubectl wait --for=condition=Established --timeout=60s \
        crd/dynamographdeployments.nvidia.com crd/dynamoworkermetadatas.nvidia.com
    if [[ -s "$STATE/image" ]] && [[ "$(cat "$STATE/image")" == "$(expected_image)" ]]; then
        read_image
        if ! docker image inspect "$DYNAMO_SIM_IMAGE" >/dev/null 2>&1; then
            build_image
        else
            kind load docker-image "$DYNAMO_SIM_IMAGE" --name dynamo-sim
        fi
    else
        build_image
    fi
    deploy_graph
    printf 'Inference: http://127.0.0.1:8000\nPrometheus: http://127.0.0.1:9090\n'
}

case "${1:-status}" in
    up) up ;;
    rebuild) guard_cluster; build_image; deploy_graph ;;
    deploy) guard_cluster; deploy_graph ;;
    benchmark-setup) shift; prepare_benchmark "$@" ;;
    benchmark)
        shift
        require_file scripts/mooncake_lab.py
        [[ -x "$STATE/aiperf-venv/bin/aiperf" ]] ||
            fail "Run 'benchmark-setup' before the benchmark"
        guard_cluster
        exec python3 "$ROOT/scripts/mooncake_lab.py" "$@"
        ;;
    report)
        shift
        require_file scripts/mooncake_report.py
        exec python3 "$ROOT/scripts/mooncake_report.py" "$@"
        ;;
    status)
        guard_cluster
        kubectl -n dynamo-sim get dgd,pods
        kubectl -n observability get pods
        ;;
    stop) colima stop --profile dynamo-sim ;;
    *)
        fail "Usage: bash deploy/local/lab.sh {up|rebuild|deploy|benchmark-setup|benchmark|report|status|stop}"
        ;;
esac
