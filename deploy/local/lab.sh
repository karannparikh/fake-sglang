#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE="$ROOT/.local"
DYNAMO_SOURCE="${DYNAMO_SOURCE:-$(dirname "$ROOT")/dynamo}"
export KUBECONFIG="$STATE/kubeconfig"
export DOCKER_HOST="unix://$HOME/.colima/dynamo-sim/docker.sock"
export DOCKER_CONTEXT=colima-dynamo-sim
mkdir -p "$STATE"

fail() { echo "dynamo-sim: $*" >&2; exit 1; }

require_benchmark_script() {
    [[ -f "$ROOT/$1" ]] ||
        fail "Missing benchmark script: $1; restore the complete local benchmark bundle"
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
    image="fake-sglang-dynamo:1.5.0-local-$(date -u +%Y%m%d%H%M%S)"
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

grafana_secret() {
    kubectl create namespace observability --dry-run=client -o yaml | kubectl apply -f -
    if [[ ! -s "$STATE/grafana-password" ]]; then
        if kubectl -n observability get secret grafana-admin >/dev/null 2>&1; then
            kubectl -n observability get secret grafana-admin -o json |
                python3 -c 'import base64,json,sys; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)["data"]["password"]))' \
                > "$STATE/grafana-password"
        else
            python3 -c 'import secrets,sys; sys.stdout.write(secrets.token_urlsafe(24))' > "$STATE/grafana-password"
        fi
    fi
    python3 - "$STATE/grafana-password" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
path.write_text(path.read_text().rstrip("\r\n"))
PY
    chmod 600 "$STATE/grafana-password"
    kubectl -n observability create secret generic grafana-admin \
        --from-file=password="$STATE/grafana-password" --dry-run=client -o yaml |
        kubectl apply -f -
}

up() {
    local tool node attempt
    [[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]] ||
        fail "This configuration targets Apple Silicon macOS"
    for tool in colima docker kind kubectl helm brew python3 git; do
        command -v "$tool" >/dev/null || fail "Missing tool: $tool"
    done
    [[ "$(git -C "$DYNAMO_SOURCE" rev-parse HEAD)" == b83b1d9304ebfc624709ac46db32b1b6f1ff1615 ]] ||
        fail "DYNAMO_SOURCE must point at NVIDIA Dynamo v1.5.0"
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
    kubectl apply --server-side -f "$DYNAMO_SOURCE/deploy/operator/config/crd/bases"
    kubectl wait --for=condition=Established --timeout=60s \
        crd/dynamographdeployments.nvidia.com crd/dynamoworkermetadatas.nvidia.com
    grafana_secret
    python3 "$ROOT/deploy/local/render.py" observability | kubectl apply --server-side -f -
    kubectl -n observability rollout restart deployment
    kubectl -n observability rollout status deployment --timeout=180s
    helm upgrade --install dynamo-platform \
        "$DYNAMO_SOURCE/deploy/helm/charts/platform/components/operator" \
        --namespace dynamo-system --values "$ROOT/deploy/local/operator-values.yaml" \
        --wait --timeout 5m
    if [[ -s "$STATE/image" ]]; then
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
    printf 'Inference: http://127.0.0.1:8000\nGrafana: http://127.0.0.1:3000/d/dynamo-local-sglang\n'
}

case "${1:-status}" in
    up) up ;;
    rebuild) guard_cluster; build_image; deploy_graph ;;
    deploy) guard_cluster; deploy_graph ;;
    status)
        guard_cluster
        kubectl -n dynamo-sim get dgd,pods
        kubectl -n observability get pods
        ;;
    traffic)
        shift; require_benchmark_script scripts/traffic.py
        exec python3 "$ROOT/scripts/traffic.py" "$@"
        ;;
    routing-test)
        shift; require_benchmark_script deploy/local/routing_compare.py; guard_cluster
        exec python3 "$ROOT/deploy/local/routing_compare.py" "$@"
        ;;
    routing-dashboard)
        shift; require_benchmark_script deploy/local/routing_dashboard.py; guard_cluster
        exec python3 "$ROOT/deploy/local/routing_dashboard.py" "$@"
        ;;
    dashboard)
        guard_cluster
        python3 "$ROOT/deploy/local/dashboard.py"
        kubectl -n observability create configmap grafana-dashboard \
            --from-file=dynamo-lab.json="$ROOT/deploy/local/dashboard.json" \
            --from-file=dynamo-routing-live.json="$ROOT/deploy/local/routing-live-dashboard.json" \
            --dry-run=client -o yaml | kubectl apply --server-side -f -
        ;;
    stop) colima stop --profile dynamo-sim ;;
    *) fail "Usage: bash deploy/local/lab.sh {up|rebuild|deploy|dashboard|status|traffic|routing-test|routing-dashboard|stop}" ;;
esac
