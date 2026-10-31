#!/usr/bin/env bash
# Any mix of prefill / decode / aggregated workers, placed in a network
# topology, from one YAML file. See launch/deployment.py and launch/deployments/.
#   ./launch/deployment.sh launch/deployments/two-regions.yaml [--dry-run] [-- worker args]
set -euo pipefail
source "$(dirname "$0")/common.sh"
PY="$PY" exec "$PY" "$ROOT/launch/deployment.py" "$@"
