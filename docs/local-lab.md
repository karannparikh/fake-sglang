# Local Dynamo + fake-SGLang lab

An operating guide for the native macOS workflows included in this repository.
Dynamo supplies the frontend, routing, and SGLang integration; fake-SGLang
replaces the engine with a CPU-only simulation. No model weights or GPUs are
needed. Output is deterministic noise, and simulated timings are not GPU
performance measurements.

This guide uses only the bootstrap script, launchers, configuration examples,
and model metadata available on `main`. It does not describe a Kubernetes
deployment, provision dashboards, or provide a benchmark replay/reporting
pipeline.

## Prerequisites and setup

Use a dedicated development directory with sibling source checkouts:

```text
workspace/
  dynamo/
  fake-sglang/
```

Obtain Dynamo and this repository through your approved source-management
process. Dynamo is an external dependency: having fake-SGLang alone is not
enough to launch the frontend or workers. Use a Dynamo revision compatible
with the shim; this repository does not pin the bootstrap to a Dynamo release.
See the [source-contract checks](#source-contract-checks) below when changing
that checkout.

Before bootstrapping, install approved versions of Homebrew, CMake, Protobuf
(`protoc`), Rust/Cargo, and `uv` through your normal tooling process. Review
[`scripts/bootstrap_macos.sh`](../scripts/bootstrap_macos.sh) first: it can
install missing system tools and download Rust or `uv` installers. Preinstall
the approved toolchain rather than allowing unreviewed installer scripts.
Python packages and Rust build dependencies also require approved sources.

Run these commands from the fake-SGLang repository root:

```bash
./scripts/bootstrap_macos.sh
source ../.venv/bin/activate
```

The script creates or reuses `../.venv`, defaults to Python 3.10, builds
Dynamo's Python bindings from the sibling source checkout, installs Dynamo
without its real SGLang extra, and installs fake-SGLang. Its final checks
verify the shim and both Dynamo entry points. Do not continue after a failed
verification.

The bootstrap rebuilds the Rust bindings on reruns. If an existing virtualenv
uses a different Python version, it stops rather than replacing it silently.
Do not use `FORCE_RECREATE=1` on an environment shared with other work.
See [README setup](../README.md#setup) for bootstrap overrides.

The default model is the bundled `models/qwen3-0.6b/` directory: configuration
and tokenizer files only, not weights. Its provenance is recorded in
[`SOURCE.md`](../models/qwen3-0.6b/SOURCE.md).

## Choose a launch mode

The native launchers default to file-based discovery and use a TCP request
plane. They do not require etcd, NATS, Docker, Colima, or Kubernetes.

| Launcher | Processes |
|---|---|
| `./launch/agg.sh` | One frontend and one worker doing both prefill and decode |
| `./launch/disagg.sh` | One frontend, one prefill worker, and one decode worker |
| `./launch/agg_kv_router.sh` | One frontend with KV-aware routing and two aggregated workers |
| `./launch/cluster.sh <config.yaml>` | One frontend and the worker groups defined in a YAML configuration |

Run one mode at a time unless you deliberately isolate its ports and
discovery registry. The simple shell launchers share default discovery state
and use frontend port 8000; they are not independent concurrent deployments.

For the smallest setup, run this in a dedicated interactive terminal:

```bash
./launch/agg.sh
```

Keep the terminal open. The launcher starts the frontend and worker, and
exits when either process stops. Watch the output for startup failures and
wait for the model to register before sending requests.

Run launchers as executables, not with `source`. The simple shell launchers'
exit cleanup signals their process group, so do not run them in a process
group shared with unrelated jobs.

## Send a local request

In a second terminal, send a small request to the frontend:

```bash
curl --fail --silent --show-error http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fake-model","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

`fake-model` is the launchers' default served name. If you change
`SERVED_NAME`, use the same value in the request. The cluster launcher instead
reads `served_model_name` from YAML.

A successful response exercises the real Dynamo request path through the
simulator. Its generated text has no semantic meaning. One request is a
smoke check, not a throughput, cache-hit, or hardware-performance benchmark.

## Configure the simulation

Stop the current launcher before starting another mode. Arguments to the
simple launchers are forwarded to their workers. For example:

```bash
./launch/disagg.sh \
  --fake-kv-bandwidth-gb-s 1 \
  --fake-itl-ms 25 \
  --fake-kv-generation-ms-per-1k-tokens 50
```

This changes simulated KV-transfer bandwidth, decode timing, and additional
uncached-prefill K/V-generation cost. It does not change the host's hardware.
See [simulation knobs](../README.md#simulation-knobs) for defaults, capacity,
context-length constraints, and other supported flags.

The simple launchers also accept environment overrides:

| Variable | Effect |
|---|---|
| `VENV` | Virtualenv containing the Python executable; defaults to `../.venv` |
| `MODEL` | Model metadata path; defaults to the bundled Qwen3-0.6B directory |
| `SERVED_NAME` | API model name; defaults to `fake-model` |
| `BOOTSTRAP_PORT` | Prefill bootstrap port for `disagg.sh`; defaults to 18998 |
| `PAGE_SIZE` | KV block size for `agg_kv_router.sh`; defaults to 16 |
| `KV_PORT_A`, `KV_PORT_B` | KV-event base ports for the two KV-router workers |

Changing Python code or launch arguments requires restarting the affected
processes. Restarting workers discards their in-memory caches.

## YAML-defined clusters

The repository includes `launch/clusters/two-zones.yaml` and
`launch/clusters/mixed.yaml`. Inspect a launch plan without starting any
frontend or worker:

```bash
./launch/cluster.sh launch/clusters/two-zones.yaml --dry-run
```

The dry run prints worker placement, assigned ports, simulated network links,
and commands. It does not check whether those ports are currently available;
the launcher performs that check when starting a real run.

After reviewing the plan, start it with per-process logs:

```bash
./launch/cluster.sh launch/clusters/two-zones.yaml --log-dir .local/native-logs
```

This log directory is not ignored by the repository's `.gitignore`. Keep it
untracked or choose a private output directory outside the checkout.

The cluster launcher creates a private discovery registry for each run unless
`DYN_FILE_KV` is explicitly set. `--port-offset N` shifts its assigned ports
when you intentionally run another cluster alongside it. Leave
`DYN_FILE_KV` unset to retain per-run discovery isolation.

Worker arguments follow `--`. For example, inspect a packed-rank configuration:

```bash
./launch/cluster.sh launch/clusters/two-zones.yaml --dry-run -- \
  --dp-size 2 --enable-dp-attention --max-running-requests 16
```

This assigns two independent simulated ranks per worker process, with eight
running-request slots per rank. Remove `--dry-run` only when ready to launch.
The cluster launcher reserves a separate range of KV-event ports for each
process's ranks. See [packed workers](../README.md#packed-workers) for capacity,
cache isolation, and routing semantics.

Telemetry support is described in
[observed cache and forward-pass telemetry](../README.md#observed-cache-and-forward-pass-telemetry).
Engine metrics do not imply a provisioned Prometheus or Grafana service.
Forward-pass metrics require Dynamo's injected IPC settings; enabling a flag
alone is not a complete metrics deployment.

## Source-contract checks

After activating the virtualenv, install this repository's declared development
dependencies and check compatibility with the sibling Dynamo checkout:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_shim_contract.py
```

The source-dependent checks skip if Dynamo's SGLang source directory cannot
be found. A skipped check is not evidence of compatibility. If using a
non-sibling checkout, set `DYNAMO_SGLANG_PATH` to its
`components/src/dynamo/sglang` directory. Resolve failures against the selected
Dynamo revision rather than suppressing them.

The full repository suite is available with `python -m pytest tests/`.
No historical pass count guarantees that a different Dynamo revision works.

## Troubleshooting and shutdown

| Symptom | Check |
|---|---|
| Python executable not found | The bootstrap completed and the launcher's `VENV` points to its environment |
| Missing Dynamo or SGLang API | Bootstrap verification and source-contract checks against the selected Dynamo checkout |
| Port already in use | Another launcher or service may own it; identify the owner rather than killing unrelated processes |
| Model not available | Worker startup output, bundled metadata path, and requested served model name |
| Unexpected workers discovered | Simple launchers share default discovery state; use the cluster launcher's private registry for isolated runs |
| No dashboard available | These native launchers do not provision a dashboard or metrics-storage service |

Press **Ctrl-C in the launcher terminal** to stop the local stack. The
launchers handle their own process cleanup; no VM, Kubernetes cluster, or
deployment image needs to be removed. If a launcher was forcibly terminated,
inspect any remaining processes and their ownership before stopping them.

Keep local logs and experiment artifacts out of commits. Do not switch or
update the source checkout while its workers are running.
