# Local Dynamo + fake-SGLang lab

An operating guide for an Apple Silicon macOS lab using genuine Dynamo 1.5.0
with a CPU-only simulated SGLang engine. No model weights are needed.
Simulation timings and logical GPU slots are not measurements of GPU hardware.

## Prerequisite status

**This is documentation for an assembled lab, not a working installation path
from this documentation branch alone.** On October 22, 2026, this fork's `main`
was `dec638758fbf775ad96014276cfb94fa5d27078b`. Lifecycle and KV-configuration
changes were merged; [telemetry #4](https://github.com/karannparikh/fake-sglang/pull/4)
and [packed ranks #5](https://github.com/karannparikh/fake-sglang/pull/5) were open.
The lab scaffold and benchmark scripts listed below were absent from `main`.
Recheck their availability before following the operational commands.

Use an isolated source checkout containing all of these prerequisites:

- Lifecycle cleanup, configurable KV capacity/context and K/V-generation delay,
  observed per-rank cache metrics, forward-pass metrics (FPM), and packed DP ranks.
- `deploy/local/lab.sh`, `kind.yaml`, `lab-config.json`, `render.py`, the
  observability manifests and dashboards, and pinned image requirements.
- `deploy/local/Dockerfile` including `COPY launch ./launch`, and the complete
  test suite. The image build and source-contract tests must pass together.
- `scripts/mooncake_lab.py`, `scripts/mooncake_report.py`,
  `scripts/packed_probe.py`, `scripts/traffic.py`, and
  `deploy/local/aiperf-requirements.txt`. The optional routing experiment also
  needs `deploy/local/routing_compare.py` and its dashboard utilities.

Do not assume merging only the engine changes supplies the lab scripts, or
that a lab-only branch supplies the engine changes. This guide adds neither.
Paths below are relative to the root of that complete source checkout.
Do not switch branches or update a checkout that owns an active lab.

## Choose the operation

| Goal | Operation | Effect |
|---|---|---|
| Open dashboards or inspect health | Read-only checks below | No deployment or traffic |
| Replay the six-case matrix | Mooncake runner | Routing changes, cold resets and traffic |
| Install or resume infrastructure | `lab.sh up` | Provisioning, deployment and observability restarts |
| Change worker configuration | `lab.sh deploy` | Redeploys the recorded image |
| Change image source | `lab.sh rebuild` | Builds, tests, loads and deploys a new image |
| Stop the lab | `lab.sh stop` | Stops the dedicated VM |

**Do not run `up`, `deploy`, or `rebuild` merely to open Grafana.** Do not
run probes, extra traffic, deployments or another experiment during a measured
benchmark. Keep the host awake, powered and free of unrelated heavy load.

## Architecture and intended configuration

```text
AIPerf on macOS
  -> one Dynamo frontend: chat templating, tokenization and routing
  -> four prefill processes x ten independent DP ranks
  -> simulated KV handoff
  -> four decode processes x ten independent DP ranks
  -> frontend detokenization and streaming response

Worker metrics -> OpenTelemetry Collector -> Prometheus -> Grafana
```

Workers use `python -m dynamo.sglang`; only the engine is replaced. FPM uses
Dynamo's real local ZMQ integration. Packed-rank cache metrics use the
multiprocess Prometheus path.

| Setting | Intended lab configuration |
|---|---|
| Frontend / Planner | One replica each |
| Processes | Four prefill and four decode |
| Independent DP ranks | Ten per process: 40 prefill + 40 decode |
| KV capacity per rank | 8,192 blocks x 16 tokens = 131,072 token slots |
| DCP size | One, distinct from DP rank count |
| Running-request slots | Eight per rank, 80 per process |
| Context length | 131,072; simulation override, not a model capability claim |
| Prefill cost | 55 ms per 1,000 uncached tokens plus 4 ms per step |
| Additional K/V-generation cost | 50 ms per 1,000 uncached tokens |
| Decode cost | 9 ms plus 0.35 ms per additional running request |
| Tokenizer | Qwen3-0.6B configuration/tokenizer only |

Ranks share process resources but have independent schedulers and caches.
Ten DP ranks do not split a request into ten pieces. KV-aware routing addresses
`(worker_id, dp_rank)`; power-of-two chooses process endpoints with local rank
fallback, so the policies are not equivalent rank-selection mechanisms.

## Connect without redeploying

From the checkout that owns the running lab:

```bash
export LAB_KUBECONFIG="$PWD/.local/kubeconfig"
colima status --profile dynamo-sim
docker --context colima-dynamo-sim ps
kubectl --kubeconfig "$LAB_KUBECONFIG" --context kind-dynamo-sim \
  -n dynamo-sim get dgd,pods
kubectl --kubeconfig "$LAB_KUBECONFIG" --context kind-dynamo-sim \
  -n observability get pods
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models
curl --fail --silent --show-error http://127.0.0.1:3000/api/health
curl --fail --silent --show-error http://127.0.0.1:9090/-/ready
```

Always supply both the kubeconfig and `--context kind-dynamo-sim` for manual
kubectl commands. Do not change the default context: it may target another
cluster. For the intended topology, expect four Ready prefill pods, four Ready
decode pods, one frontend and one Planner; resets temporarily replace pods.

Check images before any deployment:

```bash
cat .local/image
kubectl --kubeconfig "$LAB_KUBECONFIG" --context kind-dynamo-sim \
  -n dynamo-sim get pods \
  -o 'custom-columns=NAME:.metadata.name,IMAGE:.spec.containers[0].image'
```

**`.local/image` is authoritative for `deploy` and image reuse in `up`.**
A manual image rollout does not update it. A stale marker can revert that
rollout; exporting `DYNAMO_SIM_IMAGE` alone does not override `read_image()`.
Resolve discrepancies before deploying, rather than copying a historical tag.

| Service | Local URL |
|---|---|
| Main lab dashboard | http://127.0.0.1:3000/d/dynamo-local-sglang |
| Prometheus | http://127.0.0.1:9090 |
| Inference API | http://127.0.0.1:8000 |
| Separate synthetic routing dashboard | http://127.0.0.1:3000/d/dynamo-routing-live |

Use the main dashboard for Mooncake runs. The isolated routing dashboard can
correctly show no data during a Mooncake replay. Select the current run's time
window; history depends on retained Prometheus data.

Grafana's username is `dynamo`. Provisioning stores its generated password in
`.local/grafana-password` (mode 0600) and the `observability/grafana-admin`
Secret. Use the current lab's credential locally; never paste it into reports
or commits. On macOS, `pbcopy < .local/grafana-password` avoids terminal output.
If credentials were changed separately, consult the current Secret instead
of assuming the file is current.

The kind configuration maps NodePorts to loopback host ports, with Colima
forwarding to macOS. Manual kubectl port-forwarding is not normally required.
Identify any existing listener before creating a duplicate forward.

## Fresh installation

Only proceed after assembling the source prerequisites and obtaining permission
to create or modify this dedicated lab. Install approved tooling through your
normal package-management process:

```bash
brew install colima docker docker-buildx kind helm kubernetes-cli
git clone --branch v1.5.0 --depth 1 \
  https://github.com/ai-dynamo/dynamo.git dynamo-v1.5.0
export DYNAMO_SOURCE="$PWD/dynamo-v1.5.0"
```

Use a new clone destination; do not repurpose another working Dynamo checkout.
The launcher requires release commit
`b83b1d9304ebfc624709ac46db32b1b6f1ff1615`.
From the complete fake-SGLang source root:

```bash
bash deploy/local/lab.sh up
bash deploy/local/lab.sh status
```

`up` creates or reuses the `dynamo-sim` Colima profile (eight vCPUs, 24 GiB RAM,
100 GiB disk), uses Docker context `colima-dynamo-sim`, and creates or reuses
a three-node kind cluster. It advertises simulated GPU slots, applies
infrastructure, **restarts observability deployments**, installs the operator,
builds or reuses an image, and deploys the graph. State is checkout-local under
`.local/`; this is not a harmless health check.

The build runs the full image test suite before loading the image into kind
or recording it. A fresh `up` is **not guaranteed to succeed** merely because
an existing image runs. Check `tests/test_shim_contract.py` against the pinned
Dynamo version: `test_deliberately_absent_list_is_current` checks guarded
imports, while `test_dynamo_sees_cache_salt_support` expects
`dynamo.sglang._compat.cache_salt_kwargs`. These are source-version contracts,
not a portable historical pass/fail count. Reconcile any failure with the
assembled source and pinned release; do not suppress tests or delete a working
lab to bypass the gate.

## Prepare the benchmark inputs

Use a supported host Python such as Python 3.13, not macOS system Python 3.9.
The host replay environment is separate from the image:

```bash
umask 077
mkdir -p .local/mooncake-100rps
python3.13 -m venv .local/aiperf-venv
.local/aiperf-venv/bin/python -m pip install \
  -r deploy/local/aiperf-requirements.txt
HF_HOME="$PWD/.local/aiperf-hf" .local/aiperf-venv/bin/python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
)
PY
```

The requirements pin AIPerf 0.13.0, Transformers 4.57.1 and huggingface-hub
0.36.0. Cache the tokenizer before a run; the runner enables offline mode.
It reads inputs from its own checkout's `.local/`, not from arbitrary
`LAB_KUBECONFIG` or output-directory overrides.

Obtain the complete Mooncake FAST'25 traces through an approved source,
record provenance, and place them at:

```text
.local/mooncake-100rps/conversation_trace.jsonl
.local/mooncake-100rps/toolagent_trace.jsonl
```

For this six-case recipe, verify the exact inputs:

| Trace | JSONL requests | SHA-256 |
|---|---:|---|
| Conversation | 12,031 | `b8cbb061a85206d729d91cdc2981f43c9e0d99209dce588d3af5f7934408b9df` |
| ToolAgent | 23,608 | `48a2db1a13d3bc05e6330140c64f604ba366df20d3c9e128b5c35a01c1fa5f71` |

```bash
shasum -a 256 .local/mooncake-100rps/conversation_trace.jsonl \
  .local/mooncake-100rps/toolagent_trace.jsonl
wc -l .local/mooncake-100rps/conversation_trace.jsonl \
  .local/mooncake-100rps/toolagent_trace.jsonl
```

Stop if either differs. The runner records the input checksum and derives
request count from the file, but does not enforce this table. Samples or
different traces are different experiments. No trace data is bundled here.

## Run the six-case comparison

**This mutates the running lab.** The runner temporarily changes routing
arguments and cold-resets frontend/prefill/decode pods before every case.
Reserve exclusive use of the lab first.

```bash
umask 077
export LAB_RUN="$PWD/.local/mooncake-100rps/comparison-$(date +%Y%m%d-%H%M%S)"
python3.13 scripts/mooncake_lab.py --request-rate 10 \
  --traces conversation toolagent \
  --modes kv load-aware power-of-two \
  --output "$LAB_RUN"
```

Keep these options explicit: the runner defaults to 50 RPS and omits
power-of-two. The six cases run sequentially, with each full trace replayed
once per routing mode: **106,917 requests**, about three hours of offered
traffic plus initialization, reset and drain time.

AIPerf uses streaming chat, seed 42, four client workers, a 120-second request
timeout and server token counts. `ignore_eos: true` is a root-level request
field, not nested under sampling parameters. The constant 10-RPS arrival
schedule replaces original trace timestamps. Client workers are not Dynamo
frontend replicas; offered rate is not necessarily achieved dispatch rate.

The stock runner saves original routing arguments and attempts restoration
in `finally`; it does not restore an entire deployment snapshot. Uncatchable
kills, default SIGTERM handling or VM loss cannot guarantee restoration.
After interruption inspect remaining AIPerf processes, current routing
arguments, `failure.json` and `restoration-failure.json` before another run.
Absence of an error file is not proof of successful restoration. Do not
blindly apply an old full deployment snapshot or kill unrelated processes.

### Progress and reports

Use the same `LAB_RUN` value in another terminal:

```bash
find "$LAB_RUN" -maxdepth 2 \( -name invocation.json -o -name exit.json \)
```

Each case has a `console.log`; later case directories do not exist until
started. Missing `exit.json` means the case has not recorded completion,
not necessarily that it failed.

Generate the report separately after the runner finishes:

```bash
python3.13 scripts/mooncake_report.py "$LAB_RUN"
```

| Artifact | Meaning |
|---|---|
| `initial-deployment.json` | Pre-run graph, including original routing arguments |
| `<case>/invocation.json` | Exact AIPerf command, offered rate, input checksum and request count |
| `<case>/deployment.json` | Case configuration |
| `<case>/initial-pods.json`, `final-pods.json` | Pod identities and restart counters |
| `<case>/aiperf/` | Raw request records and AIPerf summary |
| `<case>/cpu.jsonl` | Kubelet CPU and memory samples |
| `<case>/cache-metrics.json` | End-of-case cache counters |
| `<case>/summary.json`, `summary.json` | Generated per-case and aggregate reports |

The stock runner/report do not produce `completion.json` or a Markdown final
report. Report generation discovers available case artifacts, so check that
all six case names are present. For each case require `is_complete`, and
`expected`, `recorded`, `unique_conversations` and `completed` equal to its
full trace count. Completed requests can include errors: also inspect
`successful`, error types/rates, dispatch RPS, TTFT, latency and output-length
mismatches.

CPU/memory observations describe actual CPU containers, not simulated GPU
utilization. Inspect coverage and counter validity before interpreting
averages. Missing rank metrics are **not zero cache hits**: whole-fleet cache
fractions need complete rank coverage and stable pod identities. The cache
scope is admitted prefill requests, not all offered traffic. Do not add
per-role memory peaks from different times and call the sum a simultaneous
fleet peak.

## Other traffic and maintenance

Run only outside measured experiments, with the intended topology:

```bash
bash deploy/local/lab.sh traffic --requests 32 --concurrency 8
python3.13 scripts/packed_probe.py \
  --targets-per-role 40 --output .local/packed-probe.json
```

The probe checks rank addressing, capacity and cache isolation and changes
cache contents. `bash deploy/local/lab.sh routing-test` is a different,
isolated synthetic routing experiment, not the full Mooncake matrix; its live
dashboard is `dynamo-routing-live`.

Edit `deploy/local/lab-config.json` only in the intended complete checkout:
it controls process/rank counts, context, per-rank KV blocks/page size,
KV-event publication and generation cost.

```bash
bash deploy/local/lab.sh deploy
bash deploy/local/lab.sh rebuild
bash deploy/local/lab.sh dashboard
```

Choose one operation deliberately: `deploy` renders configuration with the
recorded image and does not rebuild Python changes; `rebuild` builds and tests
a new image before deploying. Both can replace workers and clear caches.
`dashboard` regenerates and applies dashboard definitions from live metric
metadata. None is needed to simply view Grafana.

## Troubleshooting and shutdown

| Symptom | Check |
|---|---|
| No dashboard data | Main vs isolated routing dashboard, selected time window, collector and Prometheus readiness |
| Login fails | Username `dynamo`, current Secret rather than a static password |
| Connection refused | Dedicated Colima profile, kind nodes, service readiness and existing port mappings |
| Worker replacement | Deliberate between-case cold reset vs unexpected within-case restart |
| Startup stalls | Preserve ready-only OTel scraping of Dynamo pods |
| Incomplete cache/CPU coverage | Inspect collection errors, rank counts and pod stability; do not substitute zeros |
| Source-contract tests fail | Reconcile the assembled source with pinned Dynamo before deployment |

Ready-only OTel scraping avoids scraping Python metrics callbacks during FPM
relay initialization. It is a local startup mitigation, **not a fix for an
upstream Dynamo deadlock**; unready pods will not supply scraped metrics.
Check Tempo/Loki independently before relying on traces/logs: healthy
Prometheus/Grafana does not establish trace-storage health.

When no experiment is running:

```bash
bash deploy/local/lab.sh stop
```

This stops the dedicated Colima VM and preserves its data. Do not delete the
cluster or `.local/` for normal shutdown. Resume may require re-advertising
simulated GPU slots and reconciling services; `up` does this but also reapplies
deployments, so review the image marker first.

Keep `.local/`, credentials, traces, raw logs, deployment snapshots and
recorded benchmark artifacts private and out of commits.
