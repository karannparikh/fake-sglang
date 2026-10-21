# fake-sglang

A CPU-only **fake SGLang engine** that simulates prefill, decode, and KV-cache
transfer, so you can develop and test a distributed inference framework on
[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) + SGLang without GPUs.

It works by shipping an importable `sglang` package that implements the subset
of SGLang's Python API that Dynamo's SGLang backend consumes. **Dynamo runs
completely unmodified** — `python -m dynamo.sglang` loads this instead of real
SGLang and never knows the difference.

> This is **not** SGLang and does no inference. Output tokens are deterministic
> noise; latencies come from a model, not a forward pass. Every worker prints a
> `SIMULATED engine` banner at startup for exactly this reason.

## Why not something that already exists

| Existing | What it gives | Why it wasn't enough |
|---|---|---|
| `python -m dynamo.mocker` | Dynamo's Rust scheduler simulator: KV blocks, chunked prefill, preemption, bandwidth-modelled transfer | Plugs in as `EngineType.Mocker`. The `dynamo.sglang` component never loads, so none of the SGLang integration is exercised |
| `dynamo.common.backend.sample_engine` | A CPU fake engine on Dynamo's Unified Backend SDK | Same: it bypasses `dynamo.sglang` entirely |
| `sglang/tools/sglang-simulator` | Real SGLang scheduler with a mocked `ModelRunner.forward`, GPU-free in CI | **No PD-disaggregation support at all**, and it still needs the full CUDA-only `sglang` + `torch` install |

## What it simulates

- **Prefill** — chunked against a token budget, cost proportional to tokens
  actually recomputed (prefix-cache hits and transferred KV are free).
- **Decode** — one shared step per batch whose cost grows with batch size, so
  queueing shows up as ITL growth the way it does on a GPU.
- **KV cache** — a finite pool of fixed-size blocks, cumulative-hash prefix
  reuse, and LIFO preemption under pressure.
- **KV transfer** — a **real** cross-process HTTP bootstrap handshake between
  the prefill and decode workers, with the payload cost simulated as
  `tokens x kv_bytes_per_token / bandwidth`. `kv_bytes_per_token` is derived
  from the model's real `config.json`.
- **KV events** — block stores and evictions published over ZMQ in SGLang's
  exact wire format, so Dynamo's KV-aware router builds a real prefix tree and
  routes on cache hits.
- **Scheduler metrics** — pushed over ZMQ to the socket Dynamo binds, so
  load-based routing sees real occupancy.

Aggregated and disaggregated modes produce **byte-identical output** for the
same prompt, and both are reproducible run to run.

## Local Kubernetes lab (Apple Silicon)

The optional local lab runs the released **Dynamo 1.5.0 frontend, SGLang
adapter, operator and Planner unchanged**, with only SGLang's engine replaced
by this CPU simulator.

**Integration dependencies:** this scaffold expects the separate simulator
changes for configurable KV capacity/context and K/V-generation delay,
observed cache metrics, forward-pass metrics (FPM), and packed data-parallel
ranks. Those engine changes are not implemented by the lab files. Rendering
manifests and regenerating the baseline dashboard work independently, but
running this topology on an engine without those changes is not supported.
The traffic and routing-benchmark utilities are a separate follow-up;
`traffic`, `routing-test`, and `routing-dashboard` report an explicit error
until their scripts are present. Benchmark dashboards and their observability
wiring are not provisioned by this scaffold.

```text
OpenAI streaming traffic
  -> Dynamo frontend / KV-aware router
  -> dynamo.sglang prefill -> simulated KV handoff -> dynamo.sglang decode

Kubernetes discovery + worker metadata <-> Dynamo operator / Planner
NATS JetStream <-> Dynamo events, KV events and forward-pass telemetry

Service /metrics -> OpenTelemetry Collector -> Prometheus -> Grafana
Dynamo OTLP traces/logs -> Collector -> Tempo / Loki -> Grafana
```

The dedicated Colima profile and three-node kind cluster are both named
`dynamo-sim`. They use 8 virtual CPUs, 24 GiB RAM and a 100 GiB virtual disk.
Existing Docker/Kubernetes contexts are not activated or overwritten.
Standalone etcd is also deployed and monitored, but active Dynamo discovery
uses Kubernetes: its worker metadata supplies the Planner's KV capacity.
The Colima VM uses the gRPC port forwarder so inference traffic does not
share the SSH forwarding connection.

### Start or resume

Install prerequisites through Homebrew if needed:

```bash
brew install colima docker docker-buildx kind helm kubernetes-cli
```

The sibling `../dynamo` checkout must be NVIDIA's `v1.5.0` release. For a fresh
workspace, clone it with
`git clone --branch v1.5.0 --depth 1 https://github.com/ai-dynamo/dynamo.git`.
Alternatively, set `DYNAMO_SOURCE` to an existing checkout at that release;
the lab verifies the release commit and never changes its source.

After integrating the engine dependencies above, from this repository:

```bash
bash deploy/local/lab.sh up
bash deploy/local/lab.sh status
```

The first run builds a native ARM64 Linux image containing genuine Dynamo
wheels and explicitly CPU-only PyTorch. Subsequent runs reuse it. No weights
or remote model code are downloaded. Image dependencies are pinned in
`deploy/local/requirements.txt`; the separate AIPerf requirements file is
reserved for the benchmark follow-up and is not installed by the lab image.

| Endpoint | URL |
|---|---|
| Inference API | http://127.0.0.1:8000 |
| Metrics dashboard | http://127.0.0.1:3000/d/dynamo-local-sglang |
| Prometheus | http://127.0.0.1:9090 |
| OTLP gRPC | `127.0.0.1:4317` |
| OTLP HTTP | http://127.0.0.1:4318 |

Grafana's username is `dynamo`. Its generated password is in
`.local/grafana-password` (mode 0600). `.local/` is ignored by Git and excluded
from the Docker build context. Host ports bind only to localhost. This is a
development lab, not a production-hardened deployment; the operator metrics
scrape uses bearer authentication but skips verification of its locally
generated certificate.

### Worker topology and cache configuration

The default topology is **40 prefill + 40 decode logical workers**, packed
into four Python processes per role with ten data-parallel (DP) ranks per
process. With the packed-rank engine changes, Dynamo routes to
`(worker_id, dp_rank)`; each rank has its own scheduler, KV pool, cache events,
scheduler load and FPM. Runtime imports and model metadata are shared.

The real Planner receives FPM and evaluates load, but its four-endpoint
minimum per role and eight-slot GPU budget pin the fleet at eight processes.
The budget counts the single simulated GPU scheduling resource per process,
not its ten logical ranks. `optimization_target=load` always enables load
scaling in Dynamo, so disabling its boolean flag alone would not freeze the
fleet. Each worker node advertises ten **simulated Kubernetes GPU resource
slots**; each worker process requests one. These are opaque scheduling
resources, not physical CUDA devices. `up` re-advertises them after restarts.

Edit `deploy/local/lab-config.json`, then run
`bash deploy/local/lab.sh deploy`. These settings apply to each fake prefill
and decode worker; redeployment restarts affected workers and clears caches.

| Setting | Local default | Meaning |
|---|---|---|
| `worker_processes_per_role` | 4 | Kubernetes worker pods per role |
| `ranks_per_process` | 10 | Independent simulated workers per pod |
| `context_length` | 131072 | Simulated maximum sequence length; not a real-model capability claim |
| `kv_blocks_per_worker` | 8192 | Finite KV pool per logical worker |
| `kv_block_size` | 16 | Token slots per block; default capacity is 131,072 tokens |
| `publish_kv_events` | `true` | Publish retained prefixes and evictions to real Dynamo |
| `kv_generation_ms_per_1k_tokens` | 50.0 | Additional modeled K/V-generation cost per 1,000 uncached tokens |

The pool stores block metadata, not GPU KV tensors. Longer requests and more
cached blocks still increase host-memory use; 131,072 slots do not guarantee
that every concurrent request fits. The tokenizer/model configuration remains
Qwen3-0.6B; the larger context is an explicit simulation-only override.

The delay is additive to prefill cost and overhead, divided by
`fake_speedup_ratio`. Reused prefixes and transferred decode KV incur no
K/V-generation cost; recomputation after preemption is charged again.
The engine's standalone default is zero; the lab deliberately sets 50 ms.
Turning event publication off does not disable caching or observed-hit
metrics. Both settings keep event-driven routing enabled rather than
enabling Dynamo's approximate prediction of cache state from previous routes.

`deploy/local/router-policy.yaml` is a renderer template using FCFS, an
8,192-active-token prefill busy threshold, and per-logical-worker queue budgets
of 10 requests and 65,536 raw input tokens. The renderer multiplies these by
`ranks_per_process` because Dynamo scales budgets by process endpoints.
Budgets apply per routing queue/policy class, not globally; executing requests
are excluded and one accepted request can exceed the token budget. Each rank
has eight running scheduler slots. For ten ranks, process-wide admission is
80 engine requests plus 20 shared overflow requests. The frontend also uses
an 85% active-decode-block threshold with output-block tracking. These are
initial bounded lab settings, not production-calibrated limits.

### Metrics, traces and logs

The committed dashboard is a reproducible, offline-generated inference/cache/
Planner overview with a live raw-metric explorer. It contains no captured
host metrics or timestamps. Regenerate it without any running services:

```bash
python3 deploy/local/dashboard.py --offline
python3 deploy/local/render.py infrastructure
python3 deploy/local/render.py observability
python3 deploy/local/render.py dynamo
```

After the lab has collected metrics, expand the persisted catalog from live
Prometheus metadata:

```bash
bash deploy/local/lab.sh dashboard
```

This adds collapsed, type-aware sections for every discovered metric family
while retaining the dashboard UID. Grafana picks up the ConfigMap update
automatically. Source/Instance variables filter the dashboard. Counters show
cumulative values and reset-aware rates; gauges show raw values; histograms
show p50/p95/p99 and mean. No data means no samples or an inactive feature,
not a fabricated zero. Tempo and Loki are also provisioned in Grafana Explore.

Cache panels expect the separate engine metrics changes. Observed prefix-hit
ratios use `sglang:simulated_prefix_cache_hit_tokens_total` divided by
`sglang:simulated_prefix_cache_query_tokens_total`, with pod/DP-rank labels.
Observations count initial prefill admission once, exclude transferred decode
KV and health/warmup probes, and reset on worker restart. Idle windows have no
defined ratio. The per-request histogram mean is not token-weighted, and router
predictions count a different population. The separate backend-reported
cached fraction includes transferred KV and is not cross-request prefix reuse.
Resident/allocated/capacity and pressure-eviction panels use engine counters,
not router event counts; explicit cache clears are shown separately.

The collector scrapes Dynamo pods only after Kubernetes reports them Ready.
This avoids a reproduced startup interaction between synchronous FPM relay
construction (holding Python's GIL) and concurrent Python metrics callbacks.
It is a **lab-side mitigation, not a fix to Dynamo's Rust bindings**; unready
pods do not export scraped metrics through this collector.

### Rebuild and stop

```bash
bash deploy/local/lab.sh rebuild  # test a new image, load it, then roll workers
bash deploy/local/lab.sh deploy   # reapply worker config using the recorded image
bash deploy/local/lab.sh stop     # stop only this Colima VM; preserve its data
```

The kubeconfig is `.local/kubeconfig`; use it explicitly for manual commands:

```bash
KUBECONFIG="$PWD/.local/kubeconfig" kubectl -n dynamo-sim get dgd,pods
```

Prometheus retains 48 hours of metrics (up to 3 GB); Tempo retains 24 hours
of traces. Persistent volumes survive `stop`. The lab does not provide
physical GPU utilization, CUDA/NCCL/RDMA execution, real KV tensors, Grove,
Kai scheduling, snapshots, GPU power management or multi-tier KV offload.
Simulated timings are not calibrated GPU benchmarks.

## Native Python setup (alternative)

Clone this repo next to a Dynamo checkout — the bootstrap script expects
`../dynamo` and creates the shared virtualenv at `../.venv`:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
git clone <this repo> fake-sglang
```

Then one command, ~5 minutes on a clean machine:

```bash
cd fake-sglang
./scripts/bootstrap_macos.sh
```

It installs cmake/protobuf, Rust, `uv`, creates `../.venv` (Python 3.10),
builds Dynamo's PyO3 bindings from source (there is no macOS wheel), installs
`ai-dynamo` and this package, and verifies the result.

**Safe to re-run.** Every step checks before it acts, so a second run reuses
the existing toolchain and virtualenv and reinstalls nothing already
satisfied (~19s, almost all of it the incremental Rust rebuild). The binding
build is repeated on purpose — it is the only way a change under `dynamo/lib`
reaches the wheel, and cargo's cache keeps it at ~15s rather than ~2.5min.

| Override | Effect |
|---|---|
| `SKIP_BINDINGS=1` | Skip the Rust rebuild when `dynamo._core` already imports (~2s total) |
| `FORCE_RECREATE=1` | Delete and rebuild the virtualenv from scratch |
| `PY_VERSION=3.12` | Use a different Python (see the `asyncio.timeout` note under *Known gaps*) |

If `../.venv` exists but is a different Python than expected, the script stops
with an explicit message rather than half-provisioning against it.

Model metadata ships in the repo: `models/qwen3-0.6b/` holds Qwen3-0.6B's
**config + tokenizer only, no weights** (~15 MB instead of ~1.5 GB), pinned to
a Hugging Face revision so every checkout is byte-identical (see
`models/qwen3-0.6b/SOURCE.md`). The launch scripts use it by default. Pointing
`--model-path` at a local directory also makes Dynamo skip its own model
download entirely, so runs are offline and instant.

To use a different model, or to move to a newer Qwen3 revision, fetch its
metadata and point `MODEL` at it:

```bash
source ../.venv/bin/activate
python scripts/fetch_model_metadata.py <hf-repo-id> models/<name> [REVISION]
MODEL=$PWD/models/<name> ./launch/agg.sh
```

`fetch_model_metadata.py` needs `huggingface_hub` and network access. When
bumping the Qwen3 revision, update `QWEN3_0_6B_REVISION` in the script and
`SOURCE.md` alongside the regenerated files.

## Run native processes

No etcd, no NATS, no Docker — the launch scripts use Dynamo's file-based
discovery backend (`DYN_DISCOVERY_BACKEND=file`) and a TCP request plane.

```bash
./launch/agg.sh            # frontend :8000 + one aggregated worker
./launch/disagg.sh         # frontend :8000 + prefill worker + decode worker
./launch/agg_kv_router.sh  # frontend :8000 + 2 workers behind the KV-aware router
```

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"fake-model","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'
```

Any flag is forwarded to the workers, so you can dial the simulation:

```bash
./launch/disagg.sh --fake-kv-bandwidth-gb-s 1 --fake-itl-ms 25
```

## Clusters and network topology

`launch/cluster.sh` runs any mix of workers from a YAML file: how many prefill,
decode and aggregated workers, where each group sits in the network, and what
a KV transfer costs over each kind of link.

```bash
./launch/cluster.sh launch/clusters/two-zones.yaml --dry-run   # print the plan only
./launch/cluster.sh launch/clusters/two-zones.yaml             # run it
./launch/cluster.sh launch/clusters/mixed.yaml -- --fake-itl-ms 20   # args after -- go to every worker
```

```yaml
frontend: {router_mode: kv}          # kv also turns on per-worker KV events
topology:
  levels: [zone, rack, node]         # a location has one component per level
  links:                             # named by the deepest level both ends share
    node:    {bandwidth_gb_s: 450,  latency_ms: 0.01}   # same node
    rack:    {bandwidth_gb_s: 50,   latency_ms: 0.05}   # same rack, other node
    zone:    {bandwidth_gb_s: 12.5, latency_ms: 0.5}    # same zone, other rack
    default: {bandwidth_gb_s: 1.25, latency_ms: 5}      # nothing shared
workers:
  - {role: prefill, count: 2, location: zone-a/rack-1/node-1}
  - {role: decode,  count: 4, location: zone-b/rack-1/node-1, args: [--fake-itl-ms, 12]}
  - {role: agg,     count: 1, location: zone-a/rack-2/node-1}
```

The prefill worker publishes its location with each KV room. The decode worker
looks up the link between the two locations and prices the transfer as
`latency_ms + bytes / bandwidth_gb_s`. It logs every handoff at INFO, so you
can count how far KV traveled in a run:

```
decode-2 | fake-sglang: KV room 5439... zone-a/rack-1/node-1 -> zone-b/rack-1/node-1 over 'default' link (1.25 GB/s): 2011 tokens, 219.95 MiB, 189.51 ms
```

```bash
./launch/cluster.sh launch/clusters/two-zones.yaml --log-dir runs/1
grep -h "KV room" runs/1/decode-*.log | grep -o "over '[a-z]*'" | sort | uniq -c
```

At startup the launcher prints the placement and the link that each
prefill → decode pair would use. Ports are assigned automatically. Use
`--port-offset N` to run a cluster beside another deployment. Each run gets
its own discovery registry (`DYN_FILE_KV`), so concurrent runs never discover
each other's workers.

**What Dynamo does with this today: nothing.** In the current Dynamo,
`PrefillRouter` picks the prefill worker by KV overlap plus load and the
decode worker by load alone. Nothing links the two choices by distance.
Measured on `two-zones.yaml` with 20 requests: 11 handoffs crossed zones,
6 crossed racks within a zone, and 3 stayed on one node. The topology is
there so you can measure that cost, and measure any topology-aware pairing
you add to Dynamo against it.

Aggregated workers serving the same model form their own worker set. The
frontend splits requests between that set and the prefill → decode set
instead of adding the aggregated workers to the decode pool.

## Simulation knobs

| Flag | Default | Effect |
|---|---|---|
| `--fake-prefill-ms-per-1k-tokens` | 55.0 | Prefill cost per 1k recomputed tokens |
| `--fake-prefill-overhead-ms` | 4.0 | Fixed per-prefill-step cost |
| `--fake-itl-ms` | 9.0 | Base inter-token latency |
| `--fake-itl-ms-per-running-req` | 0.35 | ITL added per extra request in the batch |
| `--fake-kv-bandwidth-gb-s` | 64.0 | Simulated KV-transfer bandwidth |
| `--fake-kv-transfer-overhead-ms` | 1.0 | Fixed per-transfer cost |
| `--fake-speedup-ratio` | 1.0 | Divides every simulated sleep (time compression) |
| `--fake-num-kv-blocks` | 8192 | KV pool size; lower it to reach preemption |
| `--fake-bootstrap-timeout-s` | 30.0 | How long decode waits for prefill's KV |
| `--fake-location` | unset | This worker's place in the network, e.g. `zone-a/rack-1/node-2` |
| `--fake-topology` | unset | JSON link table; with locations on both ends it replaces the two `--fake-kv-*` knobs per transfer |

KV event publishing is enabled with SGLang's own flag, not a `--fake-` one:

```bash
--page-size 16 --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:25557"}'
```

Dynamo derives `use_kv_events` from that same config, so one flag turns on both
the engine's publishing and Dynamo's subscription. Give each worker its own
port.

## How disaggregation flows

Dynamo's Rust `PrefillRouter` orchestrates this; the two workers only do the
KV handoff.

```
frontend --> PrefillRouter --> prefill worker   (max_tokens forced to 1)
                                    |
                    yields disaggregated_params {bootstrap_host, port, room}
                    and publishes the room on its bootstrap HTTP server
                                    |
             PrefillRouter stamps it onto the decode leg as bootstrap_info
                                    v
                              decode worker
                    GET /room/{id}  -> Bootstrapping   (real HTTP wait)
                    sleep(payload)  -> Transferring    (simulated bytes)
                                    -> Success
                    decodes from a "prebuilt" batch, leading with the
                    token prefill actually produced
```

`bootstrap_host == "2.2.2.2"` (SGLang's `FAKE_BOOTSTRAP_HOST`) short-circuits
the whole handshake. Dynamo's health-check canary and prefill warmup both use
it; without that path, warmup would block for its 1800 s timeout.

## KV-aware routing

Each worker publishes `BlockStored` / `BlockRemoved` events as its prefix cache
changes. Dynamo's router subscribes, builds a radix tree per worker, and sends
a request to whichever worker already holds the longest matching prefix.

The subtlety worth knowing: Dynamo treats the published `block_hashes` as
**opaque engine ids** and recomputes its own prefix key from the published
`token_ids` (`lib/kv-router/src/zmq_wire/convert.rs`). So the token payload has
to be exactly right — wrong or misaligned tokens produce hashes that silently
never match the frontend's, and routing degrades to round-robin with no error
anywhere. `tests/test_kv_events.py` asserts the published tokens rehash to what
`compute_block_hash_for_seq` gives for the same prompt.

Measured on this stack with two workers (`./launch/agg_kv_router.sh`), a
~6700-token prompt sent twice:

```
pass   overlap_blocks
cold   0
warm   422          <- full prefix hit, routed back to the worker holding it
```

With `--kv-events-config` omitted the same test gives `0` on both passes, which
is what proves the match comes from the published events rather than the
router's own bookkeeping.

> **Known flakiness.** Occasionally the warm pass still reports
> `overlap_blocks=0` on a freshly started stack. The engine is emitting
> correctly in those runs (verified with a raw ZMQ subscriber), so this is a
> delivery or indexer-timing race on the Dynamo side, not in the fake engine.
> A block is published once and never re-announced, so an event lost in that
> window is not recovered. Restarting the stack clears it.

### Per-tenant cache isolation (`cache_salt`)

A request's `cache_salt` names a cache namespace: only requests with the same
salt share prefix-cache blocks, as in SGLang. Clients send it as
`"nvext": {"cache_salt": "tenant-a"}`. The worker seeds its block-hash chain
with the salt and publishes the salt on each `BlockStored` event. Dynamo's
router folds that salt into the prefix key it recomputes from the tokens, so
it builds a separate tree per tenant. An empty salt is no salt.

Measured with two aggregated workers behind the KV router, sending the same
1,500-word prompt under different salts:

```
request         overlap_blocks   worker
tenant-a  1st   0                W1
tenant-a  2nd   94               W1    <- full hit, same tenant
tenant-b  1st   0                W2    <- same prompt, other tenant: no hit
tenant-b  2nd   94               W2
no salt         0                W2    <- matches neither tenant
```

`tests/test_kv_events.py` checks that the published tokens, rehashed under
the published salt with Dynamo's own `compute_block_hash_for_seq`, reproduce
the frontend's salted key, and that this key differs from the unsalted one.

## Layout

```
src/sglang/       the shim: SGLang-shaped facade, no simulation logic
                    (incl. the KV-event wire format, matched to SGLang's)
src/fakeengine/   the simulator: scheduler, KV pool, timing, bootstrap,
                    transfer, KV-event policy
launch/           agg.sh, disagg.sh, agg_kv_router.sh,
                    cluster.sh (+ clusters/*.yaml)
scripts/          bootstrap_macos.sh, fetch_model_metadata.py
models/           pinned model metadata (config + tokenizer, no weights)
tests/            contract tests + simulator behavior + disagg + KV events
                    + topology + cluster launcher
```

The split is the point: `src/sglang/` translates, `src/fakeengine/` behaves.
Swapping in real SGLang, or driving the simulator from a different facade, only
touches one side.

## Memory per worker

A worker's footprint is ~93 MB. Most of what's left is the Python, Dynamo
runtime, and tokenizer imports. Without intervention it would be ~280 MB:
Dynamo's SGLang component imports `torch` at module scope, and `transformers`
pulls in its torch modeling stack once it sees torch installed. The engine
never touches a tensor, so `src/sglang/_no_torch.py` (active only under
`python -m dynamo.sglang`) tells transformers torch is absent and gives
Dynamo a stub `torch`. Any real call into the stub raises an error. Set
`FAKE_SGLANG_REAL_TORCH=1` to load the real torch instead.

## Tests

```bash
pytest tests/                                                    # 85 tests
pytest ../dynamo/components/src/dynamo/sglang/tests -m unit      # 1006 pass, 7 fail (below)
```

Counts are against Dynamo `2db5333ef6`.

`tests/test_shim_contract.py` is the drift guard. It reads Dynamo's own source
and fails by name when Dynamo starts using something the shim lacks: every
`sglang.*` import, every `server_args.<field>` / `parsed_args.<field>`, every
`tokenizer_manager.<attr>` and `engine.<attr>`, every `sgl.Engine.<attr>` read
off the class (private names included, since a class-level read breaks import
of the whole worker), and every `envs.SGLANG_*` field, which must have
SGLang's type. It understands Dynamo's `_compat.py` N/N-1 fallback groups,
requiring only that *one* alternative in each group resolves. Groups the shim
leaves unresolved on purpose are listed in `DELIBERATELY_ABSENT` with a
reason, and the list fails if it goes stale.

After `git pull` in `../dynamo`, re-run `./scripts/bootstrap_macos.sh`. Its
verify step fails, and points you at the contract test, if Dynamo has moved
past the shim.

### Known gaps

7 of Dynamo's own SGLang unit tests fail. None of them affects running workers:

| Tests | Cause |
|---|---|
| `test_gateway_engine_facade_generates_through_tokenizer_manager` | `--gateway-workers` mode is not supported (see `managers/multi_tokenizer_mixin.py`). |
| `test_parse_args_rejects_mla_dcp_on_dsa_backend` | Follows from the deliberately absent MLA backend prediction (`DELIBERATELY_ABSENT`). |
| `test_external_mm_hashes_rebuild_padding_after_hash_override`, `test_sglang_video_output_is_vp9_in_shipped_image` | Multimodal is not simulated; the video test also needs `imageio`. |
| `test_cancelled_sync_engine_route_keeps_engine_routes_serialized`, `test_disagg_decode_passes_prefill_dp_rank_to_engine` | Python 3.10: `asyncio.timeout` is 3.11+, and `inspect.signature(AsyncMock())` raises on 3.10, so Dynamo drops optional kwargs for mocked engines. The bootstrap pins 3.10 because `ai-dynamo`'s `aisimulate` dependency is 3.11+ and ships only an sdist. Build on 3.12 if you need these. |
| `test_metrics_fanout_reaches_sibling_gateways` | macOS: pytest's temp dir makes the `ipc://` path longer than the 103-character Unix socket limit. Passes on Linux. |

## Not simulated yet

- **Forward-pass metrics (FPM).** The wire schema is implemented and matched to
  Dynamo's (its contract test passes), but the engine publishes no FPM stream;
  scheduler metrics go over the simpler `metrics_ipc_name` socket instead.
- **Multi-rank KV events.** One DP rank per worker. The per-rank port offset is
  implemented, but nothing drives `dp_size > 1`.
- **Gateway mode (`--gateway-workers`)**: see Known gaps above.
- Multimodal, embedding/rerank, diffusion, LoRA, and weight updates — these
  raise a clear `NotImplementedError` rather than pretending.
