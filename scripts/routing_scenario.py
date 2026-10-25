#!/usr/bin/env python3
"""Deterministic metadata-on/off experiment using real Dynamo and fake SGLang.

Runs inside the lab image. Each case gets fresh processes and file discovery;
NATS carries genuine worker KV events. No Planner or other traffic participates.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

MODEL = "fake-model"
MODEL_PATH = "/opt/fake-sglang/models/qwen3-0.6b"
PREFIX = "sglang:simulated_prefix_cache_"
PATTERN = (0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0)
CACHE_FIELDS = ("capacity_tokens", "resident_tokens", "allocated_tokens", "evictable_tokens",
                "evicted_tokens_total", "cleared_tokens_total")


def token_ids(family, question, prefix_blocks, block_size):
    prefix = [1000 + family * 4096 + i for i in range(prefix_blocks * block_size)]
    suffix = [40000 + question * block_size + i for i in range(block_size)]
    if max(prefix + suffix) >= 151936:
        raise ValueError("Corpus token IDs exceed the pinned model vocabulary")
    return prefix + suffix


def corpus(prefix_blocks, block_size, requests):
    return [
        {"id": i, "family": PATTERN[i % len(PATTERN)],
         "tokens": token_ids(PATTERN[i % len(PATTERN)], i + 2, prefix_blocks, block_size)}
        for i in range(requests)
    ]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def metric_samples(port):
    from prometheus_client.parser import text_string_to_metric_families

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as response:
        text = response.read().decode()
    return [sample for family in text_string_to_metric_families(text) for sample in family.samples]


def metric_total(samples, name, **labels):
    return sum(s.value for s in samples
               if s.name == name and all(s.labels.get(k) == v for k, v in labels.items()))


def prefill_events(event_type):
    # Dynamo shares process-level indexer counters across prefill/decode routers;
    # their component label is not the source worker's component. Only prefill
    # publishes cache changes in this isolated scenario.
    return metric_total(
        metric_samples(8000), "dynamo_component_kv_cache_events_applied_total",
        event_type=event_type, status="ok",
    )


def worker_snapshot(port):
    samples = metric_samples(port)
    queries = [s for s in samples if s.name == PREFIX + "query_tokens_total"]
    if not queries:
        return None
    if len(queries) != 1:
        raise RuntimeError(f"Expected one DP rank on worker port {port}, got {queries}")
    capacity = metric_total(samples, "dynamo_component_total_blocks")
    if capacity <= 0:
        # The SGLang callback can appear before Dynamo's scheduler-metric relay.
        return None
    cache = {}
    for field in CACHE_FIELDS:
        values = [s.value for s in samples if s.name == PREFIX + field]
        if not values:
            return None
        if len(values) != 1:
            raise RuntimeError(f"Expected one {field} value on worker port {port}")
        cache[field] = values[0]
    if not cache["evictable_tokens"] <= cache["resident_tokens"] <= cache["allocated_tokens"] <= cache["capacity_tokens"]:
        # Shared-memory gauges are collected individually; retry a scrape that
        # overlaps a pool mutation rather than record a torn state.
        return None
    return {
        **cache, "observed_at_ms": time.time_ns() // 1_000_000,
        "worker_id": int(queries[0].labels["worker_id"], 16),
        "queries": metric_total(samples, PREFIX + "hit_ratio_count"),
        "input_tokens": queries[0].value,
        "hit_tokens": metric_total(samples, PREFIX + "hit_tokens_total"),
        "kv_seconds": metric_total(samples, "sglang:simulated_kv_generation_seconds_total"),
        "capacity_blocks": capacity,
        "occupancy": metric_total(samples, "dynamo_component_gpu_cache_usage_percent"),
        "generate_requests": metric_total(samples, "dynamo_component_requests_total", dynamo_endpoint="generate"),
    }


def wait_until(callback, description, processes=(), timeout=60):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise RuntimeError(f"Child exited with {process.returncode} while waiting for {description}")
        try:
            last = callback()
            if last:
                return last
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = repr(exc)
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {description}; last result: {last}")


def send(tokens, *, pinned=None, probe=False, max_tokens=8):
    nvext = {"token_data": tokens, "extra_fields": ["worker_id", "timing"]}
    if pinned is not None:
        nvext["prefill_worker_id"] = pinned
    if probe:
        nvext["annotations"] = ["query_instance_id:"]
    payload = {
        "model": MODEL, "messages": [{"role": "user", "content": "KV routing experiment"}],
        "nvext": nvext, "temperature": 0, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    started_at_ms = time.time_ns() // 1_000_000
    started = time.perf_counter()
    first = None
    usage = None
    workers = None
    echoed = None
    finish = None
    done = False
    text = []
    with urllib.request.urlopen(request, timeout=30) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"Inference failed: {event['error']}")
            extension = event.get("nvext") or {}
            workers = extension.get("worker_id", workers)
            echoed = extension.get("token_ids", echoed)
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    text.append(content)
                    if first is None:
                        first = (time.perf_counter() - started) * 1000
                finish = choice.get("finish_reason") or finish
    if not done or not workers:
        raise RuntimeError(f"Incomplete response: done={done}, worker IDs={workers}")
    if probe:
        if echoed != tokens:
            raise AssertionError("Router preview did not preserve the exact token-ID query")
    elif (first is None or finish != "length" or not usage
          or usage["prompt_tokens"] != len(tokens) or usage["completion_tokens"] != max_tokens):
        raise AssertionError(f"Invalid inference result: first={first}, finish={finish}, usage={usage}")
    return {
        "started_at_ms": started_at_ms, "finished_at_ms": time.time_ns() // 1_000_000,
        "prefill_worker_id": workers["prefill_worker_id"],
        "decode_worker_id": workers["decode_worker_id"],
        "ttft_ms": first, "latency_ms": (time.perf_counter() - started) * 1000,
        "output_sha256": fingerprint("".join(text)),
    }


@contextmanager
def stack(args, metadata, name):
    processes = []
    logs = []
    with tempfile.TemporaryDirectory(prefix="fake-routing-") as directory:
        env = os.environ.copy()
        # Inherit only the lab event transport, not another worker's identity/config.
        for key in list(env):
            if key.startswith(("DYN_", "PROMETHEUS_", "OTEL_")):
                env.pop(key)
        namespace = "routing-" + uuid.uuid4().hex
        env.update({
            "DYN_FILE_KV": directory + "/discovery", "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "nats", "PYTHONHASHSEED": "0",
        })
        router_flags = ["--router-mode", "kv", "--router-temperature", "0", "--router-kv-events"]

        def launch(label, command, port):
            log = open(Path(directory) / (label + ".log"), "w+")
            logs.append((label, log))
            process = subprocess.Popen(
                command, env={**env, "DYN_SYSTEM_PORT": str(port)},
                stdout=log, stderr=subprocess.STDOUT,
            )
            processes.append(process)

        try:
            for index, role in enumerate(("prefill", "prefill", "decode")):
                command = [
                    sys.executable, "-m", "dynamo.sglang", "--model-path", MODEL_PATH,
                    "--served-model-name", MODEL, "--skip-tokenizer-init",
                    "--namespace", namespace, "--discovery-backend", "file",
                    "--host", "0.0.0.0", "--disaggregation-mode", role,
                    "--disaggregation-bootstrap-port", str(8998 + index),
                    "--page-size", str(args.block_size),
                    "--fake-num-kv-blocks", str(args.kv_blocks),
                    "--fake-kv-generation-ms-per-1k-tokens", str(args.kv_generation_ms),
                    "--enable-metrics", *router_flags,
                ]
                if metadata:
                    command += ["--kv-events-config", json.dumps({
                        "publisher": "zmq", "endpoint": f"tcp://*:{5557 + index}",
                    })]
                launch(f"{role}-{index}", command, 9201 + index)
            launch("frontend", [
                sys.executable, "-m", "dynamo.frontend", "--http-port", "8000",
                "--namespace", namespace, "--discovery-backend", "file", *router_flags,
            ], 9200)
            workers = [
                wait_until(lambda p=p: worker_snapshot(p), f"worker {p} metrics", processes)
                for p in (9201, 9202, 9203)
            ]
            for worker in workers:
                if (worker["capacity_blocks"] != args.kv_blocks
                        or worker["capacity_tokens"] != args.kv_blocks * args.block_size):
                    raise AssertionError(f"Advertised KV capacity differs from configured size: {worker}")
            yield workers, processes
        finally:
            for process in reversed(processes):
                if process.poll() is None:
                    process.terminate()
            for process in reversed(processes):
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for label, log in logs:
                if sys.exc_info()[0] is not None:
                    log.seek(0)
                    print(f"{name}/{label}:\n" + log.read(), file=sys.stderr)
                log.close()


def replay_queries(args, queries, measured, name):
    started_at_ms = time.time_ns() // 1_000_000
    clock = time.monotonic()
    for index, query in enumerate(queries):
        if args.duration_seconds:
            time.sleep(max(0, clock + index - time.monotonic()))
        measured(query, query["family"], "replay", expect_full_prefix=True)
        if args.duration_seconds and (index + 1) % 60 == 0:
            print(f"{name}: {index + 1}/{len(queries)} replay requests", file=sys.stderr, flush=True)
    if args.duration_seconds:
        time.sleep(max(0, clock + args.duration_seconds - time.monotonic()))
    return started_at_ms, time.time_ns() // 1_000_000


def run_case(args, metadata, iteration):
    started_at_ms = time.time_ns() // 1_000_000
    name = f"{iteration}-metadata-{'on' if metadata else 'off'}"
    print(f"Starting {name}", file=sys.stderr, flush=True)
    queries = corpus(args.prefix_blocks, args.block_size, args.requests if args.rate == 1 else 2)
    with stack(args, metadata, name) as (workers, processes):
        ids = [w["worker_id"] for w in workers]
        aliases = {wid: index for index, wid in enumerate(ids)}
        seeds = [token_ids(i, i, args.prefix_blocks, args.block_size) for i in (0, 1)]
        # In disaggregated mode the HTTP preview selects only the decode worker.
        # Use it for readiness/token identity, not as a prefill-routing oracle.
        preview = wait_until(
            lambda: send(seeds[0], probe=True), "frontend decode discovery", processes,
        )
        if preview["decode_worker_id"] != ids[2]:
            raise AssertionError(f"Preview selected an unexpected decode worker: {preview}")
        if sum(worker_snapshot(p)["queries"] for p in (9201, 9202, 9203)) != 0:
            raise AssertionError("Routing previews or health checks executed measured inference")

        cache_samples = []

        def record_cache(stage, snapshots=None):
            snapshots = snapshots if snapshots is not None else [
                wait_until(lambda p=p: worker_snapshot(p), "consistent cache snapshot", processes, timeout=5)
                for p in (9201, 9202, 9203)
            ]
            for index, worker in enumerate(snapshots):
                if worker is None or worker["worker_id"] != ids[index]:
                    raise AssertionError(f"Cache observation lost its worker identity: {snapshots}")
                if (not all(math.isfinite(worker[key]) and worker[key] >= 0 for key in CACHE_FIELDS)
                        or not 0 <= worker["evictable_tokens"] <= worker["resident_tokens"]
                        <= worker["allocated_tokens"] <= worker["capacity_tokens"]
                        or worker["capacity_tokens"] != args.kv_blocks * args.block_size):
                    raise AssertionError(f"Invalid cache accounting: {worker}")
            cache_samples.append({"stage": stage, "workers": snapshots})
            return snapshots

        record_cache("initial")

        def publish_barrier(previous):
            if metadata:
                wait_until(
                    lambda: prefill_events("stored") > previous,
                    "prefill cache store applied by the real router", processes, timeout=15,
                )

        for index in (0, 1):
            stored = prefill_events("stored")
            result = send(seeds[index], pinned=ids[index])
            if result["prefill_worker_id"] != ids[index]:
                raise AssertionError("Pinned cache seeding did not reach the requested worker")
            publish_barrier(stored)
            record_cache("seed")

        samples = []

        def measured(query, expected_owner, stage, expect_full_prefix=False):
            before = [worker_snapshot(p) for p in (9201, 9202, 9203)]
            stored = prefill_events("stored")
            front_before = metric_samples(8000)
            result = send(query["tokens"])
            after = [worker_snapshot(p) for p in (9201, 9202, 9203)]
            front_after = metric_samples(8000)
            chosen = aliases[result["prefill_worker_id"]]
            if chosen not in (0, 1) or result["decode_worker_id"] != ids[2]:
                raise AssertionError(f"Not a genuine prefill/decode route: {result}")
            delta = {key: after[chosen][key] - before[chosen][key]
                     for key in ("queries", "input_tokens", "hit_tokens", "kv_seconds")}
            if delta["queries"] != 1 or delta["input_tokens"] != len(query["tokens"]):
                raise AssertionError(f"Engine observations do not match the exact query: {delta}")
            if any(after[i]["queries"] != before[i]["queries"] for i in range(3) if i != chosen):
                raise AssertionError("A request was observed by an unexpected worker")
            expected_seconds = (delta["input_tokens"] - delta["hit_tokens"]) * args.kv_generation_ms / 1e6
            if not math.isclose(delta["kv_seconds"], expected_seconds, abs_tol=1e-8):
                raise AssertionError(f"K/V generation was not charged only to cache misses: {delta}")
            if any(not 0 <= w["occupancy"] <= 1 for w in after):
                raise AssertionError(f"Worker exceeded configured KV capacity: {after}")
            if metadata and expect_full_prefix:
                if chosen != expected_owner or delta["hit_tokens"] != args.prefix_blocks * args.block_size:
                    raise AssertionError(f"Cache-aware routing missed the known best worker: {chosen}, {delta}")
            router_sum = "dynamo_component_router_kv_hit_rate_sum"
            router_count = "dynamo_component_router_kv_hit_rate_count"
            predicted_sum = metric_total(front_after, router_sum) - metric_total(front_before, router_sum)
            predicted_count = metric_total(front_after, router_count) - metric_total(front_before, router_count)
            if predicted_count <= 0:
                raise AssertionError("No real router hit-rate observations for this request")
            if not metadata and predicted_sum != 0:
                raise AssertionError("Metadata-off baseline learned cache state; comparison is contaminated")
            row = {
                **result, **delta, "id": query["id"], "stage": stage,
                "token_sha256": fingerprint(query["tokens"]),
                "selected_worker": chosen, "seeded_cache_owner": expected_owner,
                "selected_seeded_owner": chosen == expected_owner,
                "router_predicted_ratio": predicted_sum / predicted_count,
            }
            samples.append(row)
            record_cache(stage, after)
            publish_barrier(stored)
            return row

        load = None
        if args.rate > 1:
            from routing_load import run_load
            for family in (0, 1):
                measured({"id": f"validation-{family}", "tokens": token_ids(
                    family, 2 + family, args.prefix_blocks, args.block_size,
                )}, family, "validation", expect_full_prefix=True)
            load = run_load(args, ids, record_cache, send, token_ids)
            replay_started_at_ms, replay_finished_at_ms = load["started_at_ms"], load["finished_at_ms"]
        else:
            replay_started_at_ms, replay_finished_at_ms = replay_queries(args, queries, measured, name)

        # Evict A's family on worker 0, then seed that family on worker 1.
        # The subsequent *unpinned* request must follow the cache's new location.
        removed_before = prefill_events("removed")
        pressure_before = record_cache("before-pressure")
        for family in (2, 3):
            stored = prefill_events("stored")
            send(token_ids(family, 100 + family, args.prefix_blocks, args.block_size), pinned=ids[0])
            publish_barrier(stored)
            pressure_after = record_cache("pressure")
        if pressure_after[0]["evicted_tokens_total"] <= pressure_before[0]["evicted_tokens_total"]:
            raise AssertionError("Pressure workload did not evict any engine cache tokens")
        stored = prefill_events("stored")
        send(token_ids(0, 110, args.prefix_blocks, args.block_size), pinned=ids[1])
        publish_barrier(stored)
        record_cache("relocation")
        eviction_query = {"id": "after-eviction", "tokens": token_ids(
            0, 111, args.prefix_blocks, args.block_size,
        )}
        eviction = measured(eviction_query, 1, "eviction", expect_full_prefix=True)
        removed_after = prefill_events("removed")
        if metadata and removed_after <= removed_before:
            raise AssertionError("Cache eviction did not reach the Dynamo router as a removal event")
        return {
            "started_at_ms": started_at_ms, "finished_at_ms": time.time_ns() // 1_000_000,
            "name": name, "metadata_enabled": metadata, "corpus_sha256": fingerprint(queries),
            "workers": workers, "samples": samples, "eviction_check": eviction,
            "cache_samples": cache_samples,
            "replay_started_at_ms": replay_started_at_ms, "replay_finished_at_ms": replay_finished_at_ms,
            "removal_events": removed_after - removed_before,
            **({"load": load} if load is not None else {}),
        }


def summarize(cases):
    result = {}
    for enabled, name in ((True, "metadata_on"), (False, "metadata_off")):
        rows = [r for c in cases if c["metadata_enabled"] == enabled
                for r in c["samples"] if r["stage"] == "replay"]
        if not rows:
            raise ValueError(f"No replay samples for {name}")
        latency = sorted(r["ttft_ms"] for r in rows)
        result[name] = {
            "requests": len(rows),
            "observed_hit_ratio": sum(r["hit_tokens"] for r in rows) / sum(r["input_tokens"] for r in rows),
            "seeded_owner_fraction": sum(r["selected_seeded_owner"] for r in rows) / len(rows),
            "kv_generation_seconds": sum(r["kv_seconds"] for r in rows),
            "ttft_p50_ms": statistics.median(latency),
            "ttft_p95_ms": latency[math.ceil(0.95 * len(latency)) - 1],
            "mean_latency_ms": statistics.mean(r["latency_ms"] for r in rows),
        }
    on, off = result["metadata_on"], result["metadata_off"]
    result["change"] = {
        "hit_ratio_percentage_points": 100 * (on["observed_hit_ratio"] - off["observed_hit_ratio"]),
        "kv_generation_seconds_saved": off["kv_generation_seconds"] - on["kv_generation_seconds"],
        "ttft_p50_percent": 100 * (on["ttft_p50_ms"] / off["ttft_p50_ms"] - 1),
        "ttft_p95_percent": 100 * (on["ttft_p95_ms"] / off["ttft_p95_ms"] - 1),
        "mean_latency_percent": 100 * (on["mean_latency_ms"] / off["mean_latency_ms"] - 1),
    }
    return result


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--kv-blocks", type=int, default=96)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefix-blocks", type=int, default=64)
    parser.add_argument("--kv-generation-ms", type=float, default=250)
    workload = parser.add_mutually_exclusive_group()
    workload.add_argument("--requests", type=int)
    workload.add_argument("--duration-seconds", type=int,
                          help="Replay duration per case (1-600 seconds), at one deterministic request/second")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--rate", type=int, default=1,
                        help="Offered requests/second (1-100); rates above 1 use concurrent aggregate measurements")
    parser.add_argument("--concurrency", type=int, default=256,
                        help="Maximum in-flight client requests; excess arrivals are explicitly counted as client rejection")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(arguments)
    if args.duration_seconds is not None:
        if not 1 <= args.duration_seconds <= 600:
            parser.error("duration-seconds must be between 1 and 600 per case")
        args.requests = args.duration_seconds
    elif args.requests is None:
        args.requests = 24
    if (not 1 <= args.rate <= 100 or not 1 <= args.concurrency <= 512
            or (args.rate > 1 and args.duration_seconds is None)):
        parser.error("Require rate 1-100, concurrency 1-512, and duration-seconds for concurrent load")
    if not (args.block_size in (8, 16, 32) and 0 < args.prefix_blocks <= 64
            and args.prefix_blocks + 2 < args.kv_blocks < 2 * args.prefix_blocks
            and 0 < args.requests <= 600 and 0 < args.pairs <= 4
            and math.isfinite(args.kv_generation_ms) and 0 < args.kv_generation_ms <= 2000):
        parser.error("Require block size 8/16/32, a positive bounded workload, and one-prefix < capacity < two-prefixes")
    return args


def main():
    args = parse_args()
    started_at_ms = time.time_ns() // 1_000_000
    cases = []
    for pair in range(args.pairs):
        for enabled in ((True, False) if pair % 2 == 0 else (False, True)):
            cases.append(run_case(args, enabled, len(cases)))
    if args.rate > 1:
        from routing_load import summarize_load
        summary = summarize_load(cases)
    else:
        if len({c["corpus_sha256"] for c in cases}) != 1:
            raise AssertionError("A/B cases did not use identical queries")
        for index in range(args.requests):
            if len({c["samples"][index]["output_sha256"] for c in cases}) != 1:
                raise AssertionError(f"Output changed between A/B cases for query {index}")
        summary = summarize(cases)
    report = {
        "schema_version": 3 if args.rate > 1 else 2, "run_id": uuid.uuid4().hex,
        "started_at_ms": started_at_ms, "finished_at_ms": time.time_ns() // 1_000_000,
        "simulation": True, "configuration": vars(args), "summary": summary,
        "cases": cases,
        "method": "Fresh fixed 2-prefill/1-decode processes per case; identical token IDs; only seed/pressure requests pinned; event-driven KV router in both cases; no predicted-cache fallback.",
        "limitations": (
            "Simulated K/V costs, no tensor allocation or GPU compute. Inputs are seeded and identical across cases; "
            "worker IDs, concurrent ordering, ties and wall-clock latency are not deterministic. "
            "Concurrent mode measures one-second aggregates and engine counters, not per-request counter deltas. "
            "Client concurrency rejection is separate from server/network failures. A synthetic locality model, "
            "not a production trace or a calibrated GPU throughput benchmark."
        ),
    }
    print("ROUTING_REPORT " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
