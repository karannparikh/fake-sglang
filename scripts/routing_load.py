"""Bounded, open-loop cache-pressure traffic with deterministic locality shifts."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import random
import statistics
import sys
import threading
import time
import urllib.error


def load_query(index, args):
    rng = random.Random(args.seed + index)
    hot_start = ((index // (args.rate * 120)) % 8) * 4
    hot = list(range(hot_start, hot_start + 4))
    family = rng.choice(hot if rng.randrange(10) < 8 else [i for i in range(32) if i not in hot])
    return family, index % 4096 + 2, (16, 64, 16, 64)[family % 4]


def percentiles(values):
    if not values:
        return None
    values = sorted(values)
    return {"p50": statistics.median(values), "p95": values[math.ceil(len(values) * .95) - 1],
            "p99": values[math.ceil(len(values) * .99) - 1]}


class LoadMeasurements:
    def __init__(self, started_at_ms):
        self.started_at_ms = started_at_ms
        self.windows = {}
        self.latencies = []
        self.ttfts = []
        self.errors = Counter()
        self.error_examples = []

    def window(self, timestamp):
        index = max(0, (timestamp - self.started_at_ms) // 1000)
        return self.windows.setdefault(index, {
            "offered": 0, "sent": 0, "succeeded": 0, "failed": 0, "client_rejected": 0,
            "latencies": [], "ttfts": [], "schedule_lag": [],
        })

    def completed(self, result, offered_at_ms):
        sent = self.window(result["started_at_ms"])
        sent["sent"] += 1
        sent["schedule_lag"].append(max(0, result["started_at_ms"] - offered_at_ms))
        done = self.window(result["finished_at_ms"])
        if "error" in result:
            done["failed"] += 1
            self.errors[result["error_type"]] += 1
            if len(self.error_examples) < 10:
                self.error_examples.append(result)
        else:
            done["succeeded"] += 1
            done["latencies"].append(result["latency_ms"])
            done["ttfts"].append(result["ttft_ms"])
            self.latencies.append(result["latency_ms"])
            self.ttfts.append(result["ttft_ms"])

    def finish(self, finished_at_ms):
        end = max(finished_at_ms, self.started_at_ms + 1,
                  self.started_at_ms + max(self.windows, default=0) * 1000 + 1)
        count = math.ceil((end - self.started_at_ms) / 1000)
        inflight = 0
        rows = []
        for index in range(count):
            window = self.window(self.started_at_ms + index * 1000)
            inflight += window["sent"] - window["succeeded"] - window["failed"]
            rows.append({
                **{key: window[key] for key in ("offered", "sent", "succeeded", "failed", "client_rejected")},
                "started_at_ms": self.started_at_ms + index * 1000,
                "finished_at_ms": min(end, self.started_at_ms + (index + 1) * 1000),
                "inflight": inflight, "e2e_ms": percentiles(window["latencies"]),
                "ttft_ms": percentiles(window["ttfts"]), "schedule_lag_ms": percentiles(window["schedule_lag"]),
            })
        totals = {key: sum(row[key] for row in rows)
                  for key in ("offered", "sent", "succeeded", "failed", "client_rejected")}
        if (totals["offered"] != totals["sent"] + totals["client_rejected"]
                or totals["sent"] != totals["succeeded"] + totals["failed"]):
            raise AssertionError(f"Load accounting did not conserve requests: {totals}")
        return {
            "started_at_ms": self.started_at_ms, "finished_at_ms": end,
            "windows": rows, "totals": totals,
            "e2e_ms": percentiles(self.latencies), "ttft_ms": percentiles(self.ttfts),
            "errors": dict(self.errors), "error_examples": self.error_examples,
            "_latencies": self.latencies, "_ttfts": self.ttfts,
        }


def run_load(args, ids, record_cache, send, token_ids):
    # Metrics sampling must not block the open-loop request scheduler.
    record_cache("load-baseline")
    stop = threading.Event()
    sampling_errors = []

    def sample():
        while not stop.wait(1):
            try:
                record_cache("load")
            except (urllib.error.URLError, OSError, RuntimeError, AssertionError) as error:
                sampling_errors.append(str(error))
                return

    sampler = threading.Thread(target=sample, name="routing-load-metrics", daemon=True)
    sampler.start()
    started_at_ms = time.time_ns() // 1_000_000
    epoch = time.monotonic()
    measurements = LoadMeasurements(started_at_ms)
    pending = {}
    plan = hashlib.sha256()
    expected_outputs = {}
    mismatched_outputs = 0

    def request(query):
        family, question, blocks = query
        started = time.time_ns() // 1_000_000
        try:
            result = send(token_ids(family, question, blocks, args.block_size))
            if result["prefill_worker_id"] not in ids[:2] or result["decode_worker_id"] != ids[2]:
                raise AssertionError(f"Response has unknown worker IDs: {result}")
            return result
        except (urllib.error.URLError, OSError, RuntimeError, ValueError, KeyError, AssertionError) as error:
            return {
                "started_at_ms": started, "finished_at_ms": time.time_ns() // 1_000_000,
                "error_type": type(error).__name__, "error": str(error)[:1000],
            }

    def collect():
        nonlocal mismatched_outputs
        for future in [f for f in pending if f.done()]:
            offered, query = pending.pop(future)
            result = future.result()
            measurements.completed(result, offered)
            if "error" not in result:
                key = ":".join(map(str, query))
                previous = expected_outputs.setdefault(key, result["output_sha256"])
                mismatched_outputs += previous != result["output_sha256"]

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for index in range(args.duration_seconds * args.rate):
                scheduled = epoch + index / args.rate
                while time.monotonic() < scheduled:
                    collect()
                    time.sleep(min(.005, max(0, scheduled - time.monotonic())))
                collect()
                if sampling_errors:
                    raise RuntimeError(f"Cache sampling failed during load: {sampling_errors}")
                offered = started_at_ms + index * 1000 // args.rate
                measurements.window(offered)["offered"] += 1
                query = load_query(index, args)
                plan.update(json.dumps(query, separators=(",", ":")).encode() + b"\n")
                if len(pending) >= args.concurrency:
                    measurements.window(offered)["client_rejected"] += 1
                else:
                    pending[pool.submit(request, query)] = offered, query
                if (index + 1) % (args.rate * 60) == 0:
                    rejected = sum(w["client_rejected"] for w in measurements.windows.values())
                    failed = sum(w["failed"] for w in measurements.windows.values())
                    print(f"Load: offered={index + 1}, completed={len(measurements.latencies)}, "
                          f"failed={failed}, client_rejected={rejected}, inflight={len(pending)}",
                          file=sys.stderr, flush=True)
            time.sleep(max(0, epoch + args.duration_seconds - time.monotonic()))
            while pending:
                collect()
                if pending:
                    time.sleep(.01)
    finally:
        stop.set()
        sampler.join(timeout=20)
    if sampler.is_alive() or sampling_errors:
        raise RuntimeError(f"Cache sampler did not complete cleanly: {sampling_errors}")
    finished_at_ms = time.time_ns() // 1_000_000
    record_cache("load-final")
    return {
        **measurements.finish(finished_at_ms),
        "offered_duration_seconds": args.duration_seconds, "target_rps": args.rate,
        "concurrency_limit": args.concurrency, "plan_sha256": plan.hexdigest(),
        "output_hashes": expected_outputs, "output_mismatches": mismatched_outputs,
        "profile": "80% four hot families / 20% remaining 28; hot set rotates every 120s; 256/1024-token prefixes",
    }


def summarize_load(cases):
    summary = {}
    expected_outputs = {}
    for case in cases:
        load = case["load"]
        if load["output_mismatches"]:
            raise AssertionError("Identical load queries produced different outputs within a case")
        for key, digest in load.pop("output_hashes").items():
            if expected_outputs.setdefault(key, digest) != digest:
                raise AssertionError("Identical load queries produced different outputs across cases")
    if len({case["load"]["plan_sha256"] for case in cases}) != 1:
        raise AssertionError("Concurrent A/B cases did not use identical offered queries")
    for enabled, name in ((True, "metadata_on"), (False, "metadata_off")):
        selected = [case for case in cases if case["metadata_enabled"] == enabled]
        totals = Counter()
        latencies, ttfts = [], []
        input_tokens = hit_tokens = evicted_tokens = 0
        elapsed = offered_seconds = 0
        for case in selected:
            load = case["load"]
            totals.update(load["totals"])
            latencies.extend(load.pop("_latencies"))
            ttfts.extend(load.pop("_ttfts"))
            elapsed += (load["finished_at_ms"] - load["started_at_ms"]) / 1000
            offered_seconds += load["offered_duration_seconds"]
            first = next(s for s in case["cache_samples"] if s["stage"] == "load-baseline")["workers"]
            last = next(s for s in case["cache_samples"] if s["stage"] == "load-final")["workers"]
            for before, after in zip(first, last):
                input_tokens += after["input_tokens"] - before["input_tokens"]
                hit_tokens += after["hit_tokens"] - before["hit_tokens"]
                evicted_tokens += after["evicted_tokens_total"] - before["evicted_tokens_total"]
        summary[name] = {
            **dict(totals), "completed_rps_including_drain": totals["succeeded"] / elapsed,
            "offered_rps": totals["offered"] / offered_seconds,
            "observed_hit_ratio": hit_tokens / input_tokens if input_tokens else None,
            "evicted_tokens": evicted_tokens, "e2e_ms": percentiles(latencies), "ttft_ms": percentiles(ttfts),
        }
    return summary
