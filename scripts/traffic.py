#!/usr/bin/env python3
"""Bounded OpenAI streaming traffic against the local simulated-inference lab."""

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def read_stream(response, started):
    first_content = None
    chunks = 0
    usage = None
    finished = False
    for line in response:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            if not finished:
                raise ValueError("stream ended without a finish reason")
            if first_content is None:
                raise ValueError("stream produced no content")
            if not usage or not isinstance(usage.get("completion_tokens"), int):
                raise ValueError("stream omitted requested token usage")
            return {
                "latency_ms": (time.perf_counter() - started) * 1000,
                "ttft_ms": first_content,
                "content_chunks": chunks,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
            }
        event = json.loads(data)
        if "error" in event:
            raise ValueError(f"server stream error: {event['error']}")
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("delta", {}).get("content"):
                if first_content is None:
                    first_content = (time.perf_counter() - started) * 1000
                chunks += 1
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("length", "stop"):
                    raise ValueError(f"unexpected finish reason: {reason}")
                finished = True
    raise ValueError("connection closed before [DONE]")


def request(index, args, epoch):
    if args.rate:
        time.sleep(max(0, epoch + index / args.rate - time.perf_counter()))
    prompt = (
        f"Group {index % args.prefix_groups}: "
        + "simulation " * args.prompt_words
        + f"\nRequest {index}"
    )
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        result = read_stream(response, started)
    return {"request": index, **result}


def percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {
        "p50": round(statistics.median(ordered), 2),
        "p95": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="fake-model")
    parser.add_argument("--requests", type=positive_int, default=32)
    parser.add_argument("--concurrency", type=positive_int, default=8)
    parser.add_argument("--prompt-words", type=positive_int, default=512)
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--prefix-groups", type=positive_int, default=4)
    parser.add_argument("--rate", type=positive_float, help="target offered requests/s")
    parser.add_argument("--timeout", type=positive_float, default=120)
    args = parser.parse_args()
    results, failed = [], 0
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(request, i, args, started): i for i in range(args.requests)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                result = future.result()
            except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
                failed += 1
                print(json.dumps({"request": index, "error": str(exc)}), file=sys.stderr)
            else:
                results.append(result)
                print(json.dumps(result), flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "summary": True,
        "simulation": True,
        "succeeded": len(results),
        "failed": failed,
        "elapsed_seconds": round(elapsed, 2),
        "requests_per_second": round(len(results) / elapsed, 2),
        "latency_ms": percentiles([r["latency_ms"] for r in results]),
        "client_ttft_ms": percentiles([r["ttft_ms"] for r in results]),
        "completion_tokens": sum(r["completion_tokens"] for r in results),
    }
    print(json.dumps(summary), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
