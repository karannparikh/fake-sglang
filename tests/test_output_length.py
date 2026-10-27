"""Natural output length: answers end on their own before max_new_tokens."""

from __future__ import annotations

import asyncio
import math
import statistics

import pytest

from fakeengine import tokens as token_gen
from fakeengine.config import SimConfig
from fakeengine.scheduler import Request, Scheduler
from sglang import Engine

PROMPT = list(range(64))


def _config(server_args, **overrides) -> SimConfig:
    return SimConfig.from_server_args(server_args(**overrides))


async def _run(scheduler: Scheduler, request: Request) -> tuple:
    scheduler.submit(request)
    tokens = []
    while True:
        kind, payload = await request.queue.get()
        if kind == "token":
            tokens.append(payload)
        else:
            return tokens, payload


async def _collect(stream) -> tuple:
    tokens, meta = [], None
    async for chunk in stream:
        tokens.extend(chunk["output_ids"])
        meta = chunk["meta_info"]
    return tokens, meta


def test_natural_length_is_deterministic_per_prompt():
    assert token_gen.natural_length(PROMPT, 150, 1.5) == token_gen.natural_length(
        PROMPT, 150, 1.5
    )


def test_natural_length_follows_the_configured_lognormal():
    lengths = [token_gen.natural_length([i, i + 1], 150, 1.47) for i in range(4000)]
    assert statistics.median(lengths) == pytest.approx(150, rel=0.1)
    # Mean of a lognormal is median * exp(sigma^2 / 2): about 442 here.
    assert statistics.mean(lengths) == pytest.approx(150 * math.exp(1.47**2 / 2), rel=0.15)
    assert min(lengths) >= 1


def test_zero_sigma_gives_exactly_the_median():
    assert token_gen.natural_length(PROMPT, 7, 0.0) == 7


def test_output_length_is_off_by_default(server_args):
    assert _config(server_args).output_len_median is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"fake_output_len_median": 0.5},
        {"fake_output_len_median": float("nan")},
        {"fake_output_len_sigma": -1.0},
        {"fake_output_len_sigma": float("inf")},
    ],
)
def test_invalid_output_length_settings_fail_at_startup(server_args, overrides):
    with pytest.raises(ValueError):
        _config(server_args, **overrides)


async def test_unset_median_runs_to_max_new_tokens(server_args):
    scheduler = Scheduler(_config(server_args))
    scheduler.start()
    tokens, reason = await _run(scheduler, Request("r", PROMPT, 20))
    assert (len(tokens), reason) == (20, "length")
    await scheduler.stop()


async def test_request_stops_at_its_natural_length(server_args):
    config = _config(server_args, fake_output_len_median=4, fake_output_len_sigma=0.0)
    scheduler = Scheduler(config)
    scheduler.start()
    tokens, reason = await _run(scheduler, Request("r", PROMPT, 20))
    assert reason == "stop"
    assert tokens == token_gen.take(PROMPT, config.vocab_size, 4)
    await scheduler.stop()


async def test_smaller_budget_still_reports_length(server_args):
    scheduler = Scheduler(
        _config(server_args, fake_output_len_median=10, fake_output_len_sigma=0.0)
    )
    scheduler.start()
    tokens, reason = await _run(scheduler, Request("r", PROMPT, 3))
    assert (len(tokens), reason) == (3, "length")
    await scheduler.stop()


async def test_budget_ending_on_the_natural_token_reports_length(server_args):
    scheduler = Scheduler(
        _config(server_args, fake_output_len_median=5, fake_output_len_sigma=0.0)
    )
    scheduler.start()
    tokens, reason = await _run(scheduler, Request("r", PROMPT, 5))
    assert (len(tokens), reason) == (5, "length")
    await scheduler.stop()


async def test_single_token_answer_stops_after_the_first_token(server_args):
    scheduler = Scheduler(
        _config(server_args, fake_output_len_median=1, fake_output_len_sigma=0.0)
    )
    scheduler.start()
    tokens, reason = await _run(scheduler, Request("r", PROMPT, 20))
    assert (len(tokens), reason) == (1, "stop")
    await scheduler.stop()


async def test_ignore_eos_runs_to_max_new_tokens(server_args):
    scheduler = Scheduler(
        _config(server_args, fake_output_len_median=4, fake_output_len_sigma=0.0)
    )
    scheduler.start()
    tokens, reason = await _run(scheduler, Request("r", PROMPT, 20, ignore_eos=True))
    assert (len(tokens), reason) == (20, "length")
    await scheduler.stop()


async def test_min_new_tokens_delays_the_natural_end(server_args):
    scheduler = Scheduler(
        _config(server_args, fake_output_len_median=4, fake_output_len_sigma=0.0)
    )
    scheduler.start()
    tokens, reason = await _run(scheduler, Request("r", PROMPT, 20, min_new_tokens=9))
    assert (len(tokens), reason) == (9, "stop")
    await scheduler.stop()


async def test_preemption_keeps_the_natural_length(server_args):
    scheduler = Scheduler(_config(
        server_args, fake_num_kv_blocks=64, max_running_requests=8,
        fake_output_len_median=6, fake_output_len_sigma=0.0,
    ))
    scheduler.start()
    requests = [Request(f"r{i}", list(range(i, i + 8)), 30) for i in range(6)]
    results = await asyncio.gather(*(_run(scheduler, r) for r in requests))
    assert [(len(tokens), reason) for tokens, reason in results] == [(6, "stop")] * 6
    await scheduler.stop()


async def test_disaggregated_answer_stops_where_the_aggregated_one_does(
    server_args, unused_port
):
    knobs = dict(fake_output_len_median=40, fake_output_len_sigma=1.0)
    aggregated = Engine(server_args=server_args(**knobs))
    prefill = Engine(server_args=server_args(
        disaggregation_mode="prefill", disaggregation_bootstrap_port=unused_port, **knobs
    ))
    decode = Engine(server_args=server_args(disaggregation_mode="decode", **knobs))
    try:
        stream = await aggregated.async_generate(
            input_ids=PROMPT, sampling_params={"max_new_tokens": 5000}, stream=True
        )
        agg_tokens, agg_meta = await _collect(stream)

        bootstrap = dict(
            bootstrap_host="127.0.0.1", bootstrap_port=unused_port, bootstrap_room=7
        )
        stream = await prefill.async_generate(
            input_ids=PROMPT, sampling_params={"max_new_tokens": 1}, stream=True,
            **bootstrap,
        )
        await _collect(stream)
        stream = await decode.async_generate(
            input_ids=PROMPT, sampling_params={"max_new_tokens": 5000}, stream=True,
            **bootstrap,
        )
        disagg_tokens, disagg_meta = await _collect(stream)

        assert agg_meta["finish_reason"] == {"type": "stop", "matched": None}
        assert disagg_meta["finish_reason"] == agg_meta["finish_reason"]
        assert disagg_tokens == agg_tokens
        assert len(agg_tokens) == token_gen.natural_length(PROMPT, 40, 1.0)
    finally:
        for engine in (aggregated, prefill, decode):
            await engine._async_shutdown()


async def test_engine_forwards_ignore_eos(server_args):
    engine = Engine(server_args=server_args(
        fake_output_len_median=4, fake_output_len_sigma=0.0
    ))
    try:
        stream = await engine.async_generate(
            input_ids=PROMPT,
            sampling_params={"max_new_tokens": 12, "ignore_eos": True},
            stream=True,
        )
        tokens, meta = await _collect(stream)
        assert (len(tokens), meta["finish_reason"]["type"]) == (12, "length")
    finally:
        await engine._async_shutdown()
