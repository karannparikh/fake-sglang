from unittest.mock import AsyncMock

import pytest

from fakeengine.config import SimConfig
from fakeengine.kvcache import BlockPool, block_hashes
from fakeengine.scheduler import Request, Scheduler
from fakeengine.timing import TimingModel
from fakeengine.topology import Link
from sglang.srt.entrypoints.engine import _SchedulerInitResult
from sglang.srt.server_args import ServerArgs, prepare_server_args


@pytest.mark.parametrize("field", [
    "fake_num_kv_blocks", "page_size", "dcp_size", "max_total_tokens",
])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_cache_sizes_fail(server_args, field, value):
    with pytest.raises(ValueError, match=field):
        SimConfig.from_server_args(server_args(**{field: value}))


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), float("-inf")])
def test_invalid_generation_delay_fails(server_args, value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SimConfig.from_server_args(server_args(
            fake_kv_generation_ms_per_1k_tokens=value,
        ))


@pytest.mark.parametrize("capacity", [8, 17])
def test_token_capacity_requires_whole_blocks(server_args, capacity):
    with pytest.raises(ValueError, match="multiple of page_size"):
        SimConfig.from_server_args(server_args(
            page_size=8, dcp_size=2, max_total_tokens=capacity,
        ))


def test_token_capacity_overrides_block_count_and_is_advertised(server_args):
    config = SimConfig.from_server_args(server_args(
        fake_num_kv_blocks=500, max_total_tokens=128, page_size=8, dcp_size=2,
    ))
    assert config.num_kv_blocks == 8
    assert config.block_size == 16
    assert config.total_kv_tokens == 128
    assert _SchedulerInitResult(config).scheduler_infos[0]["max_total_num_tokens"] == 128
    assert Scheduler(config).pool.num_blocks == 8


def test_block_capacity_and_zero_generation_cost_remain_defaults(server_args):
    config = SimConfig.from_server_args(server_args(
        fake_num_kv_blocks=32, page_size=8, dcp_size=2,
    ))
    assert config.total_kv_tokens == 512
    assert config.kv_generation_ms_per_1k == 0
    assert TimingModel(config).prefill_ms(1000) == (
        config.prefill_overhead_ms + config.prefill_ms_per_1k
    )


def test_kv_return_cli_flag_defaults_off():
    disabled = prepare_server_args(["--model-path", "stub-model"])
    enabled = prepare_server_args(["--model-path", "stub-model", "--enable-kv-return"])
    assert not disabled.enable_kv_return
    assert enabled.enable_kv_return


def test_kv_return_defaults_fall_back_to_forward_settings(server_args):
    config = SimConfig.from_server_args(server_args(
        fake_kv_bandwidth_gb_s=23.5,
        fake_kv_transfer_overhead_ms=2.5,
        fake_bootstrap_timeout_s=4.25,
    ))
    assert not config.enable_kv_return
    assert config.kv_return_protocol_version == 1
    assert config.kv_return_max_tokens == 4096
    assert config.kv_return_deadline_ms == 4250
    assert config.kv_return_bytes_per_token == config.kv_bytes_per_token
    assert config.kv_return_bandwidth_gb_s == 23.5
    assert config.kv_return_overhead_ms == 2.5


def test_disabled_kv_return_preserves_zero_forward_transfer_overhead(server_args):
    config = SimConfig.from_server_args(server_args(
        enable_kv_return=False,
        fake_kv_transfer_overhead_ms=0,
    ))
    assert config.kv_transfer_overhead_ms == 0
    assert config.kv_return_overhead_ms == 0


def test_kv_return_cli_overrides_are_independent():
    args = prepare_server_args([
        "--model-path", "stub-model",
        "--enable-kv-return",
        "--fake-kv-return-max-tokens", "257",
        "--fake-kv-return-deadline-ms", "1234.5",
        "--fake-kv-return-bytes-per-token", "8192",
        "--fake-kv-return-bandwidth-gb-s", "12.5",
        "--fake-kv-return-overhead-ms", "0",
    ])
    config = SimConfig.from_server_args(args)
    assert config.enable_kv_return
    assert config.kv_return_max_tokens == 257
    assert config.kv_return_deadline_ms == 1234.5
    assert config.kv_return_bytes_per_token == 8192
    assert config.kv_return_bandwidth_gb_s == 12.5
    assert config.kv_return_overhead_ms == 0


@pytest.mark.parametrize("field", [
    "fake_kv_return_max_tokens",
    "fake_kv_return_bytes_per_token",
])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_kv_return_integer_settings_fail(server_args, field, value):
    with pytest.raises(ValueError, match=field):
        SimConfig.from_server_args(server_args(**{field: value}))


@pytest.mark.parametrize("field", [
    "fake_kv_return_deadline_ms",
    "fake_kv_return_bandwidth_gb_s",
])
@pytest.mark.parametrize(
    "value",
    [0, -1, True, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_kv_return_float_settings_fail(server_args, field, value):
    with pytest.raises(ValueError, match=field):
        SimConfig.from_server_args(server_args(**{field: value}))


@pytest.mark.parametrize(
    "value",
    [-1, True, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_kv_return_overhead_fails(server_args, value):
    with pytest.raises(ValueError, match="fake_kv_return_overhead_ms"):
        SimConfig.from_server_args(server_args(
            fake_kv_return_overhead_ms=value,
        ))


@pytest.mark.parametrize("value", [0, float("nan"), float("inf")])
def test_invalid_kv_bandwidth_fallback_source_fails(server_args, value):
    with pytest.raises(ValueError, match="fake_kv_bandwidth_gb_s"):
        SimConfig.from_server_args(server_args(fake_kv_bandwidth_gb_s=value))


@pytest.mark.parametrize(
    "value",
    [-1, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_bootstrap_timeout_fails(server_args, value):
    with pytest.raises(ValueError, match="fake_bootstrap_timeout_s"):
        SimConfig.from_server_args(server_args(fake_bootstrap_timeout_s=value))


@pytest.mark.parametrize("overrides", [
    {"enable_kv_return": False},
    {"enable_kv_return": False, "disaggregation_mode": "null"},
    {"enable_kv_return": False, "disable_radix_cache": True},
    {"enable_kv_return": True, "disable_radix_cache": True},
])
def test_zero_bootstrap_timeout_accepted_when_return_ineligible(server_args, overrides):
    config = SimConfig.from_server_args(server_args(
        fake_bootstrap_timeout_s=0, **overrides,
    ))
    assert config.bootstrap_timeout_s == 0
    assert config.kv_return_deadline_ms == 0


def test_zero_bootstrap_timeout_with_positive_return_override_enables_return(server_args):
    config = SimConfig.from_server_args(server_args(
        enable_kv_return=True,
        disaggregation_mode="prefill",
        fake_bootstrap_timeout_s=0,
        fake_kv_return_deadline_ms=1234.5,
    ))
    assert config.enable_kv_return
    assert config.bootstrap_timeout_s == 0
    assert config.kv_return_deadline_ms == 1234.5


def test_enabled_return_with_zero_effective_deadline_fails(server_args):
    with pytest.raises(ValueError, match="positive reverse-return deadline"):
        SimConfig.from_server_args(server_args(
            enable_kv_return=True,
            disaggregation_mode="prefill",
            fake_bootstrap_timeout_s=0,
        ))


@pytest.mark.parametrize(
    "value",
    [-1, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_forward_transfer_overhead_fails(server_args, value):
    with pytest.raises(ValueError, match="fake_kv_transfer_overhead_ms"):
        SimConfig.from_server_args(server_args(
            fake_kv_transfer_overhead_ms=value,
        ))


def test_default_worker_profile_matches_h200_experiment():
    args = ServerArgs(model_path="stub-model", served_model_name="stub-model")
    config = SimConfig.from_server_args(args)

    assert config.block_size == 256
    assert config.num_kv_blocks == 15_509
    assert config.total_kv_tokens == 3_970_304
    assert config.context_len == 1_048_576
    assert config.max_running_requests == 32
    assert config.kv_bytes_per_token == args.get_model_config().kv_bytes_per_token(
        dtype_bytes=1
    )


def test_non_fp8_kv_cache_uses_two_byte_elements(server_args):
    args = server_args(kv_cache_dtype="auto")
    config = SimConfig.from_server_args(args)
    assert config.kv_bytes_per_token == args.get_model_config().kv_bytes_per_token(
        dtype_bytes=2
    )


@pytest.mark.parametrize("prebuilt", [False, True])
@pytest.mark.parametrize("context_length,limit", [(128, 32), (16, 16)])
def test_oversized_prompt_is_rejected_without_queueing(
    server_args, prebuilt, context_length, limit,
):
    scheduler = Scheduler(SimConfig.from_server_args(server_args(
        fake_num_kv_blocks=8, page_size=4, context_length=context_length,
    )))
    with pytest.raises(ValueError, match=f"worker capacity is {limit}"):
        scheduler.submit(Request("too-large", list(range(limit + 1)), 1, prebuilt=prebuilt))
    assert scheduler.waiting == []
    assert scheduler.by_rid == {}
    assert scheduler.pool.used_blocks == 0
    assert not scheduler._wakeup.is_set()


def test_cli_context_override_and_simulation_settings_reach_model_card(tmp_path):
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 8192}')
    args = prepare_server_args([
        "--model-path", str(tmp_path),
        "--context-length", "131072",
        "--fake-num-kv-blocks", "8192",
        "--page-size", "16",
        "--fake-kv-generation-ms-per-1k-tokens", "100",
    ])
    config = SimConfig.from_server_args(args)
    assert args.get_model_config().context_len == 131072
    assert config.context_len == 131072
    assert config.total_kv_tokens == 131072
    assert config.kv_generation_ms_per_1k == 100
    assert _SchedulerInitResult(config).scheduler_infos[0]["max_req_input_len"] == 131072


async def test_generation_delay_bills_only_uncached_tokens(server_args, monkeypatch):
    config = SimConfig.from_server_args(server_args(
        page_size=4, max_prefill_tokens=4,
        fake_kv_generation_ms_per_1k_tokens=100, fake_speedup_ratio=10,
    ))
    sleep = AsyncMock()
    monkeypatch.setattr("fakeengine.timing.asyncio.sleep", sleep)
    scheduler = Scheduler(config)
    try:
        for rid, length, prebuilt, salt, computed_per_step in [
            ("cold", 8, False, "a", [4, 4]),
            ("warm", 8, False, "a", [0, 0]),
            ("extend", 10, False, "a", [0, 0, 2]),
            ("transfer", 12, True, "a", [0, 0, 0]),
            ("other-tenant", 8, False, "b", [4, 4]),
        ]:
            sleep.reset_mock()
            request = scheduler.submit(Request(
                rid, list(range(length)), 1, prebuilt=prebuilt, cache_salt=salt,
            ))
            for _ in computed_per_step:
                assert await scheduler._prefill_step()
            assert request.finished
            expected_ms = [
                config.prefill_overhead_ms + tokens / 1000 * (config.prefill_ms_per_1k + 100)
                for tokens in computed_per_step
            ]
            assert [call.args[0] for call in sleep.await_args_list] == pytest.approx(
                [ms / 1000 / 10 for ms in expected_ms]
            )
            assert scheduler.last_step_ms == pytest.approx(expected_ms[-1])
    finally:
        await scheduler.stop()


async def test_disabled_cache_bills_generation_on_repeated_prompts(server_args, monkeypatch):
    scheduler = Scheduler(SimConfig.from_server_args(server_args(
        disable_radix_cache=True, fake_kv_generation_ms_per_1k_tokens=100,
        fake_prefill_ms_per_1k_tokens=0, fake_prefill_overhead_ms=0,
        fake_speedup_ratio=10,
    )))
    sleep = AsyncMock()
    monkeypatch.setattr("fakeengine.timing.asyncio.sleep", sleep)
    try:
        for rid in ("first", "repeat"):
            request = scheduler.submit(Request(rid, list(range(1000)), 1))
            assert await scheduler._prefill_step()
            assert request.finished
            assert request.cached_tokens == 0
        assert [call.args[0] for call in sleep.await_args_list] == pytest.approx([0.01, 0.01])
    finally:
        await scheduler.stop()


def test_generation_cost_does_not_change_decode_or_transfer(server_args):
    baseline = TimingModel(SimConfig.from_server_args(server_args()))
    timing = TimingModel(SimConfig.from_server_args(server_args(
        fake_kv_generation_ms_per_1k_tokens=100,
    )))
    assert timing.kv_generation_ms(1000) == 100
    assert timing.prefill_ms(1000) == baseline.prefill_ms(1000) + 100
    assert timing.decode_step_ms(4) == baseline.decode_step_ms(4)
    assert timing.transfer_ms(1000) == baseline.transfer_ms(1000)
    link = Link(name="rack", bandwidth_gb_s=50, latency_ms=0.05)
    assert timing.transfer_ms(1000, link) == baseline.transfer_ms(1000, link)


async def test_observed_generation_time_excludes_cache_transfer_and_no_peer_probes(server_args):
    config = SimConfig.from_server_args(server_args(
        page_size=4, max_prefill_tokens=4, fake_kv_generation_ms_per_1k_tokens=100,
        fake_speedup_ratio=10,
    ))
    scheduler = Scheduler(config)
    try:
        for rid, prompt, prebuilt, enabled, expected_seconds in [
            ("cold", list(range(8)), False, True, 0.00008),
            ("warm", list(range(8)), False, True, 0.00008),
            ("extend", list(range(10)), False, True, 0.0001),
            ("transfer", list(range(12)), True, True, 0.0001),
            ("probe", list(range(100, 108)), False, False, 0.0001),
        ]:
            request = scheduler.submit(Request(
                rid, prompt, 1, prebuilt=prebuilt, observe_cache=enabled,
            ))
            while not request.finished:
                await scheduler._prefill_step()
            assert scheduler.kv_generation_seconds == pytest.approx(expected_seconds)
    finally:
        await scheduler.stop()


@pytest.mark.parametrize("publish", [False, True])
def test_pool_observes_pressure_eviction_separately_from_flush_and_protects_active_blocks(publish):
    observations, removed = [], []
    pool = BlockPool(
        3, 4, on_evict=removed.extend if publish else None,
        on_change=lambda: observations.append(
            (pool.cached_blocks, pool.used_blocks, pool.evictable_blocks,
             pool.evicted_blocks, pool.cleared_blocks),
        ),
    )
    hashes = block_hashes(list(range(12)), 4)
    blocks = pool.allocate(3)
    pool.insert_prefix(hashes, blocks)
    assert observations[-1] == (3, 3, 0, 0, 0)
    assert pool.allocate(1) is None
    pool.clear_cache()
    assert observations[-1] == (3, 3, 0, 0, 0)
    pool.free(blocks)
    assert observations[-1] == (3, 3, 3, 0, 0)
    shared = pool.acquire_prefix(hashes[:1])
    assert observations[-1] == (3, 3, 2, 0, 0)
    allocated = pool.allocate(1)
    assert observations[-1] == (2, 3, 1, 1, 0)
    pool.free(allocated)
    pool.clear_cache()
    assert observations[-1] == (1, 1, 0, 1, 1)
    pool.free(shared)
    pool.clear_cache()
    assert observations[-1] == (0, 0, 0, 1, 2)
    assert len(removed) == (3 if publish else 0)
