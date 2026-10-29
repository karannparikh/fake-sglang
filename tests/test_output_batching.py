import asyncio

import pytest

from fakeengine import tokens as token_gen
from fakeengine.config import SimConfig
from fakeengine.scheduler import Request, Scheduler
from sglang.srt.entrypoints.engine import Engine


def _config(server_args, **overrides):
    return SimConfig.from_server_args(server_args(**overrides))


async def _wait_until(predicate, timeout=2):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_stream_interval_must_be_a_positive_integer(server_args, value):
    with pytest.raises(ValueError, match="stream_interval"):
        _config(server_args, stream_interval=value)


async def test_scheduler_publishes_cross_request_output_batches(server_args):
    config = _config(server_args)
    scheduler = Scheduler(config)
    published = []
    separate = scheduler._separate_outputs

    def capture(batch):
        published.append(batch)
        separate(batch)

    scheduler._separate_outputs = capture
    first = scheduler.submit(Request("first", [1, 2, 3], 3))
    second = scheduler.submit(Request("second", [4, 5, 6], 3))

    assert await scheduler._prefill_step()
    await scheduler._decode_step()
    await scheduler._decode_step()

    assert len(published) == 3
    assert [
        [output.request.rid for output in batch.outputs]
        for batch in published
    ] == [["first", "second"]] * 3
    first_tokens, first_reason = await first.next_output()
    second_tokens, second_reason = await second.next_output()
    assert first_tokens == token_gen.take(first.prompt_ids, config.vocab_size, 3)
    assert second_tokens == token_gen.take(second.prompt_ids, config.vocab_size, 3)
    assert first_reason == second_reason == "length"


async def test_default_interval_streams_one_token_per_chunk(server_args):
    engine = Engine(server_args())
    try:
        stream = await engine.async_generate(
            input_ids=[1, 2, 3],
            sampling_params={"max_new_tokens": 3},
            rid="default-interval",
        )
        chunks = [chunk async for chunk in stream]

        assert [len(chunk["output_ids"]) for chunk in chunks] == [1, 1, 1]
        assert [
            chunk["meta_info"]["completion_tokens"] for chunk in chunks
        ] == [1, 2, 3]
        assert chunks[-1]["meta_info"]["finish_reason"]["type"] == "length"
    finally:
        await engine._async_shutdown()


async def test_stream_interval_batches_and_flushes_terminal_tail(server_args):
    engine = Engine(server_args(stream_interval=3))
    try:
        stream = await engine.async_generate(
            input_ids=[1, 2, 3],
            sampling_params={"max_new_tokens": 8},
            rid="interval",
        )
        chunks = [chunk async for chunk in stream]

        assert [len(chunk["output_ids"]) for chunk in chunks] == [3, 3, 2]
        assert [
            chunk["meta_info"]["completion_tokens"] for chunk in chunks
        ] == [3, 6, 8]
        assert [chunk["meta_info"]["finish_reason"] for chunk in chunks[:-1]] == [
            None,
            None,
        ]
        assert chunks[-1]["meta_info"]["finish_reason"]["type"] == "length"
    finally:
        await engine._async_shutdown()


async def test_abort_flushes_unsent_output_with_finish_reason(server_args):
    engine = Engine(server_args(stream_interval=100))
    try:
        stream = await engine.async_generate(
            input_ids=[10, 11, 12],
            sampling_params={"max_new_tokens": 1000},
            rid="abort-tail",
        )
        first_chunk = asyncio.create_task(anext(stream))
        await _wait_until(lambda: "abort-tail" in engine.scheduler.by_rid)
        request = engine.scheduler.by_rid["abort-tail"]
        await _wait_until(lambda: len(request.generated) >= 4)
        expected = list(request.generated)

        assert engine.scheduler.abort(request.rid)
        chunks = [await first_chunk]
        chunks.extend([chunk async for chunk in stream])

        assert len(chunks) == 1
        assert chunks[0]["output_ids"] == expected
        assert chunks[0]["meta_info"]["completion_tokens"] == len(expected)
        assert chunks[0]["meta_info"]["finish_reason"]["type"] == "abort"
    finally:
        await engine._async_shutdown()


async def test_slow_consumer_has_one_coalesced_notification(server_args):
    scheduler = Scheduler(_config(
        server_args,
        fake_speedup_ratio=1_000_000,
        stream_interval=1,
    ))
    scheduler.start()
    request = scheduler.submit(Request("slow", [1, 2, 3], 50))
    try:
        await _wait_until(lambda: request.output_finished)

        assert request.pending_output_tokens == 50
        assert request.output_notification_pending
        assert scheduler._output_queue.empty()
        assert not hasattr(request, "queue")

        token_ids, finish_reason = await request.next_output()
        assert len(token_ids) == 50
        assert finish_reason == "length"
        assert request.pending_output_tokens == 0
        assert not request.output_notification_pending
    finally:
        await scheduler.stop()


async def test_packed_dp_ranks_batch_outputs_independently(server_args):
    engine = Engine(server_args(
        dp_size=2,
        enable_dp_attention=True,
        stream_interval=2,
    ))
    try:
        async def generate(rank, rid, prompt):
            stream = await engine.async_generate(
                input_ids=prompt,
                sampling_params={"max_new_tokens": 3},
                rid=rid,
                routed_dp_rank=rank,
            )
            return [chunk async for chunk in stream]

        first, second = await asyncio.gather(
            generate(0, "rank-0", [1, 2, 3]),
            generate(1, "rank-1", [4, 5, 6]),
        )

        assert [len(chunk["output_ids"]) for chunk in first] == [2, 1]
        assert [len(chunk["output_ids"]) for chunk in second] == [2, 1]
        assert {chunk["meta_info"]["id"] for chunk in first} == {"rank-0"}
        assert {chunk["meta_info"]["id"] for chunk in second} == {"rank-1"}
        assert engine.schedulers[0]._output_queue is not engine.schedulers[1]._output_queue
    finally:
        await engine._async_shutdown()
