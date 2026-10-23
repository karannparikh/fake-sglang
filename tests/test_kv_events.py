"""KV-event publishing: the wire format and the policy around it.

These subscribe with a raw ZMQ socket and decode with msgspec, i.e. exactly
what Dynamo's Rust router does, so a passing test here means the bytes on the
wire are the bytes Dynamo parses.
"""

from __future__ import annotations

import asyncio
import json

import msgspec
import pytest
import zmq

from sglang import Engine
from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored

BLOCK_SIZE = 4


def _kv_config(port: int) -> str:
    return json.dumps({"publisher": "zmq", "endpoint": f"tcp://*:{port}"})


@pytest.mark.parametrize("dp_size", [1, 2])
def test_engine_rejects_ipc_kv_endpoint_before_creating_publisher(
    server_args, monkeypatch, dp_size,
):
    def unexpected_publisher(*args, **kwargs):
        pytest.fail("IPC configuration must fail before creating a publisher")

    monkeypatch.setattr("fakeengine.kv_events.ZmqEventPublisher", unexpected_publisher)
    with pytest.raises(ValueError, match="IPC endpoints.*--kv-events-config.*TCP"):
        Engine(server_args(
            dp_size=dp_size,
            enable_dp_attention=dp_size > 1,
            kv_events_config=json.dumps({
                "publisher": "zmq", "endpoint": "ipc:///tmp/fake-kv-events",
            }),
        ))


def test_disabled_kv_publisher_ignores_ipc_endpoint(server_args):
    from fakeengine.kv_events import NullKvEventEmitter, build_emitter

    emitter = build_emitter(server_args(kv_events_config=json.dumps({
        "publisher": "null", "endpoint": "ipc:///tmp/fake-kv-events",
    })), BLOCK_SIZE)
    assert isinstance(emitter, NullKvEventEmitter)


class Subscriber:
    """A stand-in for the Rust KvEventPublisher's ZMQ subscription."""

    def __init__(self, port: int) -> None:
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(f"tcp://127.0.0.1:{port}")

    async def drain(self, settle_s: float = 0.4) -> list:
        """Collect whatever has arrived, then return the flattened events."""
        await asyncio.sleep(settle_s)
        events = []
        while self.sock.poll(50):
            topic, seq, payload = self.sock.recv_multipart()
            batch = msgspec.msgpack.decode(payload)
            assert len(batch) == 3, "batch must encode as [ts, events, attn_dp_rank]"
            assert int.from_bytes(seq, "big") >= 0
            events.extend(batch[1])
        return events

    def close(self) -> None:
        self.sock.close(linger=0)


@pytest.fixture
async def publishing_engine(server_args, unused_port):
    subscriber = Subscriber(unused_port)
    engine = Engine(
        server_args=server_args(
            page_size=BLOCK_SIZE, kv_events_config=_kv_config(unused_port)
        )
    )
    # ZMQ PUB drops anything published before a subscriber has connected.
    await asyncio.sleep(0.3)
    try:
        yield engine, subscriber
    finally:
        subscriber.close()
        await engine._async_shutdown()


async def _generate(engine, prompt, max_new_tokens=2):
    stream = await engine.async_generate(
        input_ids=list(prompt),
        sampling_params={"max_new_tokens": max_new_tokens},
        stream=True,
    )
    async for _ in stream:
        pass


async def test_a_request_publishes_a_block_stored_event(publishing_engine):
    engine, subscriber = publishing_engine
    await _generate(engine, range(1000, 1000 + 4 * BLOCK_SIZE))

    events = await subscriber.drain()
    stored = [e for e in events if e.get("type") == "BlockStored"]
    assert stored, f"expected a BlockStored event, got {events}"


async def test_stored_event_has_the_shape_dynamo_parses(publishing_engine):
    engine, subscriber = publishing_engine
    prompt = list(range(2000, 2000 + 3 * BLOCK_SIZE))
    await _generate(engine, prompt)

    [stored] = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]

    # Dynamo drops any block whose token count != block_size, and slices
    # token_ids block_size at a time, so these two must agree exactly.
    assert stored["block_size"] == BLOCK_SIZE
    assert len(stored["token_ids"]) == len(stored["block_hashes"]) * BLOCK_SIZE
    assert stored["token_ids"] == prompt
    assert stored["parent_block_hash"] is None      # first blocks of a new chain
    assert all(isinstance(h, int) and h >= 0 for h in stored["block_hashes"])


async def test_published_tokens_hash_to_what_the_frontend_computes(publishing_engine):
    """The property KV-aware routing actually depends on.

    Dynamo recomputes its prefix key from the published token_ids
    (lib/kv-router/src/zmq_wire/convert.rs). If that recomputation does not
    reproduce the frontend's per-block hashes for the same prompt, the router
    silently never matches and falls back to round-robin.
    """
    from dynamo.llm import compute_block_hash_for_seq

    engine, subscriber = publishing_engine
    prompt = list(range(3000, 3000 + 3 * BLOCK_SIZE))
    await _generate(engine, prompt)

    [stored] = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]

    frontend = compute_block_hash_for_seq(prompt, BLOCK_SIZE)
    tokens = stored["token_ids"]
    worker = [
        compute_block_hash_for_seq(tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE], BLOCK_SIZE)[0]
        for i in range(len(stored["block_hashes"]))
    ]
    assert worker == frontend[: len(worker)]


async def test_partial_trailing_block_is_not_published(publishing_engine):
    """A block Dynamo would drop must never be put on the wire."""
    engine, subscriber = publishing_engine
    # Two whole blocks plus a remainder.
    await _generate(engine, range(4000, 4000 + 2 * BLOCK_SIZE + 2))

    stored = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]
    for event in stored:
        assert len(event["token_ids"]) == len(event["block_hashes"]) * BLOCK_SIZE
        assert len(event["block_hashes"]) <= 2


async def test_repeated_prompt_does_not_republish_cached_blocks(publishing_engine):
    engine, subscriber = publishing_engine
    prompt = list(range(5000, 5000 + 3 * BLOCK_SIZE))

    await _generate(engine, prompt)
    first = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]
    assert first

    await _generate(engine, prompt)          # identical prompt: a full cache hit
    second = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]
    assert second == [], f"cached blocks were re-announced: {second}"


async def test_extending_a_cached_prefix_chains_to_its_parent(publishing_engine):
    engine, subscriber = publishing_engine
    base = list(range(6000, 6000 + 2 * BLOCK_SIZE))

    await _generate(engine, base)
    [first] = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]

    await _generate(engine, base + list(range(7000, 7000 + BLOCK_SIZE)))
    [second] = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]

    # Only the new block is announced, parented to the last cached one, so the
    # router sees one chain rather than two disjoint roots.
    assert len(second["block_hashes"]) == 1
    assert second["parent_block_hash"] == first["block_hashes"][-1]


async def test_eviction_publishes_a_removal(server_args, unused_port):
    """The router must be told when a prefix stops being resident."""
    subscriber = Subscriber(unused_port)
    engine = Engine(
        server_args=server_args(
            page_size=BLOCK_SIZE,
            kv_events_config=_kv_config(unused_port),
            fake_num_kv_blocks=8,          # room for 8 blocks total
        )
    )
    await asyncio.sleep(0.3)
    try:
        for start in range(0, 6):
            await _generate(engine, range(start * 1000, start * 1000 + 3 * BLOCK_SIZE))

        events = await subscriber.drain()
        removed = [e for e in events if e["type"] == "BlockRemoved"]
        assert removed, "pool pressure produced no BlockRemoved events"
        assert all(e["block_hashes"] for e in removed)
    finally:
        subscriber.close()
        await engine._async_shutdown()


async def test_disaggregated_decode_does_not_republish_transferred_blocks(
    server_args, unused_port
):
    """Only the worker that computed a prefix may claim it.

    A decode worker's KV arrived over the transfer; announcing it would tell
    the router two workers hold the same prefix and split traffic onto a worker
    that never computed it.
    """
    subscriber = Subscriber(unused_port)
    decode = Engine(
        server_args=server_args(
            page_size=BLOCK_SIZE,
            kv_events_config=_kv_config(unused_port),
            disaggregation_mode="decode",
        )
    )
    await asyncio.sleep(0.3)
    try:
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        stream = await decode.async_generate(
            input_ids=list(range(8000, 8000 + 3 * BLOCK_SIZE)),
            sampling_params={"max_new_tokens": 2},
            stream=True,
            bootstrap_host=FAKE_BOOTSTRAP_HOST,
            bootstrap_port=1,
            bootstrap_room=11,
        )
        async for _ in stream:
            pass

        stored = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]
        assert stored == []
    finally:
        subscriber.close()
        await decode._async_shutdown()


async def test_publishing_is_off_without_the_flag(server_args, unused_port):
    subscriber = Subscriber(unused_port)
    engine = Engine(server_args=server_args(page_size=BLOCK_SIZE))
    await asyncio.sleep(0.2)
    try:
        await _generate(engine, range(9000, 9000 + 3 * BLOCK_SIZE))
        assert await subscriber.drain(settle_s=0.2) == []
    finally:
        subscriber.close()
        await engine._async_shutdown()


async def test_alternating_prompts_each_publish_once(publishing_engine):
    """Interleaved prompts must not evict or re-announce each other."""
    engine, subscriber = publishing_engine
    first = list(range(10_000, 10_000 + 3 * BLOCK_SIZE))
    second = list(range(20_000, 20_000 + 3 * BLOCK_SIZE))

    await _generate(engine, first)
    await _generate(engine, second)
    cold = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]
    assert len(cold) == 2, f"expected one store per prompt, got {len(cold)}"

    await _generate(engine, first)
    await _generate(engine, second)
    warm = await subscriber.drain()
    assert [e for e in warm if e["type"] == "BlockStored"] == []
    assert [e for e in warm if e["type"] == "BlockRemoved"] == []


async def _generate_salted(engine, prompt, salt, max_new_tokens=2):
    stream = await engine.async_generate(
        input_ids=list(prompt),
        sampling_params={"max_new_tokens": max_new_tokens},
        stream=True,
        cache_salt=salt,
    )
    async for _ in stream:
        pass


async def test_salted_store_carries_the_salt_and_its_own_ids(publishing_engine):
    """Same prompt, two tenants: two stores, disjoint ids, each tagged with
    its salt -- the router must see two caches, not one."""
    engine, subscriber = publishing_engine
    prompt = list(range(5000, 5000 + 2 * BLOCK_SIZE))
    await _generate_salted(engine, prompt, "tenant-a")
    await _generate_salted(engine, prompt, "tenant-b")
    stored = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]
    assert [e.get("cache_salt") for e in stored] == ["tenant-a", "tenant-b"]
    assert not set(stored[0]["block_hashes"]) & set(stored[1]["block_hashes"])
    assert stored[0]["token_ids"] == stored[1]["token_ids"] == prompt


async def test_unsalted_store_omits_the_salt(publishing_engine):
    """An empty salt is no salt (SGLang normalizes "" to None), and the field
    is omitted from the wire rather than sent as an empty namespace."""
    engine, subscriber = publishing_engine
    await _generate_salted(engine, range(6000, 6000 + 2 * BLOCK_SIZE), "")
    [stored] = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]
    assert "cache_salt" not in stored


async def test_salted_tokens_hash_to_the_frontends_salted_key(publishing_engine):
    """The routing property for tenants: rehashing the published tokens under
    the published salt reproduces the frontend's key for a request with that
    salt -- and differs from the unsalted key, so tenants never match each
    other's prefixes."""
    from dynamo.llm import compute_block_hash_for_seq

    engine, subscriber = publishing_engine
    prompt = list(range(7000, 7000 + 3 * BLOCK_SIZE))
    await _generate_salted(engine, prompt, "tenant-a")
    [stored] = [e for e in await subscriber.drain() if e["type"] == "BlockStored"]

    salt = stored["cache_salt"]
    frontend = compute_block_hash_for_seq(prompt, BLOCK_SIZE, cache_namespace="tenant-a")
    tokens = stored["token_ids"]
    worker = [
        compute_block_hash_for_seq(
            tokens[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE], BLOCK_SIZE, cache_namespace=salt
        )[0]
        for i in range(len(stored["block_hashes"]))
    ]
    assert worker == frontend[: len(worker)]
    assert worker != compute_block_hash_for_seq(prompt, BLOCK_SIZE)[: len(worker)]
