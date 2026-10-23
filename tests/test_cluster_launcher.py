"""launch/cluster.py turns a YAML file into the right processes and flags."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from contextlib import ExitStack
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("cluster", ROOT / "launch" / "cluster.py")
cluster = importlib.util.module_from_spec(_spec)
sys.modules["cluster"] = cluster    # dataclasses resolve their module by name
_spec.loader.exec_module(cluster)

TOPOLOGY = {
    "levels": ["rack", "node"],
    "links": {
        "node": {"bandwidth_gb_s": 400},
        "rack": {"bandwidth_gb_s": 50},
        "default": {"bandwidth_gb_s": 10},
    },
}


def _build(config, extra=(), offset=0):
    return cluster.build(config, list(extra), "python", offset)


def _flag(proc, name):
    return proc.argv[proc.argv.index(name) + 1] if name in proc.argv else None


@pytest.mark.parametrize("path", sorted((ROOT / "launch" / "clusters").glob("*.yaml")))
def test_shipped_examples_are_valid(path):
    procs, _ = _build(yaml.safe_load(path.read_text()))
    assert procs[0].name == "frontend"


def test_counts_roles_and_unique_ports():
    procs, _ = _build(
        {
            "frontend": {"router_mode": "kv"},
            "topology": TOPOLOGY,
            "workers": [
                {"role": "prefill", "count": 2, "location": "r1/n1"},
                {"role": "decode", "count": 3, "location": "r2/n1"},
                {"role": "agg", "location": "r1/n2"},
            ],
        }
    )
    workers = procs[1:]
    assert [p.name for p in workers] == [
        "prefill-0", "prefill-1", "decode-0", "decode-1", "decode-2", "agg-0",
    ]
    assert [_flag(p, "--disaggregation-mode") for p in workers] == (
        ["prefill"] * 2 + ["decode"] * 3 + [None]
    )
    ports = [port for p in procs for port in p.ports.values()]
    assert len(ports) == len(set(ports))
    # Every worker publishes KV events under the kv router, each on its own port.
    assert all(_flag(p, "--kv-events-config") for p in workers)
    # And every worker carries the same topology table.
    assert len({_flag(p, "--fake-topology") for p in workers}) == 1
    assert json.loads(_flag(workers[0], "--fake-topology"))["levels"] == ["rack", "node"]


def test_port_offset_and_arg_layering():
    procs, _ = _build(
        {
            "defaults": {"args": ["--fake-itl-ms", "5"]},
            "workers": [{"role": "agg", "args": "--fake-speedup-ratio 2"}],
        },
        extra=["--fake-num-kv-blocks", "64"],
        offset=100,
    )
    frontend, worker = procs
    assert frontend.ports["http"] == 8100
    assert worker.env["DYN_SYSTEM_PORT"] == "8181"
    tail = worker.argv[-6:]
    assert tail == ["--fake-itl-ms", "5", "--fake-speedup-ratio", "2", "--fake-num-kv-blocks", "64"]
    assert _flag(worker, "--kv-events-config") is None    # no kv router, no events


@pytest.mark.parametrize("extra, sizes", [
    ([], [2, 2, 3]),
    (["--dp-size", "4"], [4, 4, 4]),
    (["--dp-size=5"], [5, 5, 5]),
    (["--dp-size", "4", "--dp-size=1"], [1, 1, 1]),
])
def test_packed_workers_get_disjoint_port_ranges_after_arg_overrides(extra, sizes):
    procs, _ = _build({
        "frontend": {"router_mode": "kv"},
        "defaults": {"args": "--dp-size 2 --enable-dp-attention"},
        "workers": [
            {"role": "agg", "count": 2},
            {"role": "agg", "args": ["--dp-size=3", "--fake-itl-ms", "5"]},
        ],
    }, extra=extra, offset=100)
    next_port = cluster.KV_EVENTS_PORT_BASE + 100
    for worker, size in zip(procs[1:], sizes):
        rank_ports = {
            key: value for key, value in worker.ports.items() if key.startswith("kv_events")
        }
        assert rank_ports == {
            ("kv_events" if rank == 0 else f"kv_events_rank_{rank}"): next_port + rank
            for rank in range(size)
        }
        endpoint = json.loads(_flag(worker, "--kv-events-config"))["endpoint"]
        assert endpoint == f"tcp://*:{next_port}"
        next_port += size


@pytest.mark.parametrize("args", [
    ["--dp-size", "0"], ["--dp-size=-1"], ["--dp-size", "two"],
    ["--dp-size"], ["--dp-size", "--enable-dp-attention"],
])
def test_invalid_dp_size_fails_during_planning(args):
    with pytest.raises(cluster.ConfigError, match="workers\\[0\\].*--dp-size"):
        _build({"workers": [{"role": "agg"}]}, extra=args)


def test_packed_kv_port_range_cannot_exceed_tcp_port_limit():
    with pytest.raises(cluster.ConfigError, match="65535-65536.*1-65535"):
        _build({
            "frontend": {"router_mode": "kv"},
            "workers": [{"role": "agg", "args": "--dp-size 2 --enable-dp-attention"}],
        }, offset=65535 - cluster.KV_EVENTS_PORT_BASE)


def test_check_ports_checks_nonzero_rank_ports():
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        port = occupied.getsockname()[1]
        procs, _ = _build({
            "frontend": {"router_mode": "kv"},
            "workers": [{"role": "agg", "args": "--dp-size 2 --enable-dp-attention"}],
        }, offset=port - 1 - cluster.KV_EVENTS_PORT_BASE)
        with pytest.raises(cluster.ConfigError, match=f"agg-0 kv_events_rank_1 :{port}"):
            cluster.check_ports(procs)


def test_check_ports_rejects_duplicate_assignments(unused_port):
    procs = [
        cluster.Proc("first", [], ports={"kv_events_rank_1": unused_port}),
        cluster.Proc("second", [], ports={"kv_events": unused_port}),
    ]
    with pytest.raises(cluster.ConfigError, match="port assigned twice: first.*second"):
        cluster.check_ports(procs)


@pytest.fixture
def kv_port_base():
    for _ in range(20):
        with ExitStack() as cleanup:
            first = cleanup.enter_context(socket.socket())
            first.bind(("127.0.0.1", 0))
            base = first.getsockname()[1]
            if base + 4 > 65535:
                continue
            try:
                for port in range(base + 1, base + 5):
                    sock = cleanup.enter_context(socket.socket())
                    sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return base
    pytest.fail("Could not reserve five consecutive KV-event ports")


async def test_packed_workers_bind_all_assigned_tcp_ports(server_args, kv_port_base):
    from sglang import Engine

    procs, _ = _build({
        "frontend": {"router_mode": "kv"},
        "workers": [
            {"role": "agg", "args": "--dp-size 2 --enable-dp-attention"},
            {"role": "agg", "args": "--dp-size 3 --enable-dp-attention"},
        ],
    }, offset=kv_port_base - cluster.KV_EVENTS_PORT_BASE)
    engines = []
    try:
        for worker, size in zip(procs[1:], (2, 3)):
            engine = Engine(server_args(
                dp_size=size, enable_dp_attention=True,
                kv_events_config=_flag(worker, "--kv-events-config"),
            ))
            engines.append(engine)
            expected = [
                port for key, port in worker.ports.items() if key.startswith("kv_events")
            ]
            assert [
                event._publisher._endpoint for event in engine._rank_events
            ] == [f"tcp://*:{port}" for port in expected]
        assert len(engines) == 2
    finally:
        for engine in engines:
            await engine._async_shutdown()


@pytest.mark.parametrize(
    "config, message",
    [
        ({"workers": [{"role": "prefill"}]}, "come as a pair"),
        ({"workers": [{"role": "gpu"}]}, "role must be one of"),
        ({"workers": [{"role": "agg", "location": "r1/n1"}]}, "no topology"),
        ({"topology": TOPOLOGY, "workers": [{"role": "agg"}]}, "needs a location"),
        ({"topology": TOPOLOGY, "workers": [{"role": "agg", "location": "r1"}]}, "components"),
        ({"workers": []}, "at least one"),
    ],
)
def test_bad_configs_fail_before_anything_starts(config, message):
    with pytest.raises(cluster.ConfigError, match=message):
        _build(config)
