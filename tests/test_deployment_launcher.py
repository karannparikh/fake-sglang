"""launch/deployment.py turns a YAML file into the right processes and flags."""

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
_spec = importlib.util.spec_from_file_location("deployment", ROOT / "launch" / "deployment.py")
deployment = importlib.util.module_from_spec(_spec)
sys.modules["deployment"] = deployment    # dataclasses resolve their module by name
_spec.loader.exec_module(deployment)

TOPOLOGY = {
    "levels": ["cluster", "zone"],
    "links": {
        "zone": {"bandwidth_gb_s": 400},
        "cluster": {"bandwidth_gb_s": 50},
        "default": {"bandwidth_gb_s": 10},
    },
}


def _build(config, extra=(), offset=0):
    return deployment.build(config, list(extra), "python", offset)


def _flag(proc, name):
    return proc.argv[proc.argv.index(name) + 1] if name in proc.argv else None


@pytest.mark.parametrize("path", sorted((ROOT / "launch" / "deployments").glob("*.yaml")))
def test_shipped_examples_are_valid(path):
    procs, _ = _build(yaml.safe_load(path.read_text()))
    assert procs[0].name == "frontend"


def test_counts_roles_and_unique_ports():
    procs, _ = _build(
        {
            "frontend": {"router_mode": "kv"},
            "topology": TOPOLOGY,
            "workers": [
                {"role": "prefill", "count": 2, "location": "c1/z1"},
                {"role": "decode", "count": 3, "location": "c2/z1"},
                {"role": "agg", "location": "c1/z2"},
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
    assert json.loads(_flag(workers[0], "--fake-topology"))["levels"] == ["cluster", "zone"]


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


def test_default_worker_page_size_matches_h200_profile():
    procs, _ = _build({"workers": [{"role": "agg"}]})
    assert _flag(procs[1], "--page-size") == "256"


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
    next_port = deployment.KV_EVENTS_PORT_BASE + 100
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
    with pytest.raises(deployment.ConfigError, match="workers\\[0\\].*--dp-size"):
        _build({"workers": [{"role": "agg"}]}, extra=args)


def test_packed_kv_port_range_cannot_exceed_tcp_port_limit():
    with pytest.raises(deployment.ConfigError, match="65535-65536.*1-65535"):
        _build({
            "frontend": {"router_mode": "kv"},
            "workers": [{"role": "agg", "args": "--dp-size 2 --enable-dp-attention"}],
        }, offset=65535 - deployment.KV_EVENTS_PORT_BASE)


def test_check_ports_checks_nonzero_rank_ports():
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        port = occupied.getsockname()[1]
        procs, _ = _build({
            "frontend": {"router_mode": "kv"},
            "workers": [{"role": "agg", "args": "--dp-size 2 --enable-dp-attention"}],
        }, offset=port - 1 - deployment.KV_EVENTS_PORT_BASE)
        with pytest.raises(deployment.ConfigError, match=f"agg-0 kv_events_rank_1 :{port}"):
            deployment.check_ports(procs)


def test_check_ports_rejects_duplicate_assignments(unused_port):
    procs = [
        deployment.Proc("first", [], ports={"kv_events_rank_1": unused_port}),
        deployment.Proc("second", [], ports={"kv_events": unused_port}),
    ]
    with pytest.raises(deployment.ConfigError, match="port assigned twice: first.*second"):
        deployment.check_ports(procs)


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
    }, offset=kv_port_base - deployment.KV_EVENTS_PORT_BASE)
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
        ({"workers": [{"role": "agg", "location": "c1/z1"}]}, "no topology"),
        ({"topology": TOPOLOGY, "workers": [{"role": "agg"}]}, "needs a location"),
        ({"topology": TOPOLOGY, "workers": [{"role": "agg", "location": "r1"}]}, "components"),
        ({"workers": []}, "at least one"),
    ],
)
def test_bad_configs_fail_before_anything_starts(config, message):
    with pytest.raises(deployment.ConfigError, match=message):
        _build(config)


# -- topology-aware KV transfer (topology.kv_transfer) -------------------------

REGIONS = {
    "levels": ["region", "cluster"],
    "links": {
        "cluster": {"bandwidth_gb_s": 50},
        "region": {"bandwidth_gb_s": 12.5},
        "default": {"bandwidth_gb_s": 1.25},
    },
}


def _placed(kv_transfer, workers=None, router_mode="kv"):
    return {
        "frontend": {"router_mode": router_mode},
        "topology": {**REGIONS, "kv_transfer": kv_transfer},
        "workers": workers or [
            {"role": "prefill", "location": "region-a/cluster-1"},
            {"role": "decode", "location": "region-a/cluster-2"},
            {"role": "prefill", "location": "region-b/cluster-1"},
            {"role": "decode", "location": "region-b/cluster-1"},
            {"role": "agg", "location": "region-b/cluster-2"},
        ],
    }


def test_kv_transfer_sets_dynamo_topology_env_on_prefill_and_decode_only():
    procs, _ = _build(_placed({"domain": "region"}))
    by_name = {p.name: p for p in procs}
    for name in ("prefill-0", "decode-0", "prefill-1", "decode-1"):
        env = by_name[name].env
        assert env["DYN_TOPOLOGY_ENABLED"] == "true"
        assert env["DYN_KV_TRANSFER_DOMAIN"] == "region"
        assert env["DYN_KV_TRANSFER_ENFORCEMENT"] == "required"
        assert "DYN_KV_TRANSFER_PREFERRED_WEIGHT" not in env
    # Aggregated workers never transfer KV, and the frontend reads no topology.
    for name in ("frontend", "agg-0"):
        assert not any(k.startswith(("DYN_TOPOLOGY", "DYN_KV_TRANSFER")) for k in by_name[name].env)
        assert by_name[name].topology_domains == {}
    # Values are full path prefixes: cluster-1 in region-a and region-b are different clusters.
    assert by_name["prefill-0"].topology_domains == {"region": "region-a", "cluster": "region-a/cluster-1"}
    assert by_name["prefill-1"].topology_domains == {"region": "region-b", "cluster": "region-b/cluster-1"}


def test_kv_transfer_preferred_passes_weight():
    procs, _ = _build(_placed({"domain": "cluster", "enforcement": "preferred", "preferred_weight": 0.85}))
    env = procs[1].env
    assert env["DYN_KV_TRANSFER_ENFORCEMENT"] == "preferred"
    assert env["DYN_KV_TRANSFER_PREFERRED_WEIGHT"] == "0.85"


def test_kv_transfer_preferred_allows_prefill_without_local_decode():
    # Preferred is only a bias, so a cluster with no decode worker is fine.
    procs, _ = _build(_placed({"domain": "cluster", "enforcement": "preferred", "preferred_weight": 1}))
    assert procs[1].env["DYN_KV_TRANSFER_PREFERRED_WEIGHT"] == "1.0"


def test_child_env_drops_inherited_topology_vars():
    # A stray weight from the shell would make Dynamo reject a required policy.
    shell = {"PATH": "/bin", "DYN_TOPOLOGY_ENABLED": "true",
             "DYN_KV_TRANSFER_PREFERRED_WEIGHT": "0.5", "DYN_KV_TRANSFER_DOMAIN": "rack"}
    policy = {"DYN_TOPOLOGY_ENABLED": "true", "DYN_KV_TRANSFER_DOMAIN": "zone"}
    env = deployment.child_env(shell, {"DYN_FILE_KV": "/kv"}, policy)
    assert env == {"PATH": "/bin", "DYN_FILE_KV": "/kv", **policy}
    assert deployment.child_env(shell, {}, {}) == {"PATH": "/bin"}


def test_without_kv_transfer_no_topology_env():
    config = _placed(None)
    del config["topology"]["kv_transfer"]
    procs, _ = _build(config)
    assert all(not p.topology_domains for p in procs)
    assert all("DYN_TOPOLOGY_ENABLED" not in p.env for p in procs)


def test_topology_dirs_hold_one_file_per_level(tmp_path):
    procs, _ = _build(_placed({"domain": "region"}))
    deployment.assign_topology_dirs(procs, tmp_path)
    deployment.write_topology_dirs(procs)
    prefill = next(p for p in procs if p.name == "prefill-1")
    directory = Path(prefill.env["DYN_TOPOLOGY_MOUNT_PATH"])
    assert directory == tmp_path / "prefill-1"
    assert {f.name: f.read_text() for f in directory.iterdir()} == {
        "region": "region-b\n", "cluster": "region-b/cluster-1\n",
    }
    assert sorted(d.name for d in tmp_path.iterdir()) == [
        "decode-0", "decode-1", "prefill-0", "prefill-1",
    ]


def test_dynamo_reads_the_topology_the_launcher_writes(tmp_path, monkeypatch):
    topology = pytest.importorskip("dynamo.common.utils.topology")
    procs, _ = _build(_placed({"domain": "region"}))
    deployment.assign_topology_dirs(procs, tmp_path)
    deployment.write_topology_dirs(procs)
    decode = next(p for p in procs if p.name == "decode-1")
    for key in deployment.TOPOLOGY_ENV:
        monkeypatch.delenv(key, raising=False)
    for key, value in decode.env.items():
        monkeypatch.setenv(key, value)
    config = topology.read_topology_config(poll_interval=0.01, poll_timeout=0.1)
    assert config.topology_domains == {"region": "region-b", "cluster": "region-b/cluster-1"}
    assert config.kv_transfer_domain == "region"
    assert config.kv_transfer_enforcement == "required"


def test_shipped_zone_local_example_keeps_every_pair_in_one_zone():
    path = ROOT / "launch" / "deployments" / "zone-local.yaml"
    procs, topology = _build(yaml.safe_load(path.read_text()))
    plan = deployment.describe(procs, topology)
    assert "required within the same zone" in plan
    assert "default" not in plan.split("KV link for each")[1].split("links:")[0]


@pytest.mark.parametrize(
    "kv_transfer, router_mode, workers, message",
    [
        ({"domain": "region"}, "round-robin", None, "requires frontend.router_mode: kv"),
        ({"domain": "zone"}, "kv", None, "domain must be one of the levels"),
        ({"domain": "region", "enforcement": "strict"}, "kv", None, "required or preferred"),
        ({"domain": "region", "enforcement": "preferred"}, "kv", None, "preferred_weight"),
        ({"domain": "region", "enforcement": "preferred", "preferred_weight": 2}, "kv", None,
         "preferred_weight"),
        ({"domain": "region", "enforcement": "preferred", "preferred_weight": 0}, "kv", None,
         "above 0"),
        ({"domain": "region", "preferred_weight": 0.5}, "kv", None, "only applies to preferred"),
        ({"domain": "region", "weight": 0.5}, "kv", None, "unknown keys"),
        ("region", "kv", None, "must be a mapping"),
        # Required, and region-c's prefill has no decode in its region.
        ({"domain": "region"}, "kv", [
            {"role": "prefill", "location": "region-a/cluster-1"},
            {"role": "prefill", "location": "region-c/cluster-1"},
            {"role": "decode", "location": "region-a/cluster-1"},
        ], r"prefill-1 \(region=region-c\)"),
        # Same cluster name in two regions is still two clusters.
        ({"domain": "cluster"}, "kv", [
            {"role": "prefill", "location": "region-a/cluster-1"},
            {"role": "decode", "location": "region-b/cluster-1"},
        ], r"cluster=region-a/cluster-1"),
        ({"domain": "region"}, "kv", [
            {"role": "prefill", "location": "region-a/cluster-1",
             "env": {"DYN_KV_TRANSFER_DOMAIN": "cluster"}},
            {"role": "decode", "location": "region-a/cluster-1"},
        ], "which topology.kv_transfer manages"),
        ({"domain": "region"}, "kv", [
            {"role": "prefill", "location": "region=a/cluster-1"},
            {"role": "decode", "location": "region=a/cluster-1"},
        ], "cannot be a topology value"),
    ],
)
def test_bad_kv_transfer_configs_fail_before_anything_starts(kv_transfer, router_mode, workers, message):
    with pytest.raises(deployment.ConfigError, match=message):
        _build(_placed(kv_transfer, workers, router_mode))


def test_kv_transfer_needs_a_topology():
    with pytest.raises(deployment.ConfigError, match="needs a topology"):
        deployment._kv_transfer_policy({"domain": "region"}, None, "kv")
