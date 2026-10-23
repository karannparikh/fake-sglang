#!/usr/bin/env python3
"""Launch a frontend plus an arbitrary mix of fake workers from a YAML file.

    ./launch/cluster.sh launch/clusters/two-zones.yaml [-- extra worker args]

The file says how many prefill / decode / aggregated workers to run, where each
group sits in the network (``location``), and what a KV transfer costs over
each link (``topology``). Every worker gets the same topology table, and a
decode worker prices each handoff by the link between it and the prefill
worker that produced the KV -- so pairing decisions made by Dynamo's router
show up directly in TTFT and in the per-transfer log line:

    fake-sglang: KV room 7 zone-a/rack-1/node-1 -> zone-b/rack-1/node-1 over 'default' link ...

Like the other launch scripts: one terminal, every process's output prefixed
with its name, and the whole group torn down when any member exits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import signal
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from fakeengine.topology import Topology, TopologyError

ROOT = Path(__file__).resolve().parent.parent
ROLES = {"prefill": "prefill", "decode": "decode", "agg": None}

SYSTEM_PORT_BASE = 8081
BOOTSTRAP_PORT_BASE = 18998
KV_EVENTS_PORT_BASE = 25557


class ConfigError(Exception):
    pass


@dataclass
class Proc:
    name: str
    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    location: Optional[str] = None
    role: str = "frontend"
    ports: Dict[str, int] = field(default_factory=dict)
    popen: Optional[subprocess.Popen] = None


def _str_args(value, where: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ConfigError(f"{where}: args must be a list or a string, got {value!r}")


def _dp_size(args: List[str], where: str) -> int:
    parser = argparse.ArgumentParser(
        add_help=False, allow_abbrev=False, exit_on_error=False,
    )
    parser.add_argument("--dp-size", type=int, default=1)
    try:
        parsed, _ = parser.parse_known_args(args)
    except argparse.ArgumentError as exc:
        raise ConfigError(f"{where}: {exc}") from exc
    if parsed.dp_size < 1:
        raise ConfigError(f"{where}: --dp-size must be a positive integer")
    return parsed.dp_size


def build(
    config: dict, extra_args: List[str], python: str, port_offset: int = 0
) -> List[Proc]:
    frontend_cfg = config.get("frontend") or {}
    defaults = config.get("defaults") or {}
    workers_cfg = config.get("workers")
    if not workers_cfg:
        raise ConfigError("'workers' must list at least one worker group")

    topology = None
    if config.get("topology"):
        try:
            topology = Topology.from_dict(config["topology"])
        except TopologyError as exc:
            raise ConfigError(str(exc)) from exc
    topology_json = json.dumps(topology.to_dict()) if topology else None

    model = str(config.get("model") or "models/qwen3-0.6b")
    if model.startswith(("./", "../")) or (not model.startswith("/") and (ROOT / model).exists()):
        model = str((ROOT / model).resolve())
    if model.startswith("/") and not Path(model).is_dir():
        raise ConfigError(f"model {model} does not exist")
    served = str(config.get("served_model_name") or "fake-model")

    router_mode = frontend_cfg.get("router_mode")
    kv_events = frontend_cfg.get("kv_events", router_mode == "kv")
    page_size = int(defaults.get("page_size", 16 if kv_events else 1))

    http_port = int(frontend_cfg.get("http_port", 8000)) + port_offset
    frontend_argv = [python, "-m", "dynamo.frontend", "--http-port", str(http_port)]
    if router_mode:
        frontend_argv += ["--router-mode", str(router_mode)]
    frontend_argv += _str_args(frontend_cfg.get("args"), "frontend")
    procs = [Proc("frontend", frontend_argv, ports={"http": http_port})]

    counters = {role: itertools.count() for role in ROLES}
    system_ports = itertools.count(SYSTEM_PORT_BASE + port_offset)
    bootstrap_ports = itertools.count(BOOTSTRAP_PORT_BASE + port_offset)
    next_kv_port = KV_EVENTS_PORT_BASE + port_offset
    have = set()

    for i, group in enumerate(workers_cfg):
        where = f"workers[{i}]"
        role = group.get("role")
        if role not in ROLES:
            raise ConfigError(f"{where}: role must be one of {sorted(ROLES)}, got {role!r}")
        count = int(group.get("count", 1))
        if count < 1:
            continue
        have.add(role)

        location = group.get("location")
        if location is not None:
            if topology is None:
                raise ConfigError(f"{where}: has a location but the file has no topology")
            try:
                topology.parse_location(location)
            except TopologyError as exc:
                raise ConfigError(f"{where}: {exc}") from exc
        elif topology is not None:
            raise ConfigError(
                f"{where}: every worker group needs a location when a topology is set"
            )

        group_args = _str_args(defaults.get("args"), "defaults") + _str_args(
            group.get("args"), where
        )
        worker_args = group_args + extra_args
        dp_size = _dp_size(worker_args, where)
        for _ in range(count):
            n = next(counters[role])
            name = f"{group.get('name') or role}-{n}"
            ports = {"system": next(system_ports)}
            argv = [
                python, "-m", "dynamo.sglang",
                "--model-path", model,
                "--served-model-name", served,
                "--skip-tokenizer-init",
                "--page-size", str(page_size),
            ]
            mode = ROLES[role]
            if mode:
                argv += ["--disaggregation-mode", mode]
            if mode == "prefill":
                ports["bootstrap"] = next(bootstrap_ports)
                argv += ["--disaggregation-bootstrap-port", str(ports["bootstrap"])]
            if kv_events:
                if next_kv_port < 1 or next_kv_port + dp_size - 1 > 65535:
                    raise ConfigError(
                        f"{name}: KV-event port range "
                        f"{next_kv_port}-{next_kv_port + dp_size - 1} "
                        "must be within 1-65535"
                    )
                ports["kv_events"] = next_kv_port
                for rank in range(1, dp_size):
                    ports[f"kv_events_rank_{rank}"] = next_kv_port + rank
                next_kv_port += dp_size
                endpoint = f"tcp://*:{ports['kv_events']}"
                argv += ["--kv-events-config", json.dumps({"publisher": "zmq", "endpoint": endpoint})]
            if location:
                argv += ["--fake-location", location, "--fake-topology", topology_json]
            argv += worker_args

            env = {"DYN_SYSTEM_PORT": str(ports["system"])}
            env.update({str(k): str(v) for k, v in (group.get("env") or {}).items()})
            procs.append(Proc(name, argv, env, location, role, ports))

    if ("prefill" in have) != ("decode" in have):
        raise ConfigError(
            "prefill and decode workers come as a pair: a prefill pool with no "
            "decode workers (or the reverse) can never complete a request"
        )
    return procs, topology


def check_ports(procs: List[Proc]) -> None:
    taken = []
    owners = {}
    for proc in procs:
        for kind, port in proc.ports.items():
            owner = f"{proc.name} {kind} :{port}"
            if port in owners:
                raise ConfigError(f"port assigned twice: {owners[port]} and {owner}")
            owners[port] = owner
            with socket.socket() as sock:
                try:
                    sock.bind(("0.0.0.0", port))
                except OSError:
                    taken.append(owner)
    if taken:
        raise ConfigError(
            "ports already in use (is another launch still running?):\n  "
            + "\n  ".join(taken)
        )


def describe(procs: List[Proc], topology: Optional[Topology]) -> str:
    lines = [f"  {'name':<14} {'role':<9} {'location':<28} ports"]
    for p in procs:
        ports = " ".join(f"{k}={v}" for k, v in p.ports.items())
        lines.append(f"  {p.name:<14} {p.role:<9} {p.location or '-':<28} {ports}")

    prefills = [p for p in procs if p.role == "prefill"]
    decodes = [p for p in procs if p.role == "decode"]
    if topology and prefills and decodes and len(prefills) * len(decodes) <= 400:
        lines += ["", "  KV link for each prefill -> decode pair:"]
        width = max(len(d.name) for d in decodes) + 1
        lines.append("  " + " " * 14 + "".join(f"{d.name:>{width}}" for d in decodes))
        for p in prefills:
            row = "".join(
                f"{topology.link_between(p.location, d.location).name:>{width}}"
                for d in decodes
            )
            lines.append(f"  {p.name:<14}{row}")
        lines += ["", "  links:"] + [
            f"    {l.name:<10} {l.bandwidth_gb_s:>8g} GB/s  {l.latency_ms:>6g} ms"
            for l in topology.links.values()
        ]
    return "\n".join(lines)


_COLORS = ["36", "33", "35", "32", "34", "31", "96", "93", "95", "92", "94", "91"]


def _pump(proc: Proc, color: Optional[str], width: int, log_file) -> None:
    prefix = f"{proc.name:<{width}} | "
    if color:
        prefix = f"\033[{color}m{prefix}\033[0m"
    for raw in proc.popen.stdout:
        line = raw.decode(errors="replace").rstrip("\n")
        sys.stdout.write(prefix + line + "\n")
        sys.stdout.flush()
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()


def run(procs: List[Proc], log_dir: Optional[Path], shared_env: Dict[str, str]) -> int:
    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    width = max(len(p.name) for p in procs)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    stopping = threading.Event()

    def stop(*_):
        stopping.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    threads = []
    try:
        for i, proc in enumerate(procs):
            env = {**os.environ, **shared_env, **proc.env}
            proc.popen = subprocess.Popen(
                proc.argv,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,    # our Ctrl-C handling owns teardown
            )
            log_file = open(log_dir / f"{proc.name}.log", "w") if log_dir else None
            color = _COLORS[i % len(_COLORS)] if use_color else None
            t = threading.Thread(target=_pump, args=(proc, color, width, log_file), daemon=True)
            t.start()
            threads.append(t)

        # Return as soon as ANY process exits, so a crashed worker ends the run
        # instead of leaving a half-up deployment that looks healthy.
        exited = None
        while not stopping.is_set() and exited is None:
            exited = next((p for p in procs if p.popen.poll() is not None), None)
            time.sleep(0.2)
        if exited is not None:
            print(f"\n{exited.name} exited with code {exited.popen.returncode}; stopping the rest")
            return exited.popen.returncode or 1
        return 0
    finally:
        print("\nCleaning up...")
        live = [p for p in procs if p.popen and p.popen.poll() is None]
        for p in live:
            try:
                os.killpg(p.popen.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.time() + 10
        for p in live:
            try:
                p.popen.wait(timeout=max(deadline - time.time(), 0.1))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.popen.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for t in threads:
            t.join(timeout=1)


def main() -> int:
    argv = sys.argv[1:]
    extra: List[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1 :]

    parser = argparse.ArgumentParser(
        description="Launch a Dynamo frontend and a mix of fake-sglang workers.",
        epilog="Arguments after `--` are forwarded to every worker.",
    )
    parser.add_argument("config", type=Path, help="cluster YAML, e.g. launch/clusters/two-zones.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and commands, start nothing")
    parser.add_argument("--log-dir", type=Path, help="also write each process's output to DIR/<name>.log")
    parser.add_argument(
        "--port-offset", type=int, default=0,
        help="shift every port (http, system, bootstrap, kv events) by N, to run "
        "beside another deployment",
    )
    args = parser.parse_args(argv)

    try:
        config = yaml.safe_load(args.config.read_text()) or {}
        procs, topology = build(
            config, extra, os.environ.get("PY") or sys.executable, args.port_offset
        )
    except (ConfigError, OSError, yaml.YAMLError) as exc:
        print(f"error: {args.config}: {exc}", file=sys.stderr)
        return 2

    workers = len(procs) - 1
    print()
    print("=" * 62)
    print(f" fake-sglang cluster: {args.config.name} -- frontend + {workers} worker(s)")
    print(" SIMULATED ENGINE -- no weights, no real inference.")
    print("=" * 62)
    print(describe(procs, topology))
    print()

    if args.dry_run:
        for p in procs:
            env = " ".join(f"{k}={v}" for k, v in p.env.items())
            print(f"# {p.name}\n{env + ' ' if env else ''}{shlex.join(p.argv)}\n")
        return 0

    try:
        check_ports(procs)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # A private discovery registry per run. The default ($TMPDIR/dynamo_store_kv)
    # is shared by every file-backend deployment on the machine, so two runs
    # would discover each other's workers, and a crashed run's stale entries
    # would linger into the next one.
    own_registry = "DYN_FILE_KV" not in os.environ
    registry = os.environ.get("DYN_FILE_KV") or tempfile.mkdtemp(prefix="fake-sglang-kv-")
    print(f"discovery registry: {registry}\n")
    try:
        return run(procs, args.log_dir, {"DYN_FILE_KV": registry})
    finally:
        if own_registry:
            shutil.rmtree(registry, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
