import hashlib
import json
from pathlib import Path

import pytest
import yaml

from deploy.local import render
from deploy.local.render import load_lab_config
from fakeengine.config import SimConfig
from sglang.srt.server_args import ServerArgs


def components():
    return {
        component["type"]: component
        for component in render.dynamo()[0]["spec"]["components"]
    }


def test_lab_pins_matching_dynamo_release():
    assert "ai-dynamo==1.6.0" in (
        render.HERE / "requirements.txt"
    ).read_text().splitlines()
    operator = yaml.safe_load((render.HERE / "operator-values.yaml").read_text())
    assert operator["upgradeCRD"] is True
    assert operator["controllerManager"]["manager"]["image"]["tag"] == "1.6.0"
    planner = components()["planner"]
    assert planner["podTemplate"]["spec"]["containers"][0]["image"] == (
        "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.6.0"
    )
    assert "701da25292ecca8f64d770cc0b17d389200019a1" in (
        render.HERE / "lab.sh"
    ).read_text()


@pytest.mark.parametrize(
    "overrides",
    [
        {"ranks_per_process": 0},
        {"worker_processes_per_role": True},
        {"publish_kv_events": "false"},
        {"kv_generation_ms_per_1k_tokens": float("nan")},
        {"unexpected": 1},
    ],
)
def test_lab_configuration_rejects_invalid_settings(tmp_path, overrides):
    config = {
        "publish_kv_events": True,
        "kv_generation_ms_per_1k_tokens": 25.0,
        "worker_processes_per_role": 4,
        "ranks_per_process": 10,
        **overrides,
    }
    path = tmp_path / "lab.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_lab_config(path)


def test_lab_uses_merged_h200_application_defaults():
    config = load_lab_config()
    assert set(config) == {
        "publish_kv_events",
        "kv_generation_ms_per_1k_tokens",
        "worker_processes_per_role",
        "ranks_per_process",
    }
    args = ServerArgs(
        model_path=str(
            Path(__file__).resolve().parents[1] / "models/qwen3-0.6b"
        ),
        dp_size=config["ranks_per_process"],
        enable_dp_attention=True,
    )
    profile = SimConfig.from_server_args(args)
    assert args.fake_num_kv_blocks == 15_509
    assert args.page_size == 256
    assert args.context_length == 1_048_576
    assert args.kv_cache_dtype == "fp8_e4m3"
    assert profile.total_kv_tokens == 3_970_304
    assert profile.max_running_requests == 32

    for role in ("prefill", "decode"):
        worker_args = components()[role]["podTemplate"]["spec"]["containers"][0][
            "args"
        ]
        assert not {
            "--fake-num-kv-blocks",
            "--page-size",
            "--context-length",
            "--kv-cache-dtype",
            "--max-running-requests",
        }.intersection(worker_args)


def test_lab_pins_required_topology_and_worker_availability():
    config = load_lab_config()
    rendered = components()
    assert {
        role: component["replicas"] for role, component in rendered.items()
    } == {"frontend": 1, "prefill": 4, "decode": 4, "planner": 1}
    for role in ("prefill", "decode"):
        container = rendered[role]["podTemplate"]["spec"]["containers"][0]
        args = container["args"]
        assert args[args.index("--dp-size") + 1] == "10"
        assert "--enable-dp-attention" in args
        assert container["resources"]["limits"]["cpu"] == "2"
        assert container["readinessProbe"]["failureThreshold"] == 2
        assert container["livenessProbe"] == {
            "httpGet": {
                "path": "/live",
                "port": "system",
                "scheme": "HTTP",
            },
            "periodSeconds": 30,
            "timeoutSeconds": 10,
            "failureThreshold": 3,
            "successThreshold": 1,
        }
    planner_args = rendered["planner"]["podTemplate"]["spec"]["containers"][0][
        "args"
    ]
    planner = json.loads(planner_args[planner_args.index("--config") + 1])
    assert planner["min_endpoint"] == config["worker_processes_per_role"]
    assert planner["max_gpu_budget"] == 2 * config["worker_processes_per_role"]


def test_router_policy_is_mounted_and_rolls_policy_changes():
    resources = render.dynamo()
    policy = next(item for item in resources if item["kind"] == "ConfigMap")
    digest = hashlib.sha256(policy["data"]["router-policy.yaml"].encode()).hexdigest()
    ranks = load_lab_config()["ranks_per_process"]
    assert f"request_queue_limit_per_worker: {10 * ranks}" in policy["data"][
        "router-policy.yaml"
    ]
    for component in resources[0]["spec"]["components"]:
        if component["type"] == "planner":
            continue
        assert component["podTemplate"]["metadata"]["annotations"][
            "dynamo-sim.local/router-policy-sha256"
        ] == digest
        pod = component["podTemplate"]["spec"]
        assert pod["volumes"][0]["configMap"]["name"] == policy["metadata"]["name"]
        assert "--router-policy-config" in pod["containers"][0]["args"]


def test_observability_contains_only_benchmark_dependencies():
    resources = render.observability()
    deployments = {
        item["metadata"]["name"]: item
        for item in resources
        if item["kind"] == "Deployment"
    }
    assert set(deployments) == {"otel-collector", "prometheus"}
    collector = deployments["otel-collector"]["spec"]["template"]["spec"][
        "containers"
    ][0]
    assert collector["resources"]["requests"]["memory"] == "384Mi"
    assert collector["resources"]["limits"]["memory"] == "1Gi"


@pytest.mark.parametrize(
    "section",
    [render.infrastructure, render.observability, render.dynamo],
)
def test_lab_resources_are_json_serializable(section):
    resources = section()
    assert resources
    assert json.loads(json.dumps(resources)) == resources
