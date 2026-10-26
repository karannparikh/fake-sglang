import hashlib
import json

import pytest
import yaml

from deploy.local import render
from deploy.local.render import load_lab_config


def test_lab_pins_matching_dynamo_release():
    assert "ai-dynamo==1.6.0" in (render.HERE / "requirements.txt").read_text().splitlines()
    operator = yaml.safe_load((render.HERE / "operator-values.yaml").read_text())
    assert operator["upgradeCRD"] is True
    assert operator["controllerManager"]["manager"]["image"]["tag"] == "1.6.0"
    components = render.dynamo()[0]["spec"]["components"]
    planner = next(c for c in components if c["type"] == "planner")
    assert planner["podTemplate"]["spec"]["containers"][0]["image"] == (
        "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.6.0"
    )
    assert "701da25292ecca8f64d770cc0b17d389200019a1" in (
        render.HERE / "lab.sh"
    ).read_text()


def test_lab_operator_owns_crd_installation():
    launcher = (render.HERE / "lab.sh").read_text()
    assert "deploy/operator/config/crd/bases" not in launcher
    assert launcher.index("helm upgrade --install") < launcher.index(
        "kubectl wait --for=condition=Established"
    )


@pytest.mark.parametrize("overrides", [
    {"kv_blocks_per_worker": 0}, {"kv_block_size": -1},
    {"kv_block_size": True}, {"publish_kv_events": "false"},
    {"context_length": 0}, {"context_length": True},
    {"ranks_per_process": 0}, {"worker_processes_per_role": True},
    {"kv_generation_ms_per_1k_tokens": float("nan")}, {"unexpected": 1},
])
def test_lab_configuration_rejects_invalid_settings(tmp_path, overrides):
    config = {
        "kv_blocks_per_worker": 128, "kv_block_size": 16, "context_length": 131072,
        "publish_kv_events": True, "kv_generation_ms_per_1k_tokens": 25.0,
        "worker_processes_per_role": 2, "ranks_per_process": 2,
        **overrides,
    }
    path = tmp_path / "lab.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_lab_config(path)


@pytest.mark.parametrize("publish", [True, False])
def test_lab_renderer_wires_capacity_delay_and_event_mode(monkeypatch, publish):
    monkeypatch.setattr(render, "load_lab_config", lambda: {
        "kv_blocks_per_worker": 128, "kv_block_size": 8, "context_length": 131072,
        "publish_kv_events": publish, "kv_generation_ms_per_1k_tokens": 250,
        "worker_processes_per_role": 2, "ranks_per_process": 2,
    })
    components = render.dynamo()[0]["spec"]["components"]
    for component in components:
        if component["type"] not in ("prefill", "decode", "frontend"):
            continue
        args = component["podTemplate"]["spec"]["containers"][0]["args"]
        assert "--router-kv-events" in args
        if component["type"] != "frontend":
            assert args[args.index("--context-length") + 1] == "131072"
            assert args[args.index("--page-size") + 1] == "8"
            assert args[args.index("--fake-num-kv-blocks") + 1] == "128"
            assert args[args.index("--fake-kv-generation-ms-per-1k-tokens") + 1] == "250"
            assert ("--kv-events-config" in args) == publish


def test_lab_renderer_pins_configured_processes_and_ranks():
    config = load_lab_config()
    components = {c["name"]: c for c in render.dynamo()[0]["spec"]["components"]}
    assert components["prefill"]["replicas"] == config["worker_processes_per_role"]
    assert components["decode"]["replicas"] == config["worker_processes_per_role"]
    args = components["Planner"]["podTemplate"]["spec"]["containers"][0]["args"]
    planner = json.loads(args[args.index("--config") + 1])
    assert planner["min_endpoint"] == config["worker_processes_per_role"]
    assert planner["enable_load_scaling"] is True
    minimum_gpu_cost = sum(
        planner["min_endpoint"]
        * int(components[role]["podTemplate"]["spec"]["containers"][0]
              ["resources"]["limits"]["nvidia.com/gpu"])
        for role in ("prefill", "decode")
    )
    assert planner["max_gpu_budget"] == minimum_gpu_cost
    launcher = (render.HERE / "lab.sh").read_text()
    assert '"nvidia.com/gpu":"10"' in launcher
    assert 'get("nvidia.com/gpu") == "10"' in launcher


def test_lab_capacity_matches_configured_context():
    config = load_lab_config()
    assert config["context_length"] == 131072
    assert config["kv_blocks_per_worker"] * config["kv_block_size"] == 131072


def test_lab_workers_have_cpu_and_probe_headroom():
    components = render.dynamo()[0]["spec"]["components"]
    for component in components:
        if component["type"] not in ("prefill", "decode"):
            continue
        container = component["podTemplate"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["cpu"] == "2"
        assert container["readinessProbe"] == {
            "httpGet": {"path": "/health", "port": "system", "scheme": "HTTP"},
            "periodSeconds": 5,
            "timeoutSeconds": 4,
            "failureThreshold": 2,
            "successThreshold": 1,
        }
        assert container["livenessProbe"] == {
            "httpGet": {"path": "/live", "port": "system", "scheme": "HTTP"},
            "periodSeconds": 30,
            "timeoutSeconds": 10,
            "failureThreshold": 3,
            "successThreshold": 1,
        }


def test_otel_collector_has_memory_headroom():
    collector = next(
        item for item in render.observability()
        if item["kind"] == "Deployment"
        and item["metadata"]["name"] == "otel-collector"
    )
    resources = collector["spec"]["template"]["spec"]["containers"][0]["resources"]
    assert resources["requests"]["memory"] == "384Mi"
    assert resources["limits"]["memory"] == "1Gi"


def test_lab_admission_controls_are_bounded_and_mounted():
    ranks = load_lab_config()["ranks_per_process"]
    resources = render.dynamo()
    policy = next(r for r in resources if r["kind"] == "ConfigMap")
    assert f"request_queue_limit_per_worker: {10 * ranks}" in policy["data"]["router-policy.yaml"]
    assert f"raw_isl_token_queue_limit_per_worker: {65536 * ranks}" in policy["data"]["router-policy.yaml"]
    policy_digest = hashlib.sha256(policy["data"]["router-policy.yaml"].encode()).hexdigest()
    for component in resources[0]["spec"]["components"]:
        if component["type"] == "planner":
            continue
        assert component["podTemplate"]["metadata"]["annotations"][
            "dynamo-sim.local/router-policy-sha256"
        ] == policy_digest
        pod = component["podTemplate"]["spec"]
        container = pod["containers"][0]
        assert "--router-policy-config" in container["args"]
        assert pod["volumes"][0]["configMap"]["name"] == policy["metadata"]["name"]
        if component["type"] != "frontend":
            args = container["args"]
            assert args[args.index("--engine-request-limit") + 1] == str(8 * ranks)
            assert args[args.index("--max-running-requests") + 1] == str(8 * ranks)
            assert args[args.index("--dp-size") + 1] == str(ranks)
            assert ("--enable-dp-attention" in args) is (ranks > 1)
            assert args[args.index("--max-prefill-tokens") + 1] == "8192"
            assert {"name": "DYN_DYNAMO_REQUEST_QUEUE_LIMIT", "value": str(2 * ranks)} in container["env"]


@pytest.mark.parametrize("section", [render.infrastructure, render.observability, render.dynamo])
def test_lab_resources_render_with_benchmark_dashboards(section):
    resources = section()
    assert resources
    for item in resources:
        assert item["apiVersion"] and item["kind"] and item["metadata"]["name"]
        if item["kind"] == "ConfigMap":
            for name, data in item["data"].items():
                if name.endswith(".json"):
                    json.loads(data)
                elif name.endswith(".yaml"):
                    yaml.safe_load(data)
    assert json.loads(json.dumps(resources)) == resources


def test_grafana_uses_generated_secret_and_both_lab_dashboards():
    resources = render.observability()
    dashboard = next(r for r in resources if r["metadata"]["name"] == "grafana-dashboard")
    assert set(dashboard["data"]) == {"dynamo-lab.json", "dynamo-routing-live.json"}
    grafana = next(r for r in resources
                   if r["kind"] == "Deployment" and r["metadata"]["name"] == "grafana")
    env = grafana["spec"]["template"]["spec"]["containers"][0]["env"]
    password = next(e for e in env if e["name"] == "GF_SECURITY_ADMIN_PASSWORD")
    assert password == {
        "name": "GF_SECURITY_ADMIN_PASSWORD",
        "valueFrom": {"secretKeyRef": {"name": "grafana-admin", "key": "password"}},
    }
