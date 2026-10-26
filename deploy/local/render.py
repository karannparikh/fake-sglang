#!/usr/bin/env python3
"""Render the isolated CPU lab's Kubernetes resources as a JSON List."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def resource(kind, name, namespace=None, **fields):
    metadata = {"name": name}
    if namespace:
        metadata["namespace"] = namespace
    return {"apiVersion": "v1", "kind": kind, "metadata": metadata, **fields}


def configmap(name, namespace, files):
    return resource(
        "ConfigMap", name, namespace,
        data={key: (HERE / filename).read_text() for key, filename in files.items()},
    )


def service(name, namespace, ports):
    spec = {
        "selector": {"app": name},
        "ports": [
            {"name": key, "port": port, "targetPort": port,
             **({"nodePort": node_port} if node_port else {})}
            for key, port, node_port in ports
        ],
    }
    if any(node_port for _, _, node_port in ports):
        spec["type"] = "NodePort"
    return resource("Service", name, namespace, spec=spec)


def workload(name, namespace, image, ports, *, command=None, args=None, config=None,
             storage=None, env=None, health=None, service_account=None,
             mounts=None, memory="512Mi", memory_request="96Mi"):
    volumes = []
    volume_mounts = []
    items = []
    if config:
        filename, destination = config
        items.append(configmap(name + "-config", namespace, {filename: filename}))
        volumes.append({"name": "config", "configMap": {"name": name + "-config"}})
        volume_mounts.append(
            {"name": "config", "mountPath": destination, "subPath": filename,
             "readOnly": True}
        )
    if storage:
        size, destination = storage
        items.append(resource(
            "PersistentVolumeClaim", name + "-data", namespace,
            spec={"accessModes": ["ReadWriteOnce"],
                  "resources": {"requests": {"storage": size}}},
        ))
        volumes.append(
            {"name": "data", "persistentVolumeClaim": {"claimName": name + "-data"}}
        )
        volume_mounts.append({"name": "data", "mountPath": destination})
    for volume, mount in mounts or []:
        volumes.append(volume)
        volume_mounts.append(mount)
    container = {
        "name": name, "image": image,
        "ports": [{"name": key, "containerPort": port} for key, port, _ in ports],
        "resources": {"requests": {"cpu": "50m", "memory": memory_request},
                      "limits": {"cpu": "1", "memory": memory}},
        "volumeMounts": volume_mounts,
        "securityContext": {
            "allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]},
        },
    }
    if args:
        container["args"] = args
    if command:
        container["command"] = command
    if env:
        container["env"] = env
    if health:
        port, path = health
        probe = {"httpGet": {"path": path, "port": port}, "periodSeconds": 5,
                 "timeoutSeconds": 3}
        container["readinessProbe"] = probe
        container["startupProbe"] = {**probe, "failureThreshold": 60}
    pod_spec = {
        "securityContext": {"runAsNonRoot": True, "runAsUser": 1000,
                            "runAsGroup": 1000, "fsGroup": 1000},
        "containers": [container], "volumes": volumes,
    }
    if service_account:
        pod_spec["serviceAccountName"] = service_account
    else:
        pod_spec["automountServiceAccountToken"] = False
    deployment = resource(
        "Deployment", name, namespace,
        spec={"replicas": 1, "strategy": {"type": "Recreate"},
              "selector": {"matchLabels": {"app": name}},
              "template": {"metadata": {"labels": {"app": name}}, "spec": pod_spec}},
    )
    deployment["apiVersion"] = "apps/v1"
    items.extend([deployment, service(name, namespace, ports)])
    return items


def infrastructure():
    ns = "dynamo-system"
    items = [resource("Namespace", ns), resource("Namespace", "dynamo-sim")]
    items += workload(
        "etcd", ns, "registry.k8s.io/etcd:3.6.4-0",
        [("client", 2379, None)],
        command=["/usr/local/bin/etcd"],
        args=["--name=etcd", "--data-dir=/data",
              "--initial-cluster=etcd=http://localhost:2380",
              "--listen-client-urls=http://0.0.0.0:2379",
              "--advertise-client-urls=http://etcd.dynamo-system.svc.cluster.local:2379"],
        storage=("2Gi", "/data"), health=(2379, "/health"),
    )
    items += workload(
        "nats", ns, "nats:2.11.4",
        [("client", 4222, None), ("monitor", 8222, None)],
        args=["-js", "-sd", "/data", "-m", "8222"],
        storage=("1Gi", "/data"), health=(8222, "/healthz"),
    )
    items += workload(
        "nats-exporter", ns, "natsio/prometheus-nats-exporter:0.17.3",
        [("metrics", 7777, None)],
        args=["-varz", "-connz", "-jsz=all", "http://nats:8222"],
    )
    return items


def observability():
    ns = "observability"
    items = [resource("Namespace", ns),
             resource("ServiceAccount", "otel-collector", ns)]
    for source_namespace in ("dynamo-system", "dynamo-sim"):
        role = resource(
            "Role", "otel-discovery", source_namespace,
            rules=[{"apiGroups": [""], "resources": ["pods"],
                    "verbs": ["get", "list", "watch"]}],
        )
        binding = resource(
            "RoleBinding", "otel-discovery", source_namespace,
            roleRef={"apiGroup": "rbac.authorization.k8s.io",
                     "kind": "Role", "name": "otel-discovery"},
            subjects=[{"kind": "ServiceAccount", "name": "otel-collector",
                       "namespace": ns}],
        )
        role["apiVersion"] = binding["apiVersion"] = "rbac.authorization.k8s.io/v1"
        items.extend([role, binding])
    metrics_role = resource(
        "ClusterRole", "dynamo-sim-otel-metrics",
        rules=[{"nonResourceURLs": ["/metrics"], "verbs": ["get"]}],
    )
    metrics_binding = resource(
        "ClusterRoleBinding", "dynamo-sim-otel-metrics",
        roleRef={"apiGroup": "rbac.authorization.k8s.io",
                 "kind": "ClusterRole", "name": "dynamo-sim-otel-metrics"},
        subjects=[{"kind": "ServiceAccount", "name": "otel-collector", "namespace": ns}],
    )
    metrics_role["apiVersion"] = metrics_binding["apiVersion"] = "rbac.authorization.k8s.io/v1"
    items.extend([metrics_role, metrics_binding])
    items += workload(
        "otel-collector", ns, "otel/opentelemetry-collector-contrib:0.120.0",
        [("otlp-grpc", 4317, 30317), ("otlp-http", 4318, 30318),
         ("prometheus", 8889, None)],
        args=["--config=/etc/otel-collector.yaml"],
        config=("otel-collector.yaml", "/etc/otel-collector.yaml"),
        health=(13133, "/"), service_account="otel-collector",
        memory="1Gi", memory_request="384Mi",
    )
    items += workload(
        "prometheus", ns, "prom/prometheus:v3.4.1", [("http", 9090, 30909)],
        args=["--config.file=/etc/prometheus.yaml", "--storage.tsdb.path=/prometheus",
              "--storage.tsdb.retention.time=48h", "--storage.tsdb.retention.size=3GB"],
        config=("prometheus.yaml", "/etc/prometheus.yaml"),
        storage=("4Gi", "/prometheus"), health=(9090, "/-/ready"), memory="1Gi",
    )
    items += workload(
        "tempo", ns, "grafana/tempo:2.8.2",
        [("http", 3200, None), ("otlp-grpc", 4317, None)],
        args=["-config.file=/etc/tempo.yaml"],
        config=("tempo.yaml", "/etc/tempo.yaml"), storage=("2Gi", "/var/tempo"),
        health=(3200, "/ready"),
    )
    items += workload(
        "loki", ns, "grafana/loki:3.5.0", [("http", 3100, None)],
        args=["-config.file=/etc/loki.yaml"],
        config=("loki.yaml", "/etc/loki.yaml"), storage=("2Gi", "/var/loki"),
        health=(3100, "/ready"),
    )
    items += [
        configmap("grafana-provisioning", ns, {
            "datasources.yaml": "grafana-datasources.yaml",
            "dashboards.yaml": "grafana-dashboards.yaml",
        }),
        configmap("grafana-dashboard", ns, {
            "dynamo-lab.json": "dashboard.json", "dynamo-routing-live.json": "routing-live-dashboard.json",
        }),
    ]
    mounts = [
        ({"name": "datasources", "configMap": {
            "name": "grafana-provisioning",
            "items": [{"key": "datasources.yaml", "path": "datasources.yaml"}]}},
         {"name": "datasources", "mountPath": "/etc/grafana/provisioning/datasources",
          "readOnly": True}),
        ({"name": "dashboard-provider", "configMap": {
            "name": "grafana-provisioning",
            "items": [{"key": "dashboards.yaml", "path": "dashboards.yaml"}]}},
         {"name": "dashboard-provider", "mountPath": "/etc/grafana/provisioning/dashboards",
          "readOnly": True}),
        ({"name": "dashboards", "configMap": {"name": "grafana-dashboard"}},
         {"name": "dashboards", "mountPath": "/var/lib/grafana/dashboards",
          "readOnly": True}),
    ]
    items += workload(
        "grafana", ns, "grafana/grafana:12.2.0", [("http", 3000, 30300)],
        storage=("1Gi", "/var/lib/grafana"), mounts=mounts, health=(3000, "/api/health"),
        env=[
            {"name": "GF_SECURITY_ADMIN_USER", "value": "dynamo"},
            {"name": "GF_SECURITY_ADMIN_PASSWORD",
             "valueFrom": {"secretKeyRef": {"name": "grafana-admin", "key": "password"}}},
            {"name": "GF_USERS_ALLOW_SIGN_UP", "value": "false"},
            {"name": "GF_ANALYTICS_REPORTING_ENABLED", "value": "false"},
            {"name": "GF_ANALYTICS_CHECK_FOR_UPDATES", "value": "false"},
            {"name": "GF_ANALYTICS_CHECK_FOR_PLUGIN_UPDATES", "value": "false"},
        ],
    )
    return items


def load_lab_config(path=None):
    path = Path(path) if path is not None else HERE / "lab-config.json"
    config = json.loads(path.read_text())
    required = {
        "kv_blocks_per_worker", "kv_block_size", "publish_kv_events",
        "kv_generation_ms_per_1k_tokens", "context_length",
        "worker_processes_per_role", "ranks_per_process",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f"{path}: expected exactly these settings: {sorted(required)}")
    for name in ("kv_blocks_per_worker", "kv_block_size", "context_length",
                 "worker_processes_per_role", "ranks_per_process"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(config["publish_kv_events"]) is not bool:
        raise ValueError("publish_kv_events must be a boolean")
    delay = config["kv_generation_ms_per_1k_tokens"]
    if (type(delay) not in (float, int) or not math.isfinite(delay) or delay < 0):
        raise ValueError("kv_generation_ms_per_1k_tokens must be finite and nonnegative")
    return config


def dynamo():
    config = load_lab_config()
    ranks = config["ranks_per_process"]
    processes = config["worker_processes_per_role"]
    env = [
        {"name": "DYN_EVENT_PLANE", "value": "nats"},
        {"name": "DYN_REQUEST_PLANE", "value": "tcp"},
        {"name": "OTEL_EXPORT_ENABLED", "value": "true"},
        {"name": "OTEL_EXPORTER_OTLP_ENDPOINT",
         "value": "http://otel-collector.observability.svc.cluster.local:4317"},
        {"name": "OTEL_EXPORTER_OTLP_PROTOCOL", "value": "grpc"},
        {"name": "OTEL_RESOURCE_ATTRIBUTES",
         "value": "deployment.environment.name=local-simulation"},
    ]
    image = os.environ.get("DYNAMO_SIM_IMAGE", "fake-sglang-dynamo:1.6.0-fpm")
    model = "/opt/fake-sglang/models/qwen3-0.6b"
    components = [{
        "name": "Frontend", "type": "frontend", "replicas": 1,
        "podTemplate": {"spec": {"containers": [{
            "name": "main", "image": image, "imagePullPolicy": "Never",
            "command": ["python", "-m", "dynamo.frontend"],
            "args": ["--http-port", "8000", "--router-mode", "kv", "--router-kv-events",
                     "--active-decode-blocks-threshold", "0.85",
                     "--router-track-output-blocks"],
            "env": env + [{"name": "OTEL_SERVICE_NAME", "value": "dynamo-frontend"}],
            "resources": {"requests": {"cpu": "100m", "memory": "256Mi"},
                          "limits": {"cpu": "2", "memory": "1Gi"}},
        }]}}
    }]
    for role in ("prefill", "decode"):
        components.append({
            "name": role, "type": role, "replicas": processes,
            "podTemplate": {
                "metadata": {"annotations": {"dynamo-sim.local/engine": "fake-sglang"}},
                "spec": {
                    "nodeSelector": {"dynamo-sim.local/gpu-capacity": "simulated"},
                    "containers": [{
                        "name": "main", "image": image, "imagePullPolicy": "Never",
                        "command": ["python", "-m", "dynamo.sglang"],
                        "args": [
                            "--model-path", model, "--served-model-name", "fake-model",
                            "--context-length", str(config["context_length"]),
                            "--skip-tokenizer-init", "--host", "0.0.0.0",
                            "--disaggregation-mode", role,
                            "--router-mode", "kv", "--router-kv-events",
                            "--disaggregation-bootstrap-port", "8998",
                            "--max-prefill-tokens", "8192",
                            "--engine-request-limit", str(8 * ranks),
                            "--max-running-requests", str(8 * ranks),
                            "--dp-size", str(ranks),
                            *(["--enable-dp-attention"] if ranks > 1 else []),
                            "--page-size", str(config["kv_block_size"]),
                            "--fake-num-kv-blocks", str(config["kv_blocks_per_worker"]),
                            "--fake-kv-generation-ms-per-1k-tokens",
                            str(config["kv_generation_ms_per_1k_tokens"]),
                            *([
                                "--kv-events-config",
                                '{"publisher":"zmq","endpoint":"tcp://*:5557"}',
                            ] if config["publish_kv_events"] else []),
                            "--enable-metrics",
                            "--enable-forward-pass-metrics",
                        ],
                        "env": env + [
                            {"name": "OTEL_SERVICE_NAME", "value": "dynamo-" + role},
                            {"name": "DYN_DYNAMO_REQUEST_QUEUE_LIMIT", "value": str(2 * ranks)},
                            {"name": "SGLANG_HOST_IP",
                             "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}},
                        ],
                        "resources": {
                            "requests": {"cpu": "100m", "memory": "256Mi",
                                         "nvidia.com/gpu": "1"},
                            "limits": {"cpu": "2", "memory": "1Gi",
                                       "nvidia.com/gpu": "1"},
                        },
                        "readinessProbe": {
                            "httpGet": {
                                "path": "/health", "port": "system", "scheme": "HTTP",
                            },
                            "periodSeconds": 5, "timeoutSeconds": 4,
                            "failureThreshold": 2, "successThreshold": 1,
                        },
                        "livenessProbe": {
                            "httpGet": {
                                "path": "/live", "port": "system", "scheme": "HTTP",
                            },
                            "periodSeconds": 30, "timeoutSeconds": 10,
                            "failureThreshold": 3, "successThreshold": 1,
                        },
                    }],
                },
            },
        })
    policy = configmap(
        "lab-router-policy", "dynamo-sim", {"router-policy.yaml": "router-policy.yaml"},
    )
    policy["data"]["router-policy.yaml"] = policy["data"]["router-policy.yaml"].format(
        request_queue_limit=10 * ranks, raw_token_queue_limit=65536 * ranks,
    )
    policy_digest = hashlib.sha256(
        policy["data"]["router-policy.yaml"].encode()
    ).hexdigest()
    for component in components:
        annotations = component["podTemplate"].setdefault("metadata", {}).setdefault(
            "annotations", {}
        )
        annotations["dynamo-sim.local/router-policy-sha256"] = policy_digest
        pod = component["podTemplate"]["spec"]
        container = pod["containers"][0]
        container["args"] += ["--router-policy-config", "/etc/lab-router/router-policy.yaml"]
        container["volumeMounts"] = [
            {"name": "router-policy", "mountPath": "/etc/lab-router", "readOnly": True},
        ]
        pod["volumes"] = [
            {"name": "router-policy", "configMap": {"name": "lab-router-policy"}},
        ]
    planner_config = {
        "environment": "kubernetes", "backend": "sglang", "mode": "disagg",
        "model_name": "fake-model", "optimization_target": "load",
        "pre_deployment_sweeping_mode": "none",
        "enable_throughput_scaling": False, "enable_load_scaling": True,
        "prefill_scale_up_queue_tokens": 8192,
        "prefill_scale_down_queue_tokens": 1024,
        "decode_scale_up_kv_rate": 70, "decode_scale_down_kv_rate": 20,
        "min_endpoint": processes, "max_gpu_budget": 2 * processes,
        "load_adjustment_interval_seconds": 10,
        "load_predictor": "constant",
    }
    components.append({
        "name": "Planner", "type": "planner", "replicas": 1,
        "podTemplate": {"spec": {"containers": [{
            "name": "main",
            "image": "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.6.0",
            "command": ["python3", "-m", "dynamo.planner"],
            "args": ["--config", json.dumps(planner_config)],
            "env": env + [{"name": "OTEL_SERVICE_NAME", "value": "dynamo-planner"}],
            "resources": {"requests": {"cpu": "100m", "memory": "512Mi"},
                          "limits": {"cpu": "2", "memory": "2Gi"}},
        }]}},
    })
    dgd = resource(
        "DynamoGraphDeployment", "fake-sglang", "dynamo-sim",
        spec={"components": components},
    )
    dgd["apiVersion"] = "nvidia.com/v1beta1"
    frontend_service = resource(
        "Service", "inference", "dynamo-sim",
        spec={"type": "NodePort",
              "selector": {"nvidia.com/dynamo-graph-deployment-name": "fake-sglang",
                           "nvidia.com/dynamo-component-type": "frontend"},
              "ports": [{"name": "http", "port": 8000, "targetPort": 8000,
                         "nodePort": 30800}]},
    )
    return [dgd, frontend_service, policy]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["infrastructure", "observability", "dynamo"])
    section = parser.parse_args().section
    items = {"infrastructure": infrastructure, "observability": observability,
             "dynamo": dynamo}[section]()
    print(json.dumps({"apiVersion": "v1", "kind": "List", "items": items}, indent=2))
