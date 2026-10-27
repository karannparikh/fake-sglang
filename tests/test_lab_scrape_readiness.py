import yaml

from deploy.local import render


def test_dynamo_scrapes_wait_for_ready_pods():
    config = yaml.safe_load((render.HERE / "otel-collector.yaml").read_text())
    jobs = config["receivers"]["prometheus"]["config"]["scrape_configs"]
    job = next(item for item in jobs if item["job_name"] == "dynamo")
    rule = next(
        item
        for item in job["relabel_configs"]
        if item["source_labels"] == ["__meta_kubernetes_pod_ready"]
    )
    assert rule == {
        "source_labels": ["__meta_kubernetes_pod_ready"],
        "regex": "true",
        "action": "keep",
    }
    assert not any(item["job_name"].startswith("routing-live-") for item in jobs)


def test_unused_telemetry_is_dropped_without_extra_backends():
    config = yaml.safe_load((render.HERE / "otel-collector.yaml").read_text())
    assert set(config["exporters"]) == {"prometheus", "nop"}
    assert config["service"]["pipelines"]["traces"]["exporters"] == ["nop"]
    assert config["service"]["pipelines"]["logs"]["exporters"] == ["nop"]
    assert config["processors"]["memory_limiter"] == {
        "check_interval": "1s",
        "limit_mib": 768,
        "spike_limit_mib": 256,
    }
