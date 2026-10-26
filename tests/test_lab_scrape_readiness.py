import yaml

from deploy.local import render


def test_collector_memory_limiter_matches_container_headroom():
    config = yaml.safe_load((render.HERE / "otel-collector.yaml").read_text())
    limiter = config["processors"]["memory_limiter"]
    assert limiter["limit_mib"] == 768
    assert limiter["spike_limit_mib"] == 256


def test_dynamo_scrapes_wait_for_ready_pods():
    config = yaml.safe_load((render.HERE / "otel-collector.yaml").read_text())
    jobs = config["receivers"]["prometheus"]["config"]["scrape_configs"]
    job = next(job for job in jobs if job["job_name"] == "dynamo")
    rule = next(
        rule for rule in job["relabel_configs"]
        if rule["source_labels"] == ["__meta_kubernetes_pod_ready"]
    )
    assert rule["action"] == "keep"
    assert rule["regex"] == "true"


def test_routing_scrapes_target_only_running_experiment_pods_and_distinct_ports():
    config = yaml.safe_load((render.HERE / "otel-collector.yaml").read_text())
    jobs = config["receivers"]["prometheus"]["config"]["scrape_configs"]
    routing = [job for job in jobs if job["job_name"].startswith("routing-live-")]
    assert {job["job_name"] for job in routing} == {
        "routing-live-prefill-0", "routing-live-prefill-1",
        "routing-live-decode-2", "routing-live-frontend",
    }
    for job in routing:
        rules = job["relabel_configs"]
        assert rules[0] == {
            "source_labels": ["__meta_kubernetes_pod_label_app", "__meta_kubernetes_pod_phase"],
            "regex": "fake-routing-test;Running", "action": "keep",
        }
        ports = {rule["regex"]: rule["replacement"] for rule in rules
                 if rule.get("target_label") == "__tmp_port"}
        assert ports == {
            "routing-live-prefill-0": "9201", "routing-live-prefill-1": "9202",
            "routing-live-decode-2": "9203", "routing-live-frontend": "8000",
        }
        assert job["kubernetes_sd_configs"][0]["namespaces"]["names"] == ["dynamo-sim"]
