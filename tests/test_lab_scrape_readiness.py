import yaml

from deploy.local import render


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
