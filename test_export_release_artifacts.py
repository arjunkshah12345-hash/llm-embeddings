import json

from export_release_artifacts import export_study


def test_export_preserves_metrics_and_digest_without_checkpoints(tmp_path):
    source = tmp_path / "source"
    run = source / "seed1_tied"
    run.mkdir(parents=True)
    (source / "artifact_manifest.json").write_text(
        json.dumps({"experiment_id": "fixture", "git_commit": "abc", "seed": 1})
    )
    (run / "config.json").write_text("{}")
    (run / "manifest.json").write_text(
        json.dumps({"batch_stream": {"digest": "deadbeef", "record_count": 2}})
    )
    (run / "parameter_counts.json").write_text("{}")
    (run / "metrics.jsonl").write_text('{"step": 0}\n')
    (run / "model.pt").write_bytes(b"checkpoint")

    destination = tmp_path / "export"
    metadata = export_study("fixture", source, destination)

    assert metadata["checkpoints_exported"] is False
    assert (destination / "runs/seed1_tied/metrics.jsonl").exists()
    assert json.loads((destination / "runs/seed1_tied/batch_stream_digest.json").read_text())["digest"] == "deadbeef"
    assert not (destination / "runs/seed1_tied/model.pt").exists()
