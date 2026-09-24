import json

from aggregate_results import validate_and_load


def test_aggregate_requires_and_reports_collected_artifact_commit(tmp_path):
    source = tmp_path / "study"
    source.mkdir()
    (source / "study_validation.json").write_text(json.dumps({"passed": True}))
    (source / "study_manifest.json").write_text(
        json.dumps(
            {
                "common": {"steps": 3},
                "seeds": [1337],
                "embedding_types": ["tied"],
            }
        )
    )
    (source / "artifact_manifest.json").write_text(json.dumps({"git_commit": "abc123"}))
    run = source / "run"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "model": {"block_size": 8},
                "train": {"seed": 1337},
            }
        )
    )
    (run / "parameter_counts.json").write_text(
        json.dumps(
            {
                "embedding_type": "tied",
                "total_parameters": 10,
                "trainable_parameters": 10,
                "embedding_parameters": 4,
                "transformer_parameters": 6,
            }
        )
    )
    (run / "metrics.jsonl").write_text(
        json.dumps({"split": "val", "step": 2, "loss": 1.0, "perplexity": 2.718, "tokens_seen": 24}) + "\n"
    )

    rows, metadata = validate_and_load([source])

    assert len(rows) == 1
    assert metadata["git_commit"] == "abc123"
