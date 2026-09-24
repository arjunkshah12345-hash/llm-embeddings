import json

from aggregate_results import summarize, validate_and_load


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


def test_recovery_is_not_reported_when_untied_is_worse():
    rows = []
    for seed in (1337, 2027):
        for condition, loss, params in (
            ("tied", 5.0, 100),
            ("untied", 5.2, 200),
            ("partial", 5.1, 110),
        ):
            rows.append(
                {
                    "seed": seed,
                    "embedding_type": condition,
                    "total_parameters": params,
                    "embedding_parameters": params,
                    "final_val_loss": loss,
                    "best_val_loss": loss,
                    "final_val_perplexity": 1.0,
                    "best_val_perplexity": 1.0,
                }
            )

    result = summarize(rows, {"conditions": ["tied", "untied", "partial"], "seeds": [1337, 2027]})

    assert result["research_question"]["final_val_loss"]["recovered_fraction"] is None
    assert result["research_question"]["final_val_loss"]["recovery_status"] == "untied_not_better_than_tied"


def test_summary_reports_runtime_and_compute_metrics():
    rows = []
    for seed, wall, speed in ((1, 10.0, 100.0), (2, 12.0, 80.0)):
        for condition, loss, parameters in (("tied", 5.0, 100), ("partial", 5.1, 110), ("untied", 5.2, 200)):
            rows.append(
                {
                    "seed": seed,
                    "embedding_type": condition,
                    "total_parameters": parameters,
                    "embedding_parameters": parameters,
                    "final_val_loss": loss,
                    "best_val_loss": loss,
                    "final_val_perplexity": 148.4,
                    "best_val_perplexity": 148.4,
                    "training_tokens": 1000,
                    "tokens_per_second": speed,
                    "training_wall_time_seconds": wall,
                    "peak_gpu_memory_mb": 512.0,
                    "estimated_flops_total": 1e12,
                    "estimated_flops_non_embedding": 5e11,
                }
            )
    result = summarize(rows, {"conditions": ["tied", "partial", "untied"], "seeds": [1, 2]})
    tied = result["by_condition"]["tied"]
    assert tied["mean_tokens_per_second"] == 90.0
    assert tied["mean_training_wall_time_seconds"] == 11.0
    assert tied["mean_estimated_flops_total"] == 1e12
