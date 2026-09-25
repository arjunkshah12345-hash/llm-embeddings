import json

from aggregate_interventions import summarize_study


def test_intervention_aggregation_is_paired(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    manifest = {
        "common": {"path_ablation": "stop_input", "ablation_start": 0, "ablation_end": 5},
        "seeds": [1337],
        "embedding_types": ["tied", "partial"],
    }
    (study / "study_manifest.json").write_text(json.dumps(manifest))
    (study / "study_validation.json").write_text(json.dumps({"passed": True}))
    (study / "artifact_manifest.json").write_text(json.dumps({"git_commit": "test"}))
    for condition, loss in (("tied", 2.0), ("partial", 2.1)):
        run = study / f"seed1337_{condition}"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({"train": {"seed": 1337, "embedding_type": condition}}))
        rows = [{"step": 0, "split": "val", "loss": loss + 0.2}, {"step": 5, "split": "val", "loss": loss}]
        (run / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = summarize_study(study)
    comparison = result["paired_comparisons"]["partial_minus_tied_final_val_loss"]
    assert comparison["deltas"] == [0.10000000000000009]
    assert comparison["mean_delta"] == 0.10000000000000009
