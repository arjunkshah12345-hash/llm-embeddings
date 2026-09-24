import json

from aggregate_rank import aggregate, load_rank_rows
import pytest


def test_rank_aggregate_reports_paired_deltas():
    rows = [
        {
            "seed": 1,
            "adapter_rank": 1,
            "adapter_alpha": 8.0,
            "total_parameters": 101,
            "additional_parameters_vs_tied": 1,
            "mean_final_val_loss": 4.0,
            "mean_best_val_loss": 3.9,
            "mean_estimated_flops_total": 10.0,
            "mean_training_wall_time_seconds": 100.0,
        },
        {
            "seed": 2,
            "adapter_rank": 1,
            "adapter_alpha": 8.0,
            "total_parameters": 101,
            "additional_parameters_vs_tied": 1,
            "mean_final_val_loss": 4.2,
            "mean_best_val_loss": 4.1,
            "mean_estimated_flops_total": 14.0,
            "mean_training_wall_time_seconds": 140.0,
        },
    ]
    result = aggregate(
        rows,
        {"seeds": [1, 2], "git_commit": "abc"},
        tied_rows={
            1: {"final_val_loss": 4.1, "best_val_loss": 4.0},
            2: {"final_val_loss": 4.0, "best_val_loss": 3.9},
        },
    )
    rank = result["by_rank"]["1"]
    assert rank["run_count"] == 2
    assert rank["paired_final_delta"]["mean"] == pytest.approx(0.05)
    assert rank["mean_estimated_flops_total"] == pytest.approx(12.0)
    assert rank["mean_training_wall_time_seconds"] == pytest.approx(120.0)


def test_rank_loader_rejects_unvalidated_condition(tmp_path):
    study = tmp_path / "rank"
    study.mkdir()
    (study / "artifact_manifest.json").write_text(json.dumps({"seed": 1, "git_commit": "abc"}))
    condition = study / "rank1"
    condition.mkdir()
    (condition / "study_validation.json").write_text(json.dumps({"passed": False}))
    (study / "adapter_sweep_summary.json").write_text(json.dumps({"conditions": []}))

    with pytest.raises(SystemExit, match="fairness validation failed"):
        load_rank_rows([study])
