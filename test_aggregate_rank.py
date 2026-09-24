from aggregate_rank import aggregate
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
        },
        {
            "seed": 2,
            "adapter_rank": 1,
            "adapter_alpha": 8.0,
            "total_parameters": 101,
            "additional_parameters_vs_tied": 1,
            "mean_final_val_loss": 4.2,
            "mean_best_val_loss": 4.1,
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
