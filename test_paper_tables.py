import json

from paper.generate_tables import primary_table, rank_table


def test_primary_table_is_generated_from_aggregate():
    result = {
        "by_condition": {
            "tied": {
                "total_parameters": 100,
                "additional_parameters_vs_tied": 0,
                "mean_final_val_loss": 1.2,
                "ci95_final_val_loss": {"low": 1.1, "high": 1.3},
                "mean_best_val_loss": 1.0,
                "run_count": 3,
            }
        },
        "research_question": {
            "final_val_loss": {"tied": 1.2, "partial": 1.2, "untied": 1.2, "recovery_status": "uncertain"}
        },
    }
    table = primary_table(result)
    assert "100" in table and "1.20000" in table and "3" in table


def test_primary_table_uses_single_latex_escape_for_condition_names():
    result = {
        "by_condition": {
            "capacity_control": {
                "total_parameters": 100,
                "additional_parameters_vs_tied": 0,
                "mean_final_val_loss": 1.2,
                "ci95_final_val_loss": {"low": 1.1, "high": 1.3},
                "mean_best_val_loss": 1.0,
                "run_count": 1,
            }
        }
    }
    table = primary_table(result)
    assert "capacity\\_control" in table
    assert "capacity\\\\_control" not in table


def test_rank_table_contains_each_rank():
    result = {
        "by_rank": {
            "1": {
                "additional_parameters_vs_tied": 10,
                "mean_final_val_loss": 2.0,
                "ci95_final_val_loss": {"low": 2.0, "high": 2.0},
                "run_count": 1,
            }
        }
    }
    assert "1" in rank_table(result)
