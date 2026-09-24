from aggregate_secondary import aggregate
import pytest


def test_secondary_aggregate_preserves_study_boundaries():
    result = aggregate(
        {
            "wikitext": {
                "by_condition": {
                    "tied": {
                        "run_count": 3,
                        "total_parameters": 100,
                        "additional_parameters_vs_tied": 0,
                        "mean_final_val_loss": 4.0,
                    }
                }
            },
            "small": {
                "by_condition": {
                    "tied": {
                        "run_count": 3,
                        "total_parameters": 50,
                        "additional_parameters_vs_tied": 0,
                        "mean_final_val_loss": 4.2,
                    }
                }
            },
        }
    )
    assert len(result["comparison"]) == 2
    assert {row["study"] for row in result["comparison"]} == {"wikitext", "small"}


def test_secondary_aggregate_reports_paired_loss_and_perplexity_deltas():
    result = aggregate(
        {
            "wikitext": {
                "runs": [
                    {"seed": 1, "embedding_type": "tied", "final_val_loss": 4.0, "best_val_loss": 3.9, "final_val_perplexity": 54.6, "best_val_perplexity": 49.4},
                    {"seed": 1, "embedding_type": "partial", "final_val_loss": 3.8, "best_val_loss": 3.7, "final_val_perplexity": 44.7, "best_val_perplexity": 40.4},
                ],
                "by_condition": {},
            }
        }
    )
    pair = result["paired_comparisons"]["wikitext"]["partial_minus_tied_final_val_loss"]
    assert pair["mean_delta"] == pytest.approx(-0.2)
    assert result["paired_comparisons"]["wikitext"]["partial_minus_tied_final_val_perplexity"]["mean_delta"] < 0
