from aggregate_secondary import aggregate


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
