from analyze import bootstrap_mean_ci, paired_comparisons


def test_bootstrap_mean_ci_is_deterministic_and_contains_sample_mean():
    values = [1.0, 2.0, 4.0, 8.0]
    first = bootstrap_mean_ci(values, seed=123, samples=500)
    second = bootstrap_mean_ci(values, seed=123, samples=500)

    assert first == second
    assert first["low"] <= sum(values) / len(values) <= first["high"]
    assert bootstrap_mean_ci([3.0], seed=123) == {"low": 3.0, "high": 3.0}


def test_paired_comparisons_use_shared_seed_deltas():
    rows = []
    for seed, tied, partial, untied in [(1, 10.0, 9.8, 9.9), (2, 9.0, 8.9, 9.2)]:
        for kind, value in [("tied", tied), ("partial", partial), ("untied", untied)]:
            rows.append(
                {
                    "seed": seed,
                    "embedding_type": kind,
                    "best_val_loss": value,
                    "final_val_loss": value + 0.1,
                }
            )

    comparisons = paired_comparisons(rows)

    partial = comparisons["partial_minus_tied_best_val_loss"]
    assert partial["seed_count"] == 2
    assert abs(partial["mean_delta"] + 0.15) < 1e-9
    assert partial["ci95"]["low"] <= -0.15 <= partial["ci95"]["high"]


def test_paired_comparisons_include_parameter_matched_controls():
    rows = []
    for seed, tied, control in [(1, 10.0, 9.95), (2, 9.0, 9.1)]:
        rows.extend(
            [
                {"seed": seed, "embedding_type": "tied", "best_val_loss": tied, "final_val_loss": tied},
                {"seed": seed, "embedding_type": "capacity_control", "best_val_loss": control, "final_val_loss": control},
            ]
        )
    comparison = paired_comparisons(rows)["capacity_control_minus_tied_best_val_loss"]
    assert comparison["seed_count"] == 2
    assert abs(comparison["mean_delta"] - 0.025) < 1e-9
