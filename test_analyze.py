from analyze import paired_comparisons


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
