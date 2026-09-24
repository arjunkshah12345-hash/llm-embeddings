import json

from aggregate_mechanism import summarize


def test_mechanism_summary_aggregates_final_seed_rows():
    records = [
        {"seed": 1, "condition": "tied", "step": 10, "output_to_input_grad_ratio": 2.0},
        {"seed": 2, "condition": "tied", "step": 10, "output_to_input_grad_ratio": 4.0},
        {"seed": 1, "condition": "tied", "step": 5, "output_to_input_grad_ratio": 1.0},
    ]

    result = summarize(records, {"conditions": ["tied"], "seeds": [1, 2], "git_commit": "abc"})

    summary = result["final"]["tied"]["metrics"]["output_to_input_grad_ratio"]
    assert summary["seed_values"] == [2.0, 4.0]
    assert summary["mean"] == 3.0
    assert result["trajectory"]["tied"]["5"]["output_to_input_grad_ratio"] == 1.0


def test_fixed_checkpoint_metrics_override_stochastic_log_endpoint():
    records = [{"seed": 1, "condition": "partial", "step": 10, "output_to_input_grad_ratio": 9.0}]
    fixed = [{"seed": 1, "condition": "partial", "step": 11, "output_to_input_grad_ratio": 2.5}]

    result = summarize(records, {"conditions": ["partial"], "seeds": [1], "git_commit": "abc"}, fixed)

    assert result["final"]["partial"]["metrics"]["output_to_input_grad_ratio"]["seed_values"] == [2.5]
    assert result["trajectory"]["partial"]["10"]["output_to_input_grad_ratio"] == 9.0
