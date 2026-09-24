from aggregate_embedding_eval import summarize


def test_embedding_eval_summary_keeps_input_and_output_separate():
    records = [
        {"seed": 1, "condition": "partial", "side": "input", "neighbor_cosine": 0.3},
        {"seed": 2, "condition": "partial", "side": "input", "neighbor_cosine": 0.5},
        {"seed": 1, "condition": "partial", "side": "output", "neighbor_cosine": 0.7},
        {"seed": 2, "condition": "partial", "side": "output", "neighbor_cosine": 0.9},
    ]
    result = summarize(records, {"conditions": ["partial"], "seeds": [1, 2], "git_commit": "abc"})
    assert result["final"]["partial"]["input"]["metrics"]["neighbor_cosine"]["mean"] == 0.4
    assert result["final"]["partial"]["output"]["metrics"]["neighbor_cosine"]["mean"] == 0.8
