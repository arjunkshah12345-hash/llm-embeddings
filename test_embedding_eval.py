import torch

from embedding_eval import (
    evaluate_embedding_matrix,
    nearest_centroid_probe,
    nearest_neighbor_statistics,
    pair_similarity,
    quantile_bucket_labels,
    ridge_linear_probe,
)


def test_quantile_buckets_are_deterministic():
    labels = quantile_bucket_labels(torch.tensor([4.0, 1.0, 3.0, 2.0]))
    assert labels.tolist() == [3, 0, 2, 1]


def test_nearest_centroid_probe_beats_majority_on_separable_vectors():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    result = nearest_centroid_probe(embeddings, labels, torch.tensor([True, False, True, False]))
    assert result["accuracy"] == 1.0
    assert result["macro_accuracy"] == 1.0
    assert result["majority_baseline"] == 0.5


def test_ridge_linear_probe_is_deterministic_with_frozen_embeddings():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1], [1.0, 0.2], [-1.0, -0.2]])
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    train_mask = torch.tensor([True, True, True, True, False, False])

    first = ridge_linear_probe(embeddings, labels, train_mask)
    second = ridge_linear_probe(embeddings, labels, train_mask)

    assert first == second
    assert first["accuracy"] == 1.0
    assert first["train_count"] == 4
    assert first["test_count"] == 2


def test_nearest_neighbor_statistics_reports_bucket_agreement():
    embeddings = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    result = nearest_neighbor_statistics(
        embeddings,
        torch.tensor([10, 20, 30, 40]),
        torch.tensor([0, 0, 1, 1]),
        max_tokens=4,
        neighbors=1,
    )
    assert result["token_count"] == 4
    assert result["mean_frequency_bucket_agreement"] == 1.0


def test_pair_similarity_groups_predeclared_relations():
    class Encoder:
        def encode(self, value, allowed_special):
            return [int(value)]

    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    result = pair_similarity(
        embeddings,
        [
            {"left": "0", "right": "1", "relation": "same"},
            {"left_token_id": 0, "right_token_id": 2, "relation": "different"},
        ],
        Encoder(),
    )
    assert result["pair_count"] == 2
    assert result["mean_cosine_by_relation"]["same"] == 1.0
    assert result["mean_cosine_by_relation"]["different"] == 0.0


def test_embedding_matrix_evaluation_supports_output_side_equally():
    result = evaluate_embedding_matrix(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        type("Encoder", (), {"decode_single_token_bytes": lambda self, token_id: b" token"})(),
        max_tokens=5,
        neighbors=1,
    )
    assert result["active_token_count"] == 5
    assert "token_shape_probe" in result
    assert "token_shape_linear_probe" in result
