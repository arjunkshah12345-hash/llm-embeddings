import torch

from embedding_eval import nearest_centroid_probe, nearest_neighbor_statistics, quantile_bucket_labels


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
