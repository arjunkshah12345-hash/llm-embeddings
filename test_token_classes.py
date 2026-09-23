import torch

from token_classes import assign_classes, dataset_frequency_bucket_ids


def test_assign_classes_separates_surface_form_from_frequency():
    texts = [" ", ".", "the", "zygote", "middle"]
    counts = [100, 50, 80, 1, 5]
    assert assign_classes(texts, counts, common_k=1, rare_max_count=1) == [
        "whitespace",
        "punctuation",
        "common",
        "rare",
        "other",
    ]


def test_frequency_buckets_are_deterministic_and_cover_seen_tokens(tmp_path):
    dataset = type("Dataset", (), {})()
    dataset.root = tmp_path
    dataset.tokens = {"train": torch.tensor([0, 0, 0, 1, 1, 2, 3, 4], dtype=torch.long)}
    dataset.vocab_size = 6
    dataset.metadata = lambda: {"sha256": {"train": "fixture"}}
    labels = dataset_frequency_bucket_ids(dataset, bucket_count=4)
    assert labels.shape == (6,)
    assert labels[5].item() == -1
    assert sorted(labels[:5].tolist()) == [0, 0, 1, 2, 3]
    assert torch.equal(labels, dataset_frequency_bucket_ids(dataset, bucket_count=4))
