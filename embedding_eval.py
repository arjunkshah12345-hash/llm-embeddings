"""Evaluate effective input embeddings with deterministic intrinsic probes."""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from pathlib import Path

import tiktoken
import torch

from config import ModelConfig, TrainConfig
from data import TokenDataset
from model import GPTModel
from train import choose_device, set_seed


def quantile_bucket_labels(values: torch.Tensor, bucket_count: int = 4) -> torch.Tensor:
    """Assign deterministic rank-quantile labels, breaking ties by index."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty vector")
    if bucket_count < 2:
        raise ValueError("bucket_count must be at least 2")
    order = sorted(range(values.numel()), key=lambda index: (float(values[index]), index))
    labels = torch.empty(values.numel(), dtype=torch.long)
    for rank, index in enumerate(order):
        labels[index] = min(bucket_count - 1, rank * bucket_count // values.numel())
    return labels


def nearest_centroid_probe(embeddings: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor) -> dict:
    """Evaluate a fixed nearest-centroid classifier without learned probe parameters."""
    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.size(0) != labels.numel():
        raise ValueError("embeddings and labels have incompatible shapes")
    if train_mask.dtype is not torch.bool or train_mask.shape != labels.shape:
        raise ValueError("train_mask must be a boolean vector matching labels")
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError("probe requires both training and test examples")
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=1)
    classes = torch.unique(labels[train_mask], sorted=True)
    centroids = torch.stack(
        [torch.nn.functional.normalize(normalized[train_mask & (labels == label)].mean(dim=0, keepdim=True), dim=1)[0] for label in classes]
    )
    predictions = classes[(normalized[test_mask] @ centroids.T).argmax(dim=1)]
    targets = labels[test_mask]
    accuracy = (predictions == targets).float().mean().item()
    counts = torch.bincount(labels[train_mask], minlength=int(classes.max().item()) + 1)
    test_counts = torch.bincount(labels[test_mask], minlength=int(classes.max().item()) + 1)
    majority = test_counts.max().item() / max(int(test_mask.sum().item()), 1)
    return {
        "accuracy": accuracy,
        "majority_baseline": majority,
        "train_count": int(train_mask.sum().item()),
        "test_count": int(test_mask.sum().item()),
        "class_count": int(classes.numel()),
    }


def nearest_neighbor_statistics(
    embeddings: torch.Tensor,
    token_ids: torch.Tensor,
    bucket_labels: torch.Tensor,
    max_tokens: int = 512,
    neighbors: int = 5,
    chunk_size: int = 128,
) -> dict:
    """Measure local cosine neighbors and frequency-bucket agreement."""
    if embeddings.ndim != 2 or token_ids.ndim != 1 or bucket_labels.ndim != 1:
        raise ValueError("embeddings, token_ids, and bucket_labels must be vectors/matrix")
    if token_ids.numel() != bucket_labels.numel():
        raise ValueError("token_ids and bucket_labels must have equal length")
    if token_ids.numel() < 2:
        raise ValueError("nearest-neighbor statistics require at least two tokens")
    if max_tokens <= 0 or neighbors <= 0:
        raise ValueError("max_tokens and neighbors must be positive")
    if token_ids.numel() > max_tokens:
        bucket_count = max(1, int(bucket_labels.max().item()) + 1)
        quota = max(1, math.ceil(max_tokens / bucket_count))
        selected_positions = torch.cat(
            [torch.where(bucket_labels == bucket)[0][:quota] for bucket in range(bucket_count)]
        )[:max_tokens]
    else:
        selected_positions = torch.arange(token_ids.numel(), dtype=torch.long)
    selected = token_ids[selected_positions]
    normalized = torch.nn.functional.normalize(embeddings.float(), dim=1)
    top_k = min(neighbors, embeddings.size(0) - 1)
    agreements = []
    similarities = []
    for start in range(0, selected_positions.numel(), chunk_size):
        positions = selected_positions[start : start + chunk_size]
        scores = normalized[positions] @ normalized.T
        row_indices = torch.arange(positions.numel())
        scores[row_indices, positions] = -torch.inf
        values, indices = scores.topk(top_k, dim=1)
        agreements.append((bucket_labels[indices] == bucket_labels[positions, None]).float().mean(dim=1))
        similarities.append(values.mean(dim=1))
    agreement = torch.cat(agreements)
    similarity = torch.cat(similarities)
    return {
        "token_count": int(selected.numel()),
        "neighbors": top_k,
        "mean_frequency_bucket_agreement": agreement.mean().item(),
        "mean_neighbor_cosine": similarity.mean().item(),
    }


def token_shape_labels(encoder, token_ids: torch.Tensor) -> torch.Tensor:
    labels = []
    for token_id in token_ids.tolist():
        token_bytes = encoder.decode_single_token_bytes(int(token_id))
        text = token_bytes.decode("utf-8", errors="replace")
        if text.isspace():
            labels.append(0)
        elif all(unicodedata.category(char).startswith("P") for char in text if char):
            labels.append(1)
        elif len(token_bytes) <= 2:
            labels.append(2)
        else:
            labels.append(3)
    return torch.tensor(labels, dtype=torch.long)


def evaluate_input_embeddings(
    embedding_matrix: torch.Tensor,
    train_tokens: torch.Tensor,
    encoder,
    max_tokens: int = 512,
    neighbors: int = 5,
) -> dict:
    counts = torch.bincount(train_tokens, minlength=embedding_matrix.size(0)).float()
    active = counts > 0
    token_ids = torch.where(active)[0]
    embeddings = embedding_matrix[token_ids]
    frequencies = counts[token_ids]
    frequency_labels = quantile_bucket_labels(torch.log1p(frequencies), bucket_count=4)
    train_mask = token_ids.remainder(5).ne(4)
    shape_labels = token_shape_labels(encoder, token_ids)
    return {
        "active_token_count": int(token_ids.numel()),
        "frequency_bucket_probe": nearest_centroid_probe(embeddings, frequency_labels, train_mask),
        "token_shape_probe": nearest_centroid_probe(embeddings, shape_labels, train_mask),
        "nearest_neighbors": nearest_neighbor_statistics(
            embeddings,
            token_ids,
            frequency_labels,
            max_tokens=max_tokens,
            neighbors=neighbors,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    model_config = ModelConfig(**saved["model"])
    train_config = TrainConfig(**saved["train"])
    set_seed(train_config.seed)
    device = choose_device(args.device)
    dataset = TokenDataset(train_config.data_dir, train_config.dataset, train_config.seed)
    model = GPTModel(model_config, train_config.embedding_type, train_config.seed).to(device)
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        input_matrix = model.embeddings.weight("input").detach().cpu()
    encoder = tiktoken.get_encoding("gpt2")
    result = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "step": checkpoint.get("step"),
        "embedding_type": train_config.embedding_type,
        "parameter_counts": model.parameter_counts(),
        "probe_split": "token_id modulo 5; remainder 4 held out",
        "metrics": evaluate_input_embeddings(
            input_matrix,
            dataset.tokens["train"],
            encoder,
            max_tokens=args.max_tokens,
            neighbors=args.neighbors,
        ),
    }
    output = Path(args.output) if args.output else run_dir / "embedding_eval.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
