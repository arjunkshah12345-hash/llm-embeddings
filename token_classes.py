"""Token-type labels for mechanism checks.

Whitespace and punctuation are decided from the decoded surface form.
Among the remaining tokens, the `common_k` highest-count tokens are common,
tokens seen at most `rare_max_count` times are rare, and the rest are other.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

TOKEN_CLASSES = ("whitespace", "punctuation", "common", "rare", "other")


def surface_class(text: str) -> str | None:
    if text != "" and text.strip() == "":
        return "whitespace"
    if text and all(not character.isalnum() and not character.isspace() for character in text):
        return "punctuation"
    return None


def assign_classes(texts: list[str], counts, common_k: int = 1000, rare_max_count: int = 1) -> list[str]:
    if len(texts) != len(counts):
        raise ValueError("texts and counts must have the same length")
    if common_k < 0 or rare_max_count < 0:
        raise ValueError("common_k and rare_max_count must be non-negative")
    labels = ["other"] * len(texts)
    frequency_ids = []
    for index, text in enumerate(texts):
        surface = surface_class(text)
        if surface is not None:
            labels[index] = surface
        else:
            frequency_ids.append(index)
    ranked = sorted(frequency_ids, key=lambda index: int(counts[index]), reverse=True)
    common = set()
    for index in ranked:
        if int(counts[index]) <= rare_max_count or len(common) >= common_k:
            break
        common.add(index)
    for index in frequency_ids:
        if index in common:
            labels[index] = "common"
        elif int(counts[index]) <= rare_max_count:
            labels[index] = "rare"
    return labels


def class_id_tensor(labels: list[str]) -> torch.Tensor:
    mapping = {name: index for index, name in enumerate(TOKEN_CLASSES)}
    unknown = sorted({label for label in labels if label not in mapping})
    if unknown:
        raise ValueError(f"unknown token classes: {unknown}")
    return torch.tensor([mapping[label] for label in labels], dtype=torch.long)


def dataset_class_ids(dataset, common_k: int = 1000, rare_max_count: int = 1) -> torch.Tensor:
    """Label every GPT-2 token from the dataset's training counts. Cached beside the corpus."""
    meta = {
        "common_k": common_k,
        "rare_max_count": rare_max_count,
        "tokenizer": "gpt2",
        "train_sha256": dataset.metadata()["sha256"]["train"],
        "classes": list(TOKEN_CLASSES),
    }
    cache = Path(dataset.root) / "token_classes.pt"
    stamp = Path(dataset.root) / "token_classes.meta.json"
    if cache.exists() and stamp.exists() and json.loads(stamp.read_text()) == meta:
        return torch.load(cache, map_location="cpu", weights_only=True)
    counts = torch.bincount(dataset.tokens["train"], minlength=dataset.vocab_size)
    texts = [dataset.encoder.decode([token_id]) for token_id in range(dataset.vocab_size)]
    labels = assign_classes(texts, counts, common_k=common_k, rare_max_count=rare_max_count)
    class_ids = class_id_tensor(labels)
    torch.save(class_ids, cache)
    stamp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return class_ids
