from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import tiktoken
import torch


DATASETS = {
    "wikitext2": {
        "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "val": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
    },
    "tiny_shakespeare": {
        "train": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "val": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "test": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    },
}


class TokenDataset:
    def __init__(self, data_dir: str, dataset_name: str, seed: int = 1337):
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {sorted(DATASETS)}")
        self.dataset_name = dataset_name
        self.root = Path(data_dir) / dataset_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder = tiktoken.get_encoding("gpt2")
        self._download_files()
        self.tokens = {split: self._load_tokens(split) for split in ("train", "val", "test")}
        self.generators = {
            "train": torch.Generator().manual_seed(seed + 1),
            "val": torch.Generator().manual_seed(seed + 2),
            "test": torch.Generator().manual_seed(seed + 3),
        }

    @property
    def vocab_size(self) -> int:
        return self.encoder.n_vocab

    def _download_files(self) -> None:
        for split, url in DATASETS[self.dataset_name].items():
            path = self.root / f"{split}.txt"
            if path.exists() and path.stat().st_size > 0:
                continue
            print(f"Downloading {self.dataset_name} {split} split...")
            temp_path = path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, temp_path)
            temp_path.replace(path)

    def _load_tokens(self, split: str) -> torch.Tensor:
        path = self.root / f"{split}.txt"
        text = path.read_text(encoding="utf-8")
        encoded = self.encoder.encode(text, allowed_special=set())
        if len(encoded) < 2:
            raise ValueError(f"Dataset split {path} contains fewer than two tokens")
        return torch.tensor(encoded, dtype=torch.long)

    def metadata(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "tokenizer": "gpt2",
            "vocab_size": self.vocab_size,
            "token_counts": {split: int(tokens.numel()) for split, tokens in self.tokens.items()},
            "sha256": {
                split: hashlib.sha256((self.root / f"{split}.txt").read_bytes()).hexdigest()
                for split in ("train", "val", "test")
            },
        }

    def save_metadata(self) -> None:
        (self.root / "metadata.json").write_text(json.dumps(self.metadata(), indent=2) + "\n")

    def get_batch(self, split: str, batch_size: int, block_size: int, device: torch.device):
        tokens = self.tokens[split]
        if tokens.numel() <= block_size + 1:
            raise ValueError(f"Split {split} is too short for block_size={block_size}")
        starts = torch.randint(
            0,
            tokens.numel() - block_size,
            (batch_size,),
            generator=self.generators[split],
        )
        x = torch.stack([tokens[start : start + block_size] for start in starts])
        y = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts])
        return x.to(device), y.to(device)

    def get_fixed_batch(self, split: str, batch_index: int, batch_size: int, block_size: int, device: torch.device):
        """Return a deterministic, non-random validation batch.

        Validation uses these fixed windows so repeated evaluations and all model
        variants see exactly the same tokens independent of generator state.
        """
        tokens = self.tokens[split]
        if tokens.numel() <= block_size + 1:
            raise ValueError(f"Split {split} is too short for block_size={block_size}")
        window_count = tokens.numel() - block_size
        offsets = torch.arange(batch_size, dtype=torch.long)
        starts = ((batch_index * batch_size + offsets) * block_size) % window_count
        x = torch.stack([tokens[start : start + block_size] for start in starts])
        y = torch.stack([tokens[start + 1 : start + block_size + 1] for start in starts])
        return x.to(device), y.to(device)
