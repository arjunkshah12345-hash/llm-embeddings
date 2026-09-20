from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import tiktoken
import torch


WIKITEXT2 = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "val": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# Character fractions over the downloaded source. Contiguous, non-overlapping, deterministic.
TINY_SHAKESPEARE_SPLIT = {"train": 0.90, "val": 0.05, "test": 0.05}
TINY_SHAKESPEARE_SPLIT_VERSION = 1

DATASETS = {
    "wikitext2": WIKITEXT2,
    "tiny_shakespeare": {"source": TINY_SHAKESPEARE_URL},
}


class TokenDataset:
    def __init__(self, data_dir: str, dataset_name: str, seed: int = 1337):
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {sorted(DATASETS)}")
        self.dataset_name = dataset_name
        self.root = Path(data_dir) / dataset_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.encoder = tiktoken.get_encoding("gpt2")
        self._prepare_files()
        self.tokens = {split: self._load_tokens(split) for split in ("train", "val", "test")}
        self.generators = {
            "train": torch.Generator().manual_seed(seed + 1),
            "val": torch.Generator().manual_seed(seed + 2),
            "test": torch.Generator().manual_seed(seed + 3),
        }

    @property
    def vocab_size(self) -> int:
        return self.encoder.n_vocab

    def _prepare_files(self) -> None:
        if self.dataset_name == "tiny_shakespeare":
            self._prepare_tiny_shakespeare()
            return
        for split, url in WIKITEXT2.items():
            path = self.root / f"{split}.txt"
            if path.exists() and path.stat().st_size > 0:
                continue
            print(f"Downloading {self.dataset_name} {split} split...")
            temp_path = path.with_suffix(".tmp")
            urllib.request.urlretrieve(url, temp_path)
            temp_path.replace(path)

    def _prepare_tiny_shakespeare(self) -> None:
        """Download once and write disjoint train/val/test character splits.

        Older layouts copied the same file into all three splits, which leaked
        evaluation text into training. Rebuild when that contamination is detected
        or when the split version stamp is missing.
        """
        source_path = self.root / "input.txt"
        stamp_path = self.root / "split_manifest.json"
        split_paths = {split: self.root / f"{split}.txt" for split in ("train", "val", "test")}
        if not source_path.exists() or source_path.stat().st_size == 0:
            print("Downloading tiny_shakespeare source...")
            temp_path = source_path.with_suffix(".tmp")
            urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, temp_path)
            temp_path.replace(source_path)

        source = source_path.read_text(encoding="utf-8")
        source_sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
        expected_stamp = {
            "version": TINY_SHAKESPEARE_SPLIT_VERSION,
            "source_sha256": source_sha,
            "fractions": TINY_SHAKESPEARE_SPLIT,
        }
        need_rebuild = True
        if stamp_path.exists() and all(path.exists() and path.stat().st_size > 0 for path in split_paths.values()):
            try:
                need_rebuild = json.loads(stamp_path.read_text()) != expected_stamp
            except (OSError, ValueError):
                need_rebuild = True
        if not need_rebuild:
            # Contaminated layout: identical bodies across splits.
            bodies = [path.read_bytes() for path in split_paths.values()]
            if bodies[0] == bodies[1] == bodies[2]:
                need_rebuild = True
        if not need_rebuild:
            return

        length = len(source)
        if length < 3:
            raise ValueError("tiny_shakespeare source is too short to split")
        train_end = int(length * TINY_SHAKESPEARE_SPLIT["train"])
        val_end = train_end + int(length * TINY_SHAKESPEARE_SPLIT["val"])
        # Put the remainder on test so fractions always cover the full source.
        pieces = {
            "train": source[:train_end],
            "val": source[train_end:val_end],
            "test": source[val_end:],
        }
        for split, text in pieces.items():
            if len(text) < 2:
                raise ValueError(f"tiny_shakespeare {split} split is empty; source may be truncated")
            temp_path = split_paths[split].with_suffix(".tmp")
            temp_path.write_text(text, encoding="utf-8")
            temp_path.replace(split_paths[split])
        stamp_path.write_text(json.dumps(expected_stamp, indent=2, sort_keys=True) + "\n")

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
