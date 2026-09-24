from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import tiktoken
import torch


# Pinned public mirror of the WikiText-2 raw files. The hashes lock the exact
# bytes used by a run even if the mirror changes its hosting details later.
WIKITEXT2 = {
    "train": {
        "url": "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.train.raw",
        "sha256": "6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57",
    },
    "val": {
        "url": "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.valid.raw",
        "sha256": "4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7",
    },
    "test": {
        "url": "https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.test.raw",
        "sha256": "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08",
    },
}
WIKITEXT2_VARIANT = "wikitext-2-raw-v1"
WIKITEXT2_SOURCE_MANIFEST_VERSION = 1

TINY_SHAKESPEARE_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
TINY_SHAKESPEARE_VARIANT = "tiny-shakespeare-char-rnn-v1"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    f"{TINY_SHAKESPEARE_COMMIT}/data/tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
# Character fractions over the downloaded source. Contiguous, non-overlapping, deterministic.
TINY_SHAKESPEARE_SPLIT = {"train": 0.90, "val": 0.05, "test": 0.05}
TINY_SHAKESPEARE_SPLIT_VERSION = 1
FIXTURE_VARIANT = "deterministic-fixture-v1"
FIXTURE_SPLIT_TEXT = {
    "train": ("alpha beta gamma delta epsilon zeta eta theta\n" * 256),
    "val": ("iota kappa lambda mu nu xi omicron pi\n" * 64),
    "test": ("rho sigma tau upsilon phi chi psi omega\n" * 64),
}

DATASETS = {
    "wikitext2": WIKITEXT2,
    "tiny_shakespeare": {"source": TINY_SHAKESPEARE_URL},
    "fixture": FIXTURE_SPLIT_TEXT,
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
        if self.dataset_name == "fixture":
            self._prepare_fixture()
            return
        if self.dataset_name == "tiny_shakespeare":
            self._prepare_tiny_shakespeare()
            return
        self._prepare_wikitext2()

    def _prepare_wikitext2(self) -> None:
        source_manifest = self.root / "source_manifest.json"
        expected_manifest = {
            "version": WIKITEXT2_SOURCE_MANIFEST_VERSION,
            "variant": WIKITEXT2_VARIANT,
            "files": WIKITEXT2,
        }
        if source_manifest.exists() and all(
            (self.root / f"{split}.txt").exists() for split in WIKITEXT2
        ):
            try:
                files_match = all(
                    hashlib.sha256((self.root / f"{split}.txt").read_bytes()).hexdigest()
                    == specification["sha256"]
                    for split, specification in WIKITEXT2.items()
                )
                if json.loads(source_manifest.read_text()) == expected_manifest and files_match:
                    return
            except (OSError, ValueError):
                pass

        for split, specification in WIKITEXT2.items():
            path = self.root / f"{split}.txt"
            print(f"Downloading {self.dataset_name} {WIKITEXT2_VARIANT} {split} split...")
            temp_path = path.with_suffix(".tmp")
            urllib.request.urlretrieve(specification["url"], temp_path)
            actual_sha256 = hashlib.sha256(temp_path.read_bytes()).hexdigest()
            if actual_sha256 != specification["sha256"]:
                temp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for {split}: expected {specification['sha256']}, got {actual_sha256}"
                )
            temp_path.replace(path)
        source_manifest.write_text(json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n")

    def _prepare_fixture(self) -> None:
        for split, text in FIXTURE_SPLIT_TEXT.items():
            path = self.root / f"{split}.txt"
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")

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
            actual_sha256 = hashlib.sha256(temp_path.read_bytes()).hexdigest()
            if actual_sha256 != TINY_SHAKESPEARE_SHA256:
                temp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for tiny_shakespeare: expected {TINY_SHAKESPEARE_SHA256}, got {actual_sha256}"
                )
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
        if self.dataset_name == "wikitext2":
            source = {
                "variant": WIKITEXT2_VARIANT,
                "manifest_version": WIKITEXT2_SOURCE_MANIFEST_VERSION,
                "files": WIKITEXT2,
            }
        elif self.dataset_name == "tiny_shakespeare":
            source = {
                "variant": TINY_SHAKESPEARE_VARIANT,
                "url": TINY_SHAKESPEARE_URL,
                "commit": TINY_SHAKESPEARE_COMMIT,
                "sha256": TINY_SHAKESPEARE_SHA256,
                "split_version": TINY_SHAKESPEARE_SPLIT_VERSION,
            }
        else:
            source = {"variant": FIXTURE_VARIANT}
        return {
            "dataset": self.dataset_name,
            "tokenizer": "gpt2",
            "vocab_size": self.vocab_size,
            "source": source,
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
