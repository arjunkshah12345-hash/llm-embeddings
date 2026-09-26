from __future__ import annotations

import hashlib
import json
import urllib.request
from array import array
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

# Study 3 uses a pinned, broad corpus snapshot.  The sample contains 10B
# declared tokens; the study consumes only the first 20.48M training tokens,
# so it never cycles through the source stream.  The file manifest below is
# metadata for the immutable v1.0.0 snapshot, not a request to download the
# complete 28GB sample.
FINEWEB_EDU_VARIANT = "fineweb-edu-sample-10BT-v1.0.0"
FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_CONFIG = "sample-10BT"
FINEWEB_EDU_REVISION = "fc9850dff5e2d0f8f776efe41b24a1c49556cfc5"
FINEWEB_EDU_TRAIN_TOKENS = 20_480_000
FINEWEB_EDU_VAL_TOKENS = 262_144
FINEWEB_EDU_TEST_TOKENS = 262_144
FINEWEB_EDU_SOURCE_FILES = [
    {"path": "sample/10BT/000_00000.parquet", "oid": "2df63e03a865d1045dcb477fcdd68635faf226ef", "size": 2152819114},
    {"path": "sample/10BT/001_00000.parquet", "oid": "c2021c8c564a8a3fca7ba114afc9069bd0bcd2d2", "size": 2152222432},
    {"path": "sample/10BT/002_00000.parquet", "oid": "e42928ee6a755554d7f348456ca62a66a9abc08c", "size": 2151796315},
    {"path": "sample/10BT/003_00000.parquet", "oid": "2c9b4a5357740163228979d4887b6639977b650f", "size": 2152437524},
    {"path": "sample/10BT/004_00000.parquet", "oid": "29f28906fd7457a4bad8e8181c9432ac90aac030", "size": 2152338550},
    {"path": "sample/10BT/005_00000.parquet", "oid": "92cc8d92e5e4cc710a53a14c4999373152b83e41", "size": 2152189947},
    {"path": "sample/10BT/006_00000.parquet", "oid": "fb75f6c578c1f731b79ec5982e462e75931a6954", "size": 2152689867},
    {"path": "sample/10BT/007_00000.parquet", "oid": "a9d5d796ea550171f13e71995dfd30a77c766437", "size": 2150686637},
    {"path": "sample/10BT/008_00000.parquet", "oid": "2535d7e76c7972978eb48a8a8ca4d8b6a718fea1", "size": 2151274846},
    {"path": "sample/10BT/009_00000.parquet", "oid": "a9fcab0229d213864efab5812255f3ff272fc6d4", "size": 2151913277},
    {"path": "sample/10BT/010_00000.parquet", "oid": "2c1203939e5139226910db563e21fdf754974e65", "size": 2152798864},
    {"path": "sample/10BT/011_00000.parquet", "oid": "7f557356b7446c6209ed4e51eb97d65324d75b77", "size": 2152323681},
    {"path": "sample/10BT/012_00000.parquet", "oid": "08bc10f931c49d4f5e2e832af85c88d19f9a30e9", "size": 2152069689},
    {"path": "sample/10BT/013_00000.parquet", "oid": "3f659cc279c926e8baeda616b688b4e704b7fa09", "size": 540632672},
]

DATASETS = {
    "wikitext2": WIKITEXT2,
    "tiny_shakespeare": {"source": TINY_SHAKESPEARE_URL},
    "fixture": FIXTURE_SPLIT_TEXT,
    "fineweb_edu": {"source": FINEWEB_EDU_VARIANT},
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
        if self.dataset_name == "fineweb_edu":
            self._prepare_fineweb_edu()
            return
        self._prepare_wikitext2()

    def _prepare_fineweb_edu(self) -> None:
        """Materialize a deterministic token window from pinned FineWeb-Edu.

        This is deliberately separate from the old small-corpus paths.  The
        Kaggle scale runner installs ``datasets`` and streams the immutable
        Hub revision once per job, then all conditions sample the same local
        token arrays.  Keeping the compact arrays local makes the paired
        token stream auditable without committing a large corpus.
        """
        expected = {
            "variant": FINEWEB_EDU_VARIANT,
            "dataset": FINEWEB_EDU_DATASET,
            "config": FINEWEB_EDU_CONFIG,
            "revision": FINEWEB_EDU_REVISION,
            "train_tokens": FINEWEB_EDU_TRAIN_TOKENS,
            "val_tokens": FINEWEB_EDU_VAL_TOKENS,
            "test_tokens": FINEWEB_EDU_TEST_TOKENS,
        }
        manifest_path = self.root / "source_manifest.json"
        if all((self.root / f"{split}.pt").exists() for split in ("train", "val", "test")) and manifest_path.exists():
            try:
                if json.loads(manifest_path.read_text()) == expected:
                    return
            except (OSError, ValueError):
                pass

        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "fineweb_edu requires the scale environment; install requirements-scale-lock.txt"
            ) from exc

        print(
            f"Streaming {FINEWEB_EDU_DATASET}/{FINEWEB_EDU_CONFIG} at revision "
            f"{FINEWEB_EDU_REVISION}...",
            flush=True,
        )
        stream = load_dataset(
            FINEWEB_EDU_DATASET,
            name=FINEWEB_EDU_CONFIG,
            split="train",
            streaming=True,
            revision=FINEWEB_EDU_REVISION,
        )
        targets = {
            "train": FINEWEB_EDU_TRAIN_TOKENS,
            "val": FINEWEB_EDU_VAL_TOKENS,
            "test": FINEWEB_EDU_TEST_TOKENS,
        }
        buffers = {split: array("I") for split in targets}
        split = "train"
        documents = 0
        for example in stream:
            text = example.get("text")
            if not isinstance(text, str) or not text:
                continue
            buffers[split].extend(self.encoder.encode(text, allowed_special=set()))
            documents += 1
            while len(buffers[split]) >= targets[split]:
                if split == "train":
                    split = "val"
                elif split == "val":
                    split = "test"
                else:
                    break
            if split == "test" and len(buffers["test"]) >= targets["test"]:
                break
        if any(len(buffers[name]) < count for name, count in targets.items()):
            raise RuntimeError(f"FineWeb-Edu stream ended before requested token budget: { {k: len(v) for k, v in buffers.items()} }")

        for name, count in targets.items():
            tensor = torch.tensor(buffers[name][:count], dtype=torch.long)
            torch.save(tensor, self.root / f"{name}.pt")
        expected = {
            **expected,
            "documents_consumed": documents,
            "source_files": FINEWEB_EDU_SOURCE_FILES,
            "source_files_sha256": hashlib.sha256(
                json.dumps(FINEWEB_EDU_SOURCE_FILES, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        manifest_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")

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
        if self.dataset_name == "fineweb_edu":
            path = self.root / f"{split}.pt"
            tokens = torch.load(path, map_location="cpu")
            if not isinstance(tokens, torch.Tensor):
                raise TypeError(f"expected tensor in {path}")
            tokens = tokens.to(dtype=torch.long)
            if tokens.numel() < 2:
                raise ValueError(f"Dataset split {path} contains fewer than two tokens")
            return tokens
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
        elif self.dataset_name == "fineweb_edu":
            source = json.loads((self.root / "source_manifest.json").read_text())
        else:
            source = {"variant": FIXTURE_VARIANT}
        return {
            "dataset": self.dataset_name,
            "tokenizer": "gpt2",
            "vocab_size": self.vocab_size,
            "source": source,
            "token_counts": {split: int(tokens.numel()) for split, tokens in self.tokens.items()},
            "sha256": {
                split: hashlib.sha256(
                    (self.root / (f"{split}.pt" if self.dataset_name == "fineweb_edu" else f"{split}.txt")).read_bytes()
                ).hexdigest()
                for split in ("train", "val", "test")
            },
        }

    def save_metadata(self) -> None:
        (self.root / "metadata.json").write_text(json.dumps(self.metadata(), indent=2) + "\n")

    def get_batch(
        self,
        split: str,
        batch_size: int,
        block_size: int,
        device: torch.device,
        return_starts: bool = False,
    ):
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
        result = (x.to(device), y.to(device))
        if return_starts:
            return (*result, starts)
        return result

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
