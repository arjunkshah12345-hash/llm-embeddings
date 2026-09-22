import hashlib
import json
from pathlib import Path

from data import TokenDataset, TINY_SHAKESPEARE_SPLIT_VERSION


def test_tiny_shakespeare_splits_are_disjoint(tmp_path):
    source = ("ABCDEFGHIJ" * 200) + "\n"
    root = tmp_path / "tiny_shakespeare"
    root.mkdir()
    (root / "input.txt").write_text(source, encoding="utf-8")
    # Contaminated layout from the old identical-file downloads.
    for split in ("train", "val", "test"):
        (root / f"{split}.txt").write_text(source, encoding="utf-8")

    dataset = TokenDataset(str(tmp_path), "tiny_shakespeare", seed=1)
    texts = {split: (root / f"{split}.txt").read_text(encoding="utf-8") for split in ("train", "val", "test")}
    assert texts["train"] + texts["val"] + texts["test"] == source
    assert texts["train"] != texts["val"]
    assert texts["val"] != texts["test"]
    assert texts["train"] != texts["test"]
    assert len(texts["train"]) > len(texts["val"])
    assert len(texts["train"]) > len(texts["test"])

    stamp = json.loads((root / "split_manifest.json").read_text())
    assert stamp["version"] == TINY_SHAKESPEARE_SPLIT_VERSION
    assert stamp["source_sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert dataset.metadata()["sha256"]["train"] != dataset.metadata()["sha256"]["val"]


def test_tiny_shakespeare_split_is_stable_across_reloads(tmp_path):
    source = ("Once more unto the breach, dear friends, once more!\n" * 50)
    root = tmp_path / "tiny_shakespeare"
    root.mkdir()
    (root / "input.txt").write_text(source, encoding="utf-8")
    first = TokenDataset(str(tmp_path), "tiny_shakespeare", seed=3)
    second = TokenDataset(str(tmp_path), "tiny_shakespeare", seed=9)
    assert first.metadata()["sha256"] == second.metadata()["sha256"]
