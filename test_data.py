import hashlib
import json
from pathlib import Path

import pytest

from data import (
    TINY_SHAKESPEARE_COMMIT,
    TINY_SHAKESPEARE_SHA256,
    TINY_SHAKESPEARE_SPLIT_VERSION,
    TINY_SHAKESPEARE_VARIANT,
    WIKITEXT2,
    WIKITEXT2_SOURCE_MANIFEST_VERSION,
    WIKITEXT2_VARIANT,
    TokenDataset,
)


def test_sources_are_pinned_to_expected_variants_and_hashes():
    assert WIKITEXT2_VARIANT == "wikitext-2-raw-v1"
    assert WIKITEXT2_SOURCE_MANIFEST_VERSION == 1
    for specification in WIKITEXT2.values():
        assert "main" not in specification["url"]
        assert len(specification["sha256"]) == 64
    assert len(TINY_SHAKESPEARE_COMMIT) == 40
    assert TINY_SHAKESPEARE_VARIANT == "tiny-shakespeare-char-rnn-v1"
    assert len(TINY_SHAKESPEARE_SHA256) == 64


def test_cached_wikitext_requires_the_pinned_source_manifest(tmp_path, monkeypatch):
    root = tmp_path / "wikitext2"
    root.mkdir()
    for split, specification in WIKITEXT2.items():
        path = root / f"{split}.txt"
        path.write_text("cached text\n", encoding="utf-8")
    (root / "source_manifest.json").write_text(
        json.dumps(
            {
                "version": WIKITEXT2_SOURCE_MANIFEST_VERSION,
                "variant": WIKITEXT2_VARIANT,
                "files": WIKITEXT2,
            }
        )
    )

    def fail_if_downloaded(*args, **kwargs):
        raise AssertionError("pinned cached files should not be downloaded")

    monkeypatch.setattr("data.urllib.request.urlretrieve", fail_if_downloaded)
    with pytest.raises(AssertionError, match="should not be downloaded"):
        TokenDataset(str(tmp_path), "wikitext2", seed=1)


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
