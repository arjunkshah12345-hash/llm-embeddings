import json

import pytest
import torch

from sweep import run_is_complete, upsert_run, validate_resume_common
from train import batch_stream_record_bytes, load_batch_stream_digest, truncate_batch_stream_log


def test_run_is_complete_uses_final_validation_step(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"split": "val", "step": 4}) + "\n")
    assert run_is_complete(tmp_path, steps=5)
    assert not run_is_complete(tmp_path, steps=6)


def test_upsert_run_replaces_same_seed_and_embedding_type():
    manifest = {"runs": [{"seed": 7, "embedding_type": "tied", "command": ["old"]}]}
    upsert_run(manifest, {"seed": 7, "embedding_type": "tied", "command": ["new"]})
    assert manifest["runs"] == [{"seed": 7, "embedding_type": "tied", "command": ["new"]}]


def test_resume_rejects_a_changed_step_horizon():
    with pytest.raises(SystemExit, match="different step horizon"):
        validate_resume_common({"steps": 10_000, "learning_rate": 3e-4}, {"steps": 50_000, "learning_rate": 3e-4})


def test_resume_accepts_the_same_step_horizon():
    validate_resume_common({"steps": 10_000, "learning_rate": 3e-4}, {"steps": 10_000, "learning_rate": 3e-4})


def test_actual_batch_stream_digest_round_trip_and_truncation(tmp_path):
    path = tmp_path / "batch_offsets.jsonl"
    first = batch_stream_record_bytes(0, 0, torch.tensor([3, 9]))
    second = batch_stream_record_bytes(1, 0, torch.tensor([4, 8]))
    path.write_bytes(first + second)

    digest, count = load_batch_stream_digest(path)

    expected = __import__("hashlib").sha256(first + second).hexdigest()
    assert digest.hexdigest() == expected
    assert count == 2
    assert truncate_batch_stream_log(path, 1) == 1
    assert path.read_bytes() == first
