import json

from sweep import run_is_complete, upsert_run


def test_run_is_complete_uses_final_validation_step(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps({"split": "val", "step": 4}) + "\n")
    assert run_is_complete(tmp_path, steps=5)
    assert not run_is_complete(tmp_path, steps=6)


def test_upsert_run_replaces_same_seed_and_embedding_type():
    manifest = {"runs": [{"seed": 7, "embedding_type": "tied", "command": ["old"]}]}
    upsert_run(manifest, {"seed": 7, "embedding_type": "tied", "command": ["new"]})
    assert manifest["runs"] == [{"seed": 7, "embedding_type": "tied", "command": ["new"]}]
