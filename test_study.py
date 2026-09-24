import json

from validate_study import training_batch_stream_digest, validate_study


MODEL = {
    "vocab_size": 97,
    "block_size": 16,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "dropout": 0.0,
    "adapter_rank": 4,
    "adapter_alpha": 4.0,
}

COMMON = {
    "dataset": "fixture",
    "data_dir": "data",
    "device": "cpu",
    "steps": 5,
    "batch_size": 2,
    "grad_accum_steps": 1,
    "block_size": 16,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 32,
    "dropout": 0.0,
    "adapter_rank": 4,
    "adapter_alpha": 4.0,
    "learning_rate": 0.001,
    "min_learning_rate": 0.0001,
    "warmup_steps": 1,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_interval": 1,
    "eval_batches": 1,
    "log_interval": 1,
    "save_interval": 1,
    "path_ablation": "none",
    "ablation_start": 0,
    "ablation_end": 0,
}


def make_study(tmp_path, token_counts=(160, 160, 160)):
    dataset = {
        "dataset": "fixture",
        "tokenizer": "fixture",
        "vocab_size": 97,
        "source": {"variant": "fixture-v1", "sha256": "fixture"},
        "token_counts": {"train": 160, "val": 160, "test": 160},
        "sha256": {"train": "a", "val": "b", "test": "c"},
    }
    embedding_types = ["tied", "partial", "untied"]
    study = {
        "common": COMMON,
        "seeds": [7],
        "embedding_types": embedding_types,
        "runs": [],
    }
    for embedding_type, tokens in zip(embedding_types, token_counts):
        run_name = f"seed7_{embedding_type}"
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        config = {
            "model": MODEL,
            "train": {
                **COMMON,
                "output_dir": str(tmp_path),
                "run_name": run_name,
                "embedding_type": embedding_type,
                "seed": 7,
            },
            "device": "cpu",
            "dataset": dataset,
        }
        (run_dir / "config.json").write_text(json.dumps(config))
        expected_stream = training_batch_stream_digest(7, config, 160)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "batch_stream": {
                        "algorithm": "sha256-jsonl-start-offsets-v1",
                        "digest": expected_stream,
                        "record_count": 5,
                        "token_count": tokens,
                        "offset_log": "batch_offsets.jsonl",
                    },
                }
            )
        )
        (run_dir / "parameter_counts.json").write_text(
            json.dumps(
                {
                    "embedding_type": embedding_type,
                    "total_parameters": 100,
                    "trainable_parameters": 100,
                    "transformer_parameters": 50,
                    "embedding_parameters": 50,
                }
            )
        )
        (run_dir / "metrics.jsonl").write_text(
            json.dumps({"split": "val", "step": 4, "tokens_seen": tokens, "loss": 4.0, "perplexity": 54.6}) + "\n"
        )
        study["runs"].append({"run_name": run_name, "seed": 7, "embedding_type": embedding_type})
    (tmp_path / "study_manifest.json").write_text(json.dumps(study))


def test_study_validation_accepts_complete_fair_matrix(tmp_path):
    make_study(tmp_path)

    result = validate_study(tmp_path)

    assert result["passed"]
    assert result["expected_run_count"] == 3
    assert result["token_tolerance_by_seed"]["7"]["delta"] == 0


def test_study_validation_rejects_unequal_token_exposure(tmp_path):
    make_study(tmp_path, token_counts=(160, 160, 200))

    result = validate_study(tmp_path)

    assert not result["passed"]
    assert any("token exposure" in error for error in result["errors"])


def test_study_validation_requires_a_source_variant(tmp_path):
    make_study(tmp_path)
    for run_name in ("seed7_tied", "seed7_partial", "seed7_untied"):
        config_path = tmp_path / run_name / "config.json"
        config = json.loads(config_path.read_text())
        del config["dataset"]["source"]["variant"]
        config_path.write_text(json.dumps(config))

    result = validate_study(tmp_path)

    assert not result["passed"]
    assert any("pinned source variant" in error for error in result["errors"])


def test_study_validation_requires_exact_token_exposure(tmp_path):
    make_study(tmp_path, token_counts=(160, 160, 161))

    result = validate_study(tmp_path)

    assert not result["passed"]
    assert result["token_tolerance_by_seed"]["7"]["allowed_delta"] == 0
    assert any("must match exactly" in error for error in result["errors"])


def test_study_validation_rejects_premature_final_validation(tmp_path):
    make_study(tmp_path)
    manifest = json.loads((tmp_path / "study_manifest.json").read_text())
    manifest["common"]["steps"] = 6
    (tmp_path / "study_manifest.json").write_text(json.dumps(manifest))
    for run_name in ("seed7_tied", "seed7_partial", "seed7_untied"):
        config_path = tmp_path / run_name / "config.json"
        config = json.loads(config_path.read_text())
        config["train"]["steps"] = 6
        config_path.write_text(json.dumps(config))

    result = validate_study(tmp_path)

    assert not result["passed"]
    assert any("declared final step 5" in error for error in result["errors"])


def test_study_validation_rejects_non_finite_metrics(tmp_path):
    make_study(tmp_path)
    metrics_path = tmp_path / "seed7_partial" / "metrics.jsonl"
    metrics_path.write_text('{"split":"val","step":4,"tokens_seen":160,"loss":NaN,"perplexity":NaN}\n')

    result = validate_study(tmp_path)

    assert not result["passed"]
    assert any("non-finite" in error for error in result["errors"])


def test_study_validation_rejects_path_ablation_mismatch(tmp_path):
    make_study(tmp_path)
    config_path = tmp_path / "seed7_partial" / "config.json"
    config = json.loads(config_path.read_text())
    config["train"]["path_ablation"] = "stop_input"
    config_path.write_text(json.dumps(config))

    result = validate_study(tmp_path)

    assert not result["passed"]
    assert any("shared model/training configuration differs" in error for error in result["errors"])


def test_study_validation_rejects_run_metadata_mismatch(tmp_path):
    make_study(tmp_path)
    config_path = tmp_path / "seed7_partial" / "config.json"
    config = json.loads(config_path.read_text())
    config["train"]["seed"] = 8
    config_path.write_text(json.dumps(config))

    result = validate_study(tmp_path)

    assert not result["passed"]
    assert any("config seed does not match study entry" in error for error in result["errors"])


def test_training_batch_stream_digest_is_seed_paired_and_sensitive_to_budget():
    config = {"steps": 5, "batch_size": 2, "grad_accum_steps": 1, "block_size": 16}
    first = training_batch_stream_digest(7, config, 160)
    second = training_batch_stream_digest(7, config, 160)
    changed = training_batch_stream_digest(8, config, 160)
    longer = training_batch_stream_digest(7, {**config, "steps": 6}, 160)
    assert first == second
    assert first != changed
    assert first != longer
