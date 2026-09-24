"""Create a deterministic, non-training study fixture for CI integration checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_study import training_batch_stream_digest


CONDITIONS = ("tied", "untied", "partial", "capacity_control", "partial_input", "partial_output")
SEED = 1337
STEPS = 3
BLOCK_SIZE = 8
BATCH_SIZE = 1
TRAIN_TOKEN_COUNT = 100


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_fixture(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = {
        "dataset": "fixture",
        "tokenizer": "fixture",
        "vocab_size": 97,
        "source": {"variant": "fixture-v1", "sha256": "ci-fixture-source"},
        "token_counts": {"train": TRAIN_TOKEN_COUNT, "val": 100, "test": 100},
        "sha256": {"train": "ci-train", "val": "ci-val", "test": "ci-test"},
    }
    model = {
        "vocab_size": 97,
        "block_size": BLOCK_SIZE,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "dropout": 0.0,
        "adapter_rank": 2,
        "adapter_alpha": 2.0,
        "capacity_control_width": 0,
    }
    common = {
        "dataset": "fixture",
        "data_dir": "ci-fixture-data",
        "device": "synthetic",
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": 1,
        "block_size": BLOCK_SIZE,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 16,
        "dropout": 0.0,
        "adapter_rank": 2,
        "adapter_alpha": 2.0,
        "capacity_control_width": 0,
        "learning_rate": 0.001,
        "min_learning_rate": 0.0001,
        "warmup_steps": 1,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "eval_interval": 1,
        "eval_batches": 1,
        "log_interval": 1,
        "save_interval": 3,
        "path_ablation": "none",
        "ablation_start": 0,
        "ablation_end": 0,
    }
    study = {
        "common": common,
        "seeds": [SEED],
        "embedding_types": list(CONDITIONS),
        "runs": [],
    }
    stream_digest = training_batch_stream_digest(SEED, common, TRAIN_TOKEN_COUNT)
    for index, condition in enumerate(CONDITIONS):
        run_name = f"seed{SEED}_{condition}"
        run_dir = output_dir / run_name
        run_dir.mkdir(exist_ok=True)
        train = {**common, "output_dir": str(output_dir), "run_name": run_name, "embedding_type": condition, "seed": SEED}
        config = {"model": model, "train": train, "device": "synthetic", "dataset": dataset}
        write_json(run_dir / "config.json", config)
        tied_embedding = 97 * 16
        extra = {"tied": 0, "untied": tied_embedding, "partial": 2 * 2 * (97 + 16), "capacity_control": 2 * 2 * (97 + 16), "partial_input": 2 * (97 + 16), "partial_output": 2 * (97 + 16)}[condition]
        counts = {
            "embedding_type": condition,
            "total_parameters": 2000 + tied_embedding + extra,
            "trainable_parameters": 2000 + tied_embedding + extra,
            "transformer_parameters": 2000,
            "embedding_parameters": tied_embedding + extra,
            "embedding_compute_parameters": tied_embedding + (2 * 16 if condition in {"partial", "partial_input", "partial_output"} else 0),
            "output_projection_parameters": tied_embedding,
        }
        write_json(run_dir / "parameter_counts.json", counts)
        write_json(
            run_dir / "manifest.json",
            {
                "dataset": dataset,
                "batch_stream": {
                    "algorithm": "sha256-jsonl-start-offsets-v1",
                    "digest": stream_digest,
                    "record_count": STEPS,
                    "token_count": STEPS * BATCH_SIZE * BLOCK_SIZE,
                    "offset_log": "batch_offsets.jsonl",
                },
            },
        )
        rows = []
        for step in range(STEPS):
            loss = 4.0 + index * 0.01 - step * 0.02
            rows.append(
                {
                    "step": step,
                    "split": "train",
                    "loss": loss,
                    "perplexity": 54.6,
                    "tokens_seen": (step + 1) * BATCH_SIZE * BLOCK_SIZE,
                    "tokens_per_second": 1000.0,
                    "training_wall_time_seconds": float(step + 1),
                    "estimated_flops_total": float((step + 1) * 1000),
                    "estimated_flops_non_embedding": float((step + 1) * 900),
                    "peak_gpu_memory_mb": 0.0,
                    "input_grad_norm": 1.0,
                    "output_grad_norm": 1.2,
                    "output_to_input_grad_ratio": 1.2,
                    "input_output_grad_cosine": 0.0,
                    "input_output_correction_cosine": 0.0,
                    "input_output_left_subspace_overlap": 0.0,
                    "input_output_right_subspace_overlap": 0.0,
                    "input_correction_relative_norm": 0.0,
                    "output_correction_relative_norm": 0.0,
                    "input_correction_effective_rank": 0.0,
                    "output_correction_effective_rank": 0.0,
                }
            )
            rows.append(
                {
                    "step": step,
                    "split": "val",
                    "loss": loss + 0.1,
                    "perplexity": 60.0,
                    "tokens_seen": (step + 1) * BATCH_SIZE * BLOCK_SIZE,
                    "tokens_per_second": 1000.0,
                    "training_wall_time_seconds": float(step + 1),
                    "estimated_flops_total": float((step + 1) * 1000),
                    "estimated_flops_non_embedding": float((step + 1) * 900),
                    "peak_gpu_memory_mb": 0.0,
                }
            )
        (run_dir / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        study["runs"].append({"run_name": run_name, "seed": SEED, "embedding_type": condition, "command": ["synthetic-fixture"]})
    write_json(output_dir / "study_manifest.json", study)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/ci-e2e")
    args = parser.parse_args()
    build_fixture(Path(args.output_dir))


if __name__ == "__main__":
    main()
