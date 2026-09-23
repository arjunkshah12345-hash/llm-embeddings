"""Evaluate embedding-role mechanics from validated study checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from data import TokenDataset
from model import GPTModel
from token_classes import dataset_class_ids, dataset_frequency_bucket_ids
from train import choose_device, set_seed


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def final_train_metrics(run_dir: Path) -> dict:
    rows = [row for row in read_jsonl(run_dir / "metrics.jsonl") if row.get("split") == "train"]
    return rows[-1] if rows else {}


def study_run_dirs(runs_dir: Path) -> list[Path]:
    return sorted(
        path for path in runs_dir.iterdir()
        if path.is_dir() and (path / "config.json").exists() and (path / "parameter_counts.json").exists()
    )


def validate_parent_study(runs_dir: Path) -> dict:
    validation_path = runs_dir / "study_validation.json"
    if not validation_path.exists():
        raise SystemExit(f"missing fairness validation output: {validation_path}")
    validation = json.loads(validation_path.read_text())
    if not validation.get("passed"):
        raise SystemExit(f"fairness validation failed: {validation_path}")
    return validation


def evaluate_run(run_dir: Path, checkpoint_name: str, device_name: str) -> dict:
    saved = json.loads((run_dir / "config.json").read_text())
    model_config = ModelConfig(**saved["model"])
    train_config = TrainConfig(**saved["train"])
    set_seed(train_config.seed)
    device = choose_device(device_name)
    dataset = TokenDataset(train_config.data_dir, train_config.dataset, train_config.seed)
    model = GPTModel(model_config, train_config.embedding_type, train_config.seed).to(device)
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise SystemExit(f"missing checkpoint for {run_dir}: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    x, y = dataset.get_fixed_batch(
        "train", batch_index=0, batch_size=train_config.batch_size,
        block_size=model_config.block_size, device=device,
    )
    class_ids = dataset_class_ids(dataset)
    frequency_ids = dataset_frequency_bucket_ids(dataset)
    with torch.enable_grad():
        gradient_metrics = model.embedding_gradient_metrics(
            x, y, token_class_ids=class_ids, token_frequency_ids=frequency_ids
        )
    adapter_metrics = model.embeddings.adapter_metrics()
    train_metrics = final_train_metrics(run_dir)
    result = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint_name,
        "checkpoint_step": checkpoint.get("step"),
        "seed": train_config.seed,
        "embedding_type": train_config.embedding_type,
        "device": str(device),
        "parameter_counts": model.parameter_counts(),
        "dataset": dataset.metadata(),
        "fixed_batch": {
            "split": "train",
            "batch_index": 0,
            "batch_size": train_config.batch_size,
            "block_size": model_config.block_size,
        },
        "gradient_metrics": gradient_metrics,
        "adapter_metrics": adapter_metrics,
        "checkpoint_validation": {
            "best_val_loss": checkpoint.get("best_val_loss"),
            "final_train_step": train_metrics.get("step"),
            "embedding_cumulative_update_norm": train_metrics.get("embedding_cumulative_update_norm"),
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    validation = validate_parent_study(runs_dir)
    results = [evaluate_run(run_dir, args.checkpoint, args.device) for run_dir in study_run_dirs(runs_dir)]
    if not results:
        raise SystemExit(f"no study runs found under {runs_dir}")
    output = Path(args.output) if args.output else runs_dir / "mechanism_metrics.json"
    payload = {
        "runs_dir": str(runs_dir),
        "checkpoint": args.checkpoint,
        "study_validation": validation,
        "runs": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
