from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from config import ModelConfig, TrainConfig
from data import TokenDataset
from model import GPTModel
from train import choose_device, evaluate, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved embedding experiment checkpoint")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batches", type=int, default=100)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    saved = json.loads((run_dir / "config.json").read_text())
    model_config = ModelConfig(**saved["model"])
    train_config = TrainConfig(**saved["train"])
    train_config.eval_batches = args.batches
    set_seed(train_config.seed)
    device = choose_device(args.device)
    dataset = TokenDataset(train_config.data_dir, train_config.dataset, train_config.seed)
    model = GPTModel(model_config, train_config.embedding_type, train_config.seed).to(device)
    checkpoint = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    result = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "step": checkpoint.get("step"),
        "parameter_counts": model.parameter_counts(),
        "val_loss": evaluate(model, dataset, "val", train_config, device),
        "test_loss": evaluate(model, dataset, "test", train_config, device),
    }
    result["val_perplexity"] = math.exp(min(result["val_loss"], 20.0))
    result["test_perplexity"] = math.exp(min(result["test_loss"], 20.0))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
