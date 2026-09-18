from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch

from config import ModelConfig, TrainConfig, as_dict
from data import TokenDataset
from model import GPTModel


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0.0


def learning_rate(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


@torch.no_grad()
def evaluate(model: GPTModel, dataset: TokenDataset, split: str, config: TrainConfig, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for batch_index in range(config.eval_batches):
        x, y = dataset.get_fixed_batch(split, batch_index, config.batch_size, model.config.block_size, device)
        losses.append(model.loss(x, y).item())
    if was_training:
        model.train()
    return float(np.mean(losses))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def checkpoint_payload(model: GPTModel, model_config: ModelConfig, train_config: TrainConfig, device: torch.device, step: int, best_val_loss: float) -> dict:
    # Keep checkpoints small enough for local research runs. Optimizer state is
    # intentionally omitted; this first version supports evaluation/reproduction
    # rather than exact mid-run resume.
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return {
        "model": model_state,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": {**as_dict(model_config, train_config), "device": str(device)},
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def parse_args() -> tuple[ModelConfig, TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding_type", choices=["tied", "untied", "partial"], required=True)
    parser.add_argument("--dataset", choices=["wikitext2", "tiny_shakespeare"], default="wikitext2")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="runs")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter_rank", type=int, default=8)
    parser.add_argument("--adapter_alpha", type=float, default=8.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    args = parser.parse_args()
    model_config = ModelConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
    )
    train_config = TrainConfig(
        dataset=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name or args.embedding_type,
        embedding_type=args.embedding_type,
        seed=args.seed,
        device=args.device,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )
    return model_config, train_config, args


def main() -> None:
    model_config, train_config, _ = parse_args()
    set_seed(train_config.seed)
    device = choose_device(train_config.device)
    dataset = TokenDataset(train_config.data_dir, train_config.dataset, train_config.seed)
    dataset.save_metadata()
    model_config.vocab_size = dataset.vocab_size
    model = GPTModel(model_config, train_config.embedding_type, train_config.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    run_dir = Path(train_config.output_dir) / train_config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", {**as_dict(model_config, train_config), "device": str(device), "dataset": dataset.metadata()})
    write_json(run_dir / "parameter_counts.json", model.parameter_counts())
    write_json(
        run_dir / "manifest.json",
        {
            "git_commit": git_commit(),
            "command": [sys.executable, *sys.argv],
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "tiktoken": getattr(tiktoken, "__version__", "unknown"),
            "tokenizer": {"name": "gpt2", "vocab_size": dataset.vocab_size},
            "dataset": dataset.metadata(),
            "model": model.config_dict(),
            "parameter_counts": model.parameter_counts(),
        },
    )
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    best_val = float("inf")
    tokens_seen = 0
    peak_memory = 0.0
    started = time.perf_counter()
    model.train()

    print(f"run={train_config.run_name} embedding={train_config.embedding_type} device={device}")
    print(json.dumps(model.parameter_counts(), sort_keys=True))
    for step in range(train_config.steps):
        lr = learning_rate(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        gradient_metrics = None
        for micro_step in range(train_config.grad_accum_steps):
            x, y = dataset.get_batch("train", train_config.batch_size, model_config.block_size, device)
            logits, _ = model(x, return_hidden=True)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            step_loss += loss.detach().item()
            should_measure = step % train_config.log_interval == 0 and micro_step == train_config.grad_accum_steps - 1
            if should_measure:
                gradient_metrics = model.embedding_gradient_metrics(x, y)
            (loss / train_config.grad_accum_steps).backward()
            tokens_seen += x.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        elapsed = time.perf_counter() - started
        memory = memory_mb(device)
        peak_memory = max(peak_memory, memory)
        if step % train_config.log_interval == 0:
            record = {
                "step": step,
                "split": "train",
                "loss": step_loss / train_config.grad_accum_steps,
                "perplexity": math.exp(min(step_loss / train_config.grad_accum_steps, 20.0)),
                "learning_rate": lr,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
                "grad_norm": float(grad_norm),
                "combined_embedding_grad_norm": model.combined_embedding_grad_norm(),
                "peak_gpu_memory_mb": peak_memory,
                **(gradient_metrics or {}),
                **model.embeddings.adapter_metrics(),
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"step={step:5d} train_loss={record['loss']:.4f} "
                f"val_pending lr={lr:.2e} tok/s={record['tokens_per_second']:.1f} "
                f"out/in={record.get('output_to_input_grad_ratio', 0.0):.3f}"
            )

        if step % train_config.eval_interval == 0 or step == train_config.steps - 1:
            val_loss = evaluate(model, dataset, "val", train_config, device)
            val_record = {
                "step": step,
                "split": "val",
                "loss": val_loss,
                "perplexity": math.exp(min(val_loss, 20.0)),
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
                "peak_gpu_memory_mb": peak_memory,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(val_record) + "\n")
            print(f"step={step:5d} val_loss={val_loss:.4f} val_ppl={val_record['perplexity']:.2f}")
            checkpoint = checkpoint_payload(model, model_config, train_config, device, step, min(best_val, val_loss))
            save_checkpoint(run_dir / "last.pt", checkpoint)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(run_dir / "best.pt", checkpoint)
        elif step % train_config.save_interval == 0:
            save_checkpoint(
                run_dir / "last.pt",
                checkpoint_payload(model, model_config, train_config, device, step, best_val),
            )

    print(f"finished run={train_config.run_name} best_val_loss={best_val:.4f} output={run_dir}")


if __name__ == "__main__":
    main()
