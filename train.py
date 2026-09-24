from __future__ import annotations

import argparse
import hashlib
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
from token_classes import dataset_class_ids, dataset_frequency_bucket_ids


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


def active_path_ablation(step: int, config: TrainConfig) -> str:
    """Return the path ablation in force at this step, or none outside the interval."""
    mode = config.path_ablation
    if mode in ("", "none"):
        return "none"
    if mode not in {"stop_input", "stop_output"}:
        raise ValueError("path_ablation must be none, stop_input, or stop_output")
    end = config.ablation_end if config.ablation_end > 0 else config.steps
    if config.ablation_start <= step < end:
        return mode
    return "none"


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


BATCH_STREAM_ALGORITHM = "sha256-jsonl-start-offsets-v1"


def batch_stream_record_bytes(step: int, micro_step: int, starts: torch.Tensor) -> bytes:
    record = {
        "step": int(step),
        "micro_step": int(micro_step),
        "starts": [int(value) for value in starts.detach().cpu().tolist()],
    }
    return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def load_batch_stream_digest(path: Path, start_step: int | None = None) -> tuple[hashlib._Hash, int]:
    """Rebuild the actual sampled-offset digest from the append-only audit log."""
    digest = hashlib.sha256()
    record_count = 0
    if not path.exists():
        return digest, record_count
    for raw_line in path.read_bytes().splitlines(keepends=True):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if start_step is not None and int(record["step"]) >= start_step:
            continue
        digest.update(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
        record_count += 1
    return digest, record_count


def truncate_batch_stream_log(path: Path, start_step: int) -> int:
    """Drop sampled-offset records at or after a checkpoint resume step."""
    if not path.exists():
        return 0
    kept: list[bytes] = []
    removed = 0
    for raw_line in path.read_bytes().splitlines(keepends=True):
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if int(record["step"]) >= start_step:
            removed += 1
        else:
            kept.append(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
    path.write_bytes(b"".join(kept))
    return removed


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


def estimate_flops(parameter_counts: dict[str, int], tokens: int) -> dict[str, float]:
    """Approximate training FLOPs with the common 6ND dense-matmul rule.

    Input embedding lookups are treated as negligible. Dense output projection
    and factorized adapter matmuls are represented by
    ``embedding_compute_parameters``; this keeps untied input storage from being
    counted as dense projection work. Transformer and embedding FLOPs are
    reported separately so tied/untied/partial comparisons stay fair.
    """
    transformer = float(parameter_counts["transformer_parameters"])
    embedding = float(parameter_counts.get("embedding_compute_parameters", parameter_counts["embedding_parameters"]))
    non_embedding_flops = 6.0 * transformer * tokens
    embedding_flops = 6.0 * embedding * tokens
    return {
        "estimated_flops_non_embedding": non_embedding_flops,
        "estimated_flops_embedding": embedding_flops,
        "estimated_flops_total": non_embedding_flops + embedding_flops,
        "flops_per_token_non_embedding": 6.0 * transformer,
        "flops_per_token_embedding": 6.0 * embedding,
        "flops_per_token_total": 6.0 * (transformer + embedding),
    }


def compact_checkpoint_payload(
    model: GPTModel,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    step: int,
    best_val_loss: float,
) -> dict:
    """Evaluation checkpoint without optimizer state (small, comparable across runs)."""
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return {
        "kind": "compact",
        "model": model_state,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": {**as_dict(model_config, train_config), "device": str(device)},
    }


def optimizer_checkpoint_payload(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    step: int,
    best_val_loss: float,
    tokens_seen: int,
    training_elapsed: float,
    embedding_cumulative_update_norm: float = 0.0,
    dataset_generators: dict[str, torch.Generator] | None = None,
) -> dict:
    """Full resume checkpoint with AdamW state and RNG."""
    payload = compact_checkpoint_payload(model, model_config, train_config, device, step, best_val_loss)
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if dataset_generators is not None:
        rng["dataset_generators"] = {
            split: generator.get_state() for split, generator in dataset_generators.items()
        }
    payload.update(
        {
            "kind": "optimizer",
            "optimizer": optimizer.state_dict(),
            "tokens_seen": tokens_seen,
            "training_wall_time_seconds": training_elapsed,
            "embedding_cumulative_update_norm": embedding_cumulative_update_norm,
            "rng": rng,
        }
    )
    return payload


# Backwards-compatible alias used by tests and callers.
checkpoint_payload = compact_checkpoint_payload


def save_checkpoint(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def load_resume_checkpoint(
    path: Path,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"Resume checkpoint missing model weights: {path}")
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    rng = checkpoint.get("rng") or {}
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    return checkpoint


def restore_dataset_generators(
    dataset: TokenDataset,
    train_config: TrainConfig,
    checkpoint: dict,
    completed_steps: int,
    block_size: int,
) -> str:
    """Restore train sampling exactly, including compatibility with old checkpoints."""
    saved = (checkpoint.get("rng") or {}).get("dataset_generators")
    if saved:
        for split, state in saved.items():
            if split in dataset.generators:
                dataset.generators[split].set_state(state)
        return "checkpoint"

    # Older checkpoints did not save the dataset generators. Replaying the
    # deterministic start draws preserves the old run's stream when extending
    # one of those checkpoints.
    generator = dataset.generators["train"]
    upper = dataset.tokens["train"].numel() - block_size
    for _ in range(max(completed_steps, 0) * train_config.grad_accum_steps):
        torch.randint(0, upper, (train_config.batch_size,), generator=generator)
    return "replayed_legacy"


def truncate_metrics(path: Path, start_step: int) -> int:
    """Drop metric rows at or after start_step so resume does not duplicate history."""
    if not path.exists() or start_step <= 0:
        return 0
    rows = []
    removed = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("step", -1)) >= start_step:
            removed += 1
            continue
        rows.append(row)
    if removed:
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return removed


def parse_args() -> tuple[ModelConfig, TrainConfig, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding_type",
        choices=["tied", "untied", "partial", "partial_input", "partial_output", "capacity_control"],
        required=True,
    )
    parser.add_argument("--dataset", choices=["wikitext2", "tiny_shakespeare", "fixture"], default="wikitext2")
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
    parser.add_argument("--capacity_control_width", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--path_ablation", choices=["none", "stop_input", "stop_output"], default="none")
    parser.add_argument("--ablation_start", type=int, default=0, help="first step included in the path ablation")
    parser.add_argument("--ablation_end", type=int, default=0, help="first step excluded; 0 means through the last step")
    parser.add_argument(
        "--resume",
        default="",
        help="path to optimizer_last.pt (or any optimizer-kind checkpoint) to continue training",
    )
    parser.add_argument(
        "--save_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also write optimizer_last.pt for exact mid-run resume (default: true)",
    )
    args = parser.parse_args()
    if args.ablation_start < 0 or args.ablation_end < 0:
        parser.error("ablation_start and ablation_end must be non-negative")
    if args.ablation_end and args.ablation_end <= args.ablation_start:
        parser.error("ablation_end must be greater than ablation_start when non-zero")
    if args.ablation_start >= args.steps and args.path_ablation != "none":
        parser.error("ablation_start must be smaller than steps when path_ablation is enabled")
    model_config = ModelConfig(
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        adapter_rank=args.adapter_rank,
        adapter_alpha=args.adapter_alpha,
        capacity_control_width=args.capacity_control_width,
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
        path_ablation=args.path_ablation,
        ablation_start=args.ablation_start,
        ablation_end=args.ablation_end,
    )
    return model_config, train_config, args


def main() -> None:
    model_config, train_config, args = parse_args()
    set_seed(train_config.seed)
    device = choose_device(train_config.device)
    dataset = TokenDataset(train_config.data_dir, train_config.dataset, train_config.seed)
    dataset.save_metadata()
    model_config.vocab_size = dataset.vocab_size
    model = GPTModel(model_config, train_config.embedding_type, train_config.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    run_dir = Path(train_config.output_dir) / train_config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    parameter_counts = model.parameter_counts()
    write_json(run_dir / "config.json", {**as_dict(model_config, train_config), "device": str(device), "dataset": dataset.metadata()})
    write_json(run_dir / "parameter_counts.json", parameter_counts)
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
            "parameter_counts": parameter_counts,
            "flops_accounting": {
                "rule": "6ND dense-matmul approximation; embedding lookups treated as negligible",
                **estimate_flops(parameter_counts, tokens=1),
            },
            "resume_from": args.resume or None,
            "save_optimizer": bool(args.save_optimizer),
            "path_ablation": {
                "mode": train_config.path_ablation,
                "start": train_config.ablation_start,
                "end": train_config.ablation_end or train_config.steps,
            },
            "batch_stream": {
                "algorithm": BATCH_STREAM_ALGORITHM,
                "digest": None,
                "record_count": 0,
                "token_count": 0,
                "offset_log": "batch_offsets.jsonl",
            },
        },
    )
    metrics_path = run_dir / "metrics.jsonl"
    batch_stream_path = run_dir / "batch_offsets.jsonl"
    start_step = 0
    best_val = float("inf")
    tokens_seen = 0
    peak_memory = 0.0
    started = time.perf_counter()
    training_elapsed = 0.0
    embedding_cumulative_update_norm = 0.0
    if args.resume:
        resumed = load_resume_checkpoint(Path(args.resume), model, optimizer, device)
        start_step = int(resumed.get("step", -1)) + 1
        best_val = float(resumed.get("best_val_loss", best_val))
        tokens_seen = int(resumed.get("tokens_seen", 0))
        training_elapsed = float(resumed.get("training_wall_time_seconds", 0.0))
        embedding_cumulative_update_norm = float(resumed.get("embedding_cumulative_update_norm", 0.0))
        restore_mode = restore_dataset_generators(dataset, train_config, resumed, start_step, model_config.block_size)
        removed = truncate_metrics(metrics_path, start_step)
        if not batch_stream_path.exists():
            raise ValueError(f"Cannot resume without actual batch-offset log: {batch_stream_path}")
        truncate_batch_stream_log(batch_stream_path, start_step)
        print(
            f"resumed from {args.resume} at step={start_step} tokens_seen={tokens_seen}"
            f" dataset_generator={restore_mode}"
            + (f" truncated_metrics={removed}" if removed else "")
        )
    else:
        metrics_path.unlink(missing_ok=True)
        batch_stream_path.unlink(missing_ok=True)
    batch_stream_digest, batch_record_count = load_batch_stream_digest(batch_stream_path)
    batch_stream_handle = batch_stream_path.open("ab", buffering=1024 * 1024)
    model.train()
    token_class_ids = dataset_class_ids(dataset).to(device)
    token_frequency_ids = dataset_frequency_bucket_ids(dataset).to(device)

    print(f"run={train_config.run_name} embedding={train_config.embedding_type} device={device}")
    print(json.dumps(parameter_counts, sort_keys=True))
    for step in range(start_step, train_config.steps):
        step_started = time.perf_counter()
        lr = learning_rate(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        embedding_before_update = (
            [parameter.detach().clone() for parameter in model.embeddings.unique_parameters()]
            if step % train_config.log_interval == 0
            else None
        )
        step_loss = 0.0
        gradient_metrics = None
        path_ablation = active_path_ablation(step, train_config)
        for micro_step in range(train_config.grad_accum_steps):
            x, y, starts = dataset.get_batch(
                "train", train_config.batch_size, model_config.block_size, device, return_starts=True
            )
            batch_record = batch_stream_record_bytes(step, micro_step, starts)
            batch_stream_handle.write(batch_record)
            batch_stream_digest.update(batch_record)
            batch_record_count += 1
            logits, _ = model(x, return_hidden=True, path_ablation=path_ablation)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            step_loss += loss.detach().item()
            should_measure = step % train_config.log_interval == 0 and micro_step == train_config.grad_accum_steps - 1
            if should_measure:
                gradient_metrics = model.embedding_gradient_metrics(x, y, token_class_ids, token_frequency_ids)
            (loss / train_config.grad_accum_steps).backward()
            tokens_seen += x.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        update_metrics = {}
        if embedding_before_update is not None:
            with torch.no_grad():
                update_squared = sum(
                    (parameter.detach() - before).float().pow(2).sum()
                    for parameter, before in zip(model.embeddings.unique_parameters(), embedding_before_update)
                )
                embedding_update_norm = math.sqrt(update_squared.item())
                embedding_cumulative_update_norm += embedding_update_norm
                update_metrics = {
                    "embedding_update_norm": embedding_update_norm,
                    "embedding_cumulative_update_norm": embedding_cumulative_update_norm,
                    "embedding_update_relative_norm": embedding_update_norm / max(
                        sum(parameter.detach().float().pow(2).sum() for parameter in model.embeddings.unique_parameters()).sqrt().item(),
                        1e-12,
                    ),
                }
        training_elapsed += time.perf_counter() - step_started
        # Session wall clock resets on resume; training_wall_time_seconds stays cumulative.
        wall_elapsed = time.perf_counter() - started
        batch_stream_handle.flush()
        memory = memory_mb(device)
        peak_memory = max(peak_memory, memory)
        flops = estimate_flops(parameter_counts, tokens_seen)
        if step % train_config.log_interval == 0:
            record = {
                "step": step,
                "split": "train",
                "loss": step_loss / train_config.grad_accum_steps,
                "perplexity": math.exp(min(step_loss / train_config.grad_accum_steps, 20.0)),
                "learning_rate": lr,
                "tokens_seen": tokens_seen,
                "tokens_per_second": tokens_seen / max(training_elapsed, 1e-9),
                "training_wall_time_seconds": training_elapsed,
                "wall_time_seconds": wall_elapsed,
                "grad_norm": float(grad_norm),
                "combined_embedding_grad_norm": model.combined_embedding_grad_norm(),
                "peak_gpu_memory_mb": peak_memory,
                "path_ablation": path_ablation,
                **flops,
                **(gradient_metrics or {}),
                **update_metrics,
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
                "tokens_per_second": tokens_seen / max(training_elapsed, 1e-9),
                "training_wall_time_seconds": training_elapsed,
                "wall_time_seconds": wall_elapsed,
                "peak_gpu_memory_mb": peak_memory,
                **flops,
            }
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(val_record) + "\n")
            print(f"step={step:5d} val_loss={val_loss:.4f} val_ppl={val_record['perplexity']:.2f}")
            checkpoint = compact_checkpoint_payload(
                model, model_config, train_config, device, step, min(best_val, val_loss)
            )
            save_checkpoint(run_dir / "last.pt", checkpoint)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(run_dir / "best.pt", checkpoint)
            if args.save_optimizer:
                save_checkpoint(
                    run_dir / "optimizer_last.pt",
                    optimizer_checkpoint_payload(
                        model,
                        optimizer,
                        model_config,
                        train_config,
                        device,
                        step,
                        best_val,
                        tokens_seen,
                        training_elapsed,
                        embedding_cumulative_update_norm,
                        dataset.generators,
                    ),
                )
        elif step % train_config.save_interval == 0:
            save_checkpoint(
                run_dir / "last.pt",
                compact_checkpoint_payload(model, model_config, train_config, device, step, best_val),
            )
            if args.save_optimizer:
                save_checkpoint(
                    run_dir / "optimizer_last.pt",
                    optimizer_checkpoint_payload(
                        model,
                        optimizer,
                        model_config,
                        train_config,
                        device,
                        step,
                        best_val,
                        tokens_seen,
                        training_elapsed,
                        embedding_cumulative_update_norm,
                        dataset.generators,
                    ),
                )

    batch_stream_handle.close()
    print(f"finished run={train_config.run_name} best_val_loss={best_val:.4f} output={run_dir}")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["batch_stream"] = {
        "algorithm": BATCH_STREAM_ALGORITHM,
        "digest": batch_stream_digest.hexdigest(),
        "record_count": batch_record_count,
        "token_count": tokens_seen,
        "offset_log": batch_stream_path.name,
    }
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
