"""Run and summarize a reproducible partial-embedding rank/scaling sweep."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import bootstrap_mean_ci


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def label_number(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def condition_name(rank: int, alpha: float) -> str:
    return f"rank{rank}_alpha{label_number(alpha)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/adapter-study-001")
    parser.add_argument("--dataset", choices=["wikitext2", "tiny_shakespeare"], default="wikitext2")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337])
    parser.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--alphas", nargs="+", type=float, default=[8.0])
    parser.add_argument("--baseline_run_dir", default="", help="optional tied run directory for extra-parameter comparisons")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument(
        "--save_optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write optimizer checkpoints for rank-sweep resume (default: false)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sweep_command(args: argparse.Namespace, condition_dir: Path, rank: int, alpha: float) -> list[str]:
    command = [
        sys.executable,
        "sweep.py",
        "--output_dir",
        str(condition_dir),
        "--dataset",
        args.dataset,
        "--data_dir",
        args.data_dir,
        "--device",
        args.device,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--embedding_types",
        "partial",
        "--steps",
        str(args.steps),
        "--batch_size",
        str(args.batch_size),
        "--grad_accum_steps",
        str(args.grad_accum_steps),
        "--block_size",
        str(args.block_size),
        "--n_layer",
        str(args.n_layer),
        "--n_head",
        str(args.n_head),
        "--n_embd",
        str(args.n_embd),
        "--dropout",
        str(args.dropout),
        "--adapter_rank",
        str(rank),
        "--adapter_alpha",
        str(alpha),
        "--learning_rate",
        str(args.learning_rate),
        "--min_learning_rate",
        str(args.min_learning_rate),
        "--warmup_steps",
        str(args.warmup_steps),
        "--weight_decay",
        str(args.weight_decay),
        "--grad_clip",
        str(args.grad_clip),
        "--eval_interval",
        str(args.eval_interval),
        "--eval_batches",
        str(args.eval_batches),
        "--log_interval",
        str(args.log_interval),
        "--save_interval",
        str(args.save_interval),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if not args.save_optimizer:
        command.append("--no-save_optimizer")
    return command


def summarize_condition(
    condition_dir: Path,
    rank: int,
    alpha: float,
    tied_counts: dict | None = None,
) -> dict:
    validation = json.loads((condition_dir / "study_validation.json").read_text())
    if not validation.get("passed"):
        raise SystemExit(f"Fairness validation failed for {condition_dir}")
    results = json.loads((condition_dir / "analysis" / "results.json").read_text())
    rows = results.get("runs", [])
    best_losses = [row["best_val_loss"] for row in rows if row.get("best_val_loss") is not None]
    final_losses = [row["final_val_loss"] for row in rows if row.get("final_val_loss") is not None]
    best_perplexities = [row["best_val_perplexity"] for row in rows if row.get("best_val_perplexity") is not None]
    if not best_losses:
        raise SystemExit(f"No validation results found for {condition_dir}")
    first = rows[0]
    if tied_counts is None:
        # All rank conditions use the same Transformer. Infer the tied total
        # from the condition's transformer count and one vocabulary matrix.
        config = json.loads((condition_dir / rows[0]["run_name"] / "config.json").read_text())
        model_config = config["model"]
        tied_embedding_parameters = int(model_config["vocab_size"]) * int(model_config["n_embd"])
        tied_total_parameters = int(first["total_parameters"]) - int(first["embedding_parameters"]) + tied_embedding_parameters
    else:
        tied_total_parameters = int(tied_counts["total_parameters"])
        expected_transformer = int(first["total_parameters"]) - int(first["embedding_parameters"])
        actual_transformer = int(tied_counts["total_parameters"]) - int(tied_counts["embedding_parameters"])
        if expected_transformer != actual_transformer:
            raise SystemExit(f"Tied baseline Transformer count differs for {condition_dir}")
    additional_parameters = int(first["total_parameters"]) - tied_total_parameters
    return {
        "condition": condition_dir.name,
        "adapter_rank": rank,
        "adapter_alpha": alpha,
        "run_count": len(rows),
        "total_parameters": first["total_parameters"],
        "embedding_parameters": first["embedding_parameters"],
        "additional_parameters_vs_tied": additional_parameters,
        "mean_best_val_loss": statistics.mean(best_losses),
        "std_best_val_loss": statistics.stdev(best_losses) if len(best_losses) > 1 else 0.0,
        "mean_best_val_loss_ci95": bootstrap_mean_ci(best_losses, seed=1000 + rank),
        "mean_final_val_loss": statistics.mean(final_losses) if final_losses else None,
        "mean_best_val_perplexity": statistics.mean(best_perplexities) if best_perplexities else None,
        "mean_training_wall_time_seconds": statistics.mean(
            [row["training_wall_time_seconds"] for row in rows if row.get("training_wall_time_seconds") is not None]
        ) if any(row.get("training_wall_time_seconds") is not None for row in rows) else None,
        "mean_estimated_flops_total": statistics.mean(
            [row["estimated_flops_total"] for row in rows if row.get("estimated_flops_total") is not None]
        ) if any(row.get("estimated_flops_total") is not None for row in rows) else None,
        "run_dir": str(condition_dir),
    }


def comparable_config(config: dict) -> dict:
    variable_train_keys = {"seed", "run_name", "embedding_type", "output_dir"}
    variable_model_keys = {"adapter_rank", "adapter_alpha"}
    return {
        "model": {key: value for key, value in config["model"].items() if key not in variable_model_keys},
        "train": {key: value for key, value in config["train"].items() if key not in variable_train_keys},
        "device": config.get("device"),
    }


def validate_cross_condition_manifests(condition_dirs: list[Path], ranks: list[int], alphas: list[float]) -> None:
    reference = None
    expected_seeds = None
    for condition_dir, rank, alpha in zip(condition_dirs, ranks, alphas):
        study = json.loads((condition_dir / "study_manifest.json").read_text())
        common = dict(study.get("common", {}))
        common.pop("adapter_rank", None)
        common.pop("adapter_alpha", None)
        seeds = list(study.get("seeds", []))
        if reference is None:
            reference = common
            expected_seeds = seeds
        elif common != reference:
            raise SystemExit(f"Shared settings differ across adapter conditions at {condition_dir}")
        if seeds != expected_seeds:
            raise SystemExit(f"Seed list differs across adapter conditions at {condition_dir}")


def write_summary(output_dir: Path, summaries: list[dict]) -> None:
    write_json(output_dir / "adapter_sweep_summary.json", {"conditions": summaries})
    lines = [
        "# Adapter sweep results",
        "",
        "Lower validation loss is better. Each condition passed its own study fairness gate.",
        "",
        "| Rank | Alpha | Runs | Total params | Extra vs tied | Mean best loss | 95% CI | Mean final loss |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        ci = row["mean_best_val_loss_ci95"]
        ci_text = f"[{ci['low']:.4f}, {ci['high']:.4f}]" if ci else "—"
        extra = "—" if row["additional_parameters_vs_tied"] is None else f"{row['additional_parameters_vs_tied']:,}"
        final = "—" if row["mean_final_val_loss"] is None else f"{row['mean_final_val_loss']:.4f}"
        lines.append(
            f"| {row['adapter_rank']} | {row['adapter_alpha']:g} | {row['run_count']} | "
            f"{row['total_parameters']:,} | {extra} | {row['mean_best_val_loss']:.4f} | {ci_text} | {final} |"
        )
    (output_dir / "adapter_sweep_summary.md").write_text("\n".join(lines) + "\n")

    plt.figure(figsize=(8, 5))
    for row in summaries:
        x = row["additional_parameters_vs_tied"]
        if x is None:
            x = row["total_parameters"]
        plt.scatter(x / 1e6, row["mean_best_val_loss"], s=90, label=f"r{row['adapter_rank']} α{row['adapter_alpha']:g}")
    plt.xlabel("Additional parameters vs tied (millions)" if any(row["additional_parameters_vs_tied"] is not None for row in summaries) else "Total parameters (millions)")
    plt.ylabel("Mean best validation loss")
    plt.title("Partial-embedding adapter budget sweep")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "adapter_tradeoff.png", dpi=160)
    plt.close()

    plots = [
        ("additional_parameters_vs_tied", "Additional trainable parameters vs tied", 1e6, "rank_sweep_vs_extra_parameters.png"),
        ("total_parameters", "Total trainable parameters", 1e6, "rank_sweep_vs_total_parameters.png"),
        ("mean_estimated_flops_total", "Estimated training FLOPs", 1e12, "rank_sweep_vs_estimated_flops.png"),
        ("mean_training_wall_time_seconds", "Training wall-clock seconds", 1.0, "rank_sweep_vs_wall_clock.png"),
    ]
    for x_key, xlabel, scale, filename in plots:
        plt.figure(figsize=(8, 5))
        plotted = False
        for row in summaries:
            x = row.get(x_key)
            y = row.get("mean_final_val_loss")
            if x is None or y is None:
                continue
            plotted = True
            plt.scatter(x / scale, y, s=90, label=f"r{row['adapter_rank']} α{row['adapter_alpha']:g}")
        plt.xlabel(f"{xlabel}{' (millions)' if scale == 1e6 else ' (trillions)' if scale == 1e12 else ''}")
        plt.ylabel("Mean final validation loss")
        plt.title(f"Rank sweep: validation loss vs {xlabel.lower()}")
        plt.grid(alpha=0.25)
        if plotted:
            plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_dirs: list[Path] = []
    condition_ranks: list[int] = []
    condition_alphas: list[float] = []
    for rank in args.ranks:
        if rank <= 0:
            raise SystemExit("adapter ranks must be positive")
        for alpha in args.alphas:
            condition = condition_name(rank, alpha)
            condition_dir = output_dir / condition
            command = sweep_command(args, condition_dir, rank, alpha)
            print(f"\n=== {condition} ===", flush=True)
            subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)
            condition_dirs.append(condition_dir)
            condition_ranks.append(rank)
            condition_alphas.append(alpha)

    validate_cross_condition_manifests(condition_dirs, condition_ranks, condition_alphas)
    tied_counts = None
    if args.baseline_run_dir:
        tied_counts_path = Path(args.baseline_run_dir) / "parameter_counts.json"
        if not tied_counts_path.exists():
            raise SystemExit(f"missing tied baseline parameter counts: {tied_counts_path}")
        tied_counts = json.loads(tied_counts_path.read_text())
    summaries = [
        summarize_condition(condition_dir, rank, alpha, tied_counts=tied_counts)
        for condition_dir, rank, alpha in zip(condition_dirs, condition_ranks, condition_alphas)
    ]
    if args.baseline_run_dir:
        baseline_dir = Path(args.baseline_run_dir)
        baseline_counts = json.loads((baseline_dir / "parameter_counts.json").read_text())
        if baseline_counts.get("embedding_type") != "tied":
            raise SystemExit("--baseline_run_dir must point to a tied run")
        baseline_config = json.loads((baseline_dir / "config.json").read_text())
        first_condition_config = json.loads(
            (condition_dirs[0] / "config.json").read_text()
            if (condition_dirs[0] / "config.json").exists()
            else (next(condition_dirs[0].glob("*/config.json"))).read_text()
        )
        if comparable_config(baseline_config) != comparable_config(first_condition_config):
            raise SystemExit("--baseline_run_dir does not match the adapter sweep model/training configuration")
        tied_total = baseline_counts["total_parameters"]
        for summary in summaries:
            summary["additional_parameters_vs_tied"] = summary["total_parameters"] - tied_total
    write_summary(output_dir, summaries)
    print(f"Completed {len(summaries)} adapter conditions under {output_dir}")


if __name__ == "__main__":
    main()
