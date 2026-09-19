from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_mean_ci(values: list[float], seed: int, samples: int = 2000) -> dict[str, float] | None:
    """Return a deterministic percentile bootstrap interval for a sample mean."""
    if not values:
        return None
    values = [float(value) for value in values]
    if len(values) == 1:
        return {"low": values[0], "high": values[0]}
    rng = random.Random(seed)
    means = [
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(samples)
    ]
    means.sort()
    low_index = int(0.025 * (len(means) - 1))
    high_index = int(0.975 * (len(means) - 1))
    return {"low": means[low_index], "high": means[high_index]}


def discover_runs(runs_dir: Path) -> list[tuple[str, Path, dict, list[dict]]]:
    runs = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        counts_path = run_dir / "parameter_counts.json"
        if not config_path.exists() or not counts_path.exists():
            continue
        config = json.loads(config_path.read_text())
        counts = json.loads(counts_path.read_text())
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        runs.append((run_dir.name, run_dir, {**counts, "config": config}, metrics))
    return runs


def plot_lines(runs, split: str, x_key: str, y_key: str, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, _, _, metrics in runs:
        rows = [r for r in metrics if r.get("split") == split and y_key in r]
        if rows:
            plt.plot([r[x_key] for r in rows], [r[y_key] for r in rows], marker="o", label=name)
    plt.xlabel(x_key.replace("_", " ").title())
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_gradient(runs, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for name, _, _, metrics in runs:
        rows = [r for r in metrics if r.get("split") == "train" and "input_grad_norm" in r]
        if rows:
            steps = [r["step"] for r in rows]
            axes[0].plot(steps, [r["input_grad_norm"] for r in rows], marker="o", label=f"{name} input")
            axes[0].plot(steps, [r["output_grad_norm"] for r in rows], linestyle="--", marker="x", label=f"{name} output")
            axes[1].plot(steps, [r["output_to_input_grad_ratio"] for r in rows], marker="o", label=name)
    axes[0].set_ylabel("Gradient norm")
    axes[0].set_title("Embedding-side gradient pressure")
    axes[1].set_ylabel("Output / input")
    axes[1].set_xlabel("Step")
    axes[1].set_title("Output-to-input embedding gradient ratio")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gradient_alignment(runs, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plotted = False
    for name, _, _, metrics in runs:
        rows = [r for r in metrics if r.get("split") == "train" and "input_output_grad_cosine" in r]
        if rows:
            plotted = True
            steps = [r["step"] for r in rows]
            plt.plot(steps, [r["input_output_grad_cosine"] for r in rows], marker="o", label=f"{name} all")
            if any("shared_input_output_grad_cosine" in r for r in rows):
                plt.plot(
                    steps,
                    [r.get("shared_input_output_grad_cosine", 0.0) for r in rows],
                    linestyle="--",
                    marker="x",
                    label=f"{name} shared",
                )
    plt.xlabel("Step")
    plt.ylabel("Input/output gradient cosine")
    plt.title("Input/output embedding-gradient alignment")
    plt.ylim(-1.05, 1.05)
    plt.grid(alpha=0.25)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_corrections(runs, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, _, _, metrics in runs:
        rows = [r for r in metrics if r.get("split") == "train" and "input_correction_norm" in r]
        if rows and any(r["input_correction_norm"] or r["output_correction_norm"] for r in rows):
            steps = [r["step"] for r in rows]
            plt.plot(steps, [r["input_correction_relative_norm"] for r in rows], marker="o", label=f"{name} input")
            plt.plot(steps, [r["output_correction_relative_norm"] for r in rows], linestyle="--", marker="x", label=f"{name} output")
    plt.xlabel("Step")
    plt.ylabel("Correction norm / shared norm")
    plt.title("Effective partial-tying corrections")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_validation_loss_vs_flops(runs, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name, _, _, metrics in runs:
        rows = [
            r
            for r in metrics
            if r.get("split") == "val" and "estimated_flops_total" in r
        ]
        if rows:
            plt.plot(
                [r["estimated_flops_total"] / 1e12 for r in rows],
                [r["loss"] for r in rows],
                marker="o",
                label=name,
            )
    plt.xlabel("Estimated training FLOPs (trillions)")
    plt.ylabel("Validation loss")
    plt.title("Validation loss versus estimated training compute")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def make_summary(runs, output_path: Path) -> dict:
    summary = []
    for name, run_dir, counts, metrics in runs:
        val_rows = [r for r in metrics if r.get("split") == "val"]
        train_rows = [r for r in metrics if r.get("split") == "train"]
        best = min(val_rows, key=lambda r: r["loss"]) if val_rows else None
        final = val_rows[-1] if val_rows else None
        last_train = train_rows[-1] if train_rows else {}
        last_with_flops = next((r for r in reversed(metrics) if "estimated_flops_total" in r), {})
        summary.append(
            {
                "run": name,
                "embedding_type": counts["embedding_type"],
                "total_parameters": counts["total_parameters"],
                "embedding_parameters": counts["embedding_parameters"],
                "additional_parameters_vs_tied": None,
                "best_val_loss": best["loss"] if best else None,
                "best_val_perplexity": best["perplexity"] if best else None,
                "best_val_step": best["step"] if best else None,
                "final_val_loss": final["loss"] if final else None,
                "final_val_perplexity": final["perplexity"] if final else None,
                "tokens_per_second": last_train.get("tokens_per_second"),
                "training_wall_time_seconds": last_train.get("training_wall_time_seconds"),
                "estimated_flops_total": last_with_flops.get("estimated_flops_total"),
                "estimated_flops_non_embedding": last_with_flops.get("estimated_flops_non_embedding"),
                "peak_gpu_memory_mb": max((r.get("peak_gpu_memory_mb", 0.0) for r in metrics), default=0.0),
                "run_dir": str(run_dir),
            }
        )
    tied = next((row["total_parameters"] for row in summary if row["embedding_type"] == "tied"), None)
    for row in summary:
        if tied is not None:
            row["additional_parameters_vs_tied"] = row["total_parameters"] - tied

    aggregates = {}
    for kind in sorted({row["embedding_type"] for row in summary}):
        rows = [row for row in summary if row["embedding_type"] == kind and row["best_val_loss"] is not None]
        losses = [row["best_val_loss"] for row in rows]
        perplexities = [row["best_val_perplexity"] for row in rows]
        aggregates[kind] = {
            "run_count": len(rows),
            "total_parameters": rows[0]["total_parameters"] if rows else None,
            "embedding_parameters": rows[0]["embedding_parameters"] if rows else None,
            "additional_parameters_vs_tied": rows[0]["additional_parameters_vs_tied"] if rows else None,
            "mean_best_val_loss": statistics.mean(losses) if losses else None,
            "std_best_val_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0 if losses else None,
            "mean_best_val_loss_ci95": bootstrap_mean_ci(losses, seed=1337 + len(kind)),
            "mean_best_val_perplexity": statistics.mean(perplexities) if perplexities else None,
            "mean_best_val_perplexity_ci95": bootstrap_mean_ci(perplexities, seed=2027 + len(kind)),
            "mean_final_val_loss": statistics.mean([row["final_val_loss"] for row in rows]) if rows else None,
            "mean_final_val_loss_ci95": bootstrap_mean_ci(
                [row["final_val_loss"] for row in rows], seed=31415 + len(kind)
            ),
            "mean_final_val_perplexity": statistics.mean([row["final_val_perplexity"] for row in rows]) if rows else None,
        }

    output_path.write_text(json.dumps({"runs": summary, "by_embedding_type": aggregates}, indent=2) + "\n")
    def fmt(value):
        return "—" if value is None else f"{value:.4f}" if isinstance(value, float) else f"{value:,}"

    def fmt_ci(value):
        return "—" if value is None else f"[{value['low']:.4f}, {value['high']:.4f}]"

    lines = [
        "# Embedding experiment results",
        "",
        "Lower validation loss/perplexity is better. Results are based on the logged validation checkpoints.",
        "",
        "| Model | Total params | Embedding params | Extra vs tied | Best val loss | Final val loss | Val perplexity | Tokens/s | Train s | FLOPs (T) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['embedding_type']} | {row['total_parameters']:,} | {row['embedding_parameters']:,} | "
            f"{fmt(row['additional_parameters_vs_tied'])} | {fmt(row['best_val_loss'])} | {fmt(row['final_val_loss'])} | "
            f"{fmt(row['best_val_perplexity'])} | {fmt(row['tokens_per_second'])} | "
            f"{fmt(row['training_wall_time_seconds'])} | {fmt(row['estimated_flops_total'] / 1e12 if row['estimated_flops_total'] is not None else None)} |"
        )
    lines += [
        "",
        "## Aggregate by embedding type",
        "",
        "| Model | Runs | Mean best val loss | 95% CI | Std. dev. | Mean final val loss | Mean perplexity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, aggregate in aggregates.items():
        lines.append(
            f"| {kind} | {aggregate['run_count']} | {fmt(aggregate['mean_best_val_loss'])} | "
            f"{fmt_ci(aggregate['mean_best_val_loss_ci95'])} | {fmt(aggregate['std_best_val_loss'])} | "
            f"{fmt(aggregate['mean_final_val_loss'])} | "
            f"{fmt(aggregate['mean_best_val_perplexity'])} |"
        )
    lines += [
        "",
        "## Questions",
        "",
        "The statements below are computed from the logged runs and do not assume that untying or partial tying wins.",
    ]
    by_type = aggregates
    if all(kind in by_type and by_type[kind]["mean_best_val_loss"] is not None for kind in ("tied", "untied", "partial")):
        tied_row = by_type["tied"]
        untied_row = by_type["untied"]
        partial_row = by_type["partial"]
        untied_delta = tied_row["mean_best_val_loss"] - untied_row["mean_best_val_loss"]
        partial_vs_tied = tied_row["mean_best_val_loss"] - partial_row["mean_best_val_loss"]
        partial_vs_untied = partial_row["mean_best_val_loss"] - untied_row["mean_best_val_loss"]
        direction = "lower" if untied_delta > 0 else "higher"
        partial_direction = "lower" if partial_vs_tied > 0 else "higher"
        gap_direction = "higher" if partial_vs_untied > 0 else "lower"
        lines += [
            "",
            f"- Fully untied is {abs(untied_delta):.4f} validation-loss units {direction} than tied, at "
            f"{untied_row['additional_parameters_vs_tied']:,} additional parameters.",
            f"- Partial tying is {abs(partial_vs_tied):.4f} units {partial_direction} than tied, at "
            f"{partial_row['additional_parameters_vs_tied']:,} additional parameters. It is "
            f"{abs(partial_vs_untied):.4f} units {gap_direction} than untied.",
        ]
    else:
        lines.append("- A complete tied/partial/untied comparison is unavailable in the discovered runs.")

    for kind in ("tied", "untied", "partial"):
        ratios = [
            r["output_to_input_grad_ratio"]
            for _, _, counts, metrics in runs
            if counts["embedding_type"] == kind
            for r in metrics
            if r.get("split") == "train" and "output_to_input_grad_ratio" in r
        ]
        if ratios:
            lines.append(f"- {kind} output/input embedding-gradient ratio: mean {sum(ratios) / len(ratios):.3f}, final {ratios[-1]:.3f}.")
    partial_train = [
        r
        for _, _, counts, metrics in runs
        if counts["embedding_type"] == "partial"
        for r in metrics
        if r.get("split") == "train" and "input_correction_relative_norm" in r
    ]
    if partial_train:
        final = partial_train[-1]
        lines.append(
            f"- Partial correction evidence: final input/output correction norms are "
            f"{final['input_correction_relative_norm']:.4f}/{final['output_correction_relative_norm']:.4f} times the shared norm."
        )
    lines += [
        "",
        "Compute-aware plot: `validation_loss_vs_estimated_flops.png`; gradient plots: `gradient_norms_and_ratio.png` and `gradient_alignment.png`; correction plot: `correction_norms.png`.",
    ]
    (output_path.parent / "results_summary.md").write_text("\n".join(lines) + "\n")
    return {"runs": summary, "by_embedding_type": aggregates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and summarize embedding experiment runs")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--output_dir", default="analysis")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No complete runs found under {runs_dir}")
    plot_lines(runs, "train", "step", "loss", "Training loss", "Loss", output_dir / "training_loss.png")
    plot_lines(runs, "val", "step", "loss", "Validation loss", "Loss", output_dir / "validation_loss.png")
    plot_validation_loss_vs_flops(runs, output_dir / "validation_loss_vs_estimated_flops.png")
    plot_gradient(runs, output_dir / "gradient_norms_and_ratio.png")
    plot_gradient_alignment(runs, output_dir / "gradient_alignment.png")
    plot_corrections(runs, output_dir / "correction_norms.png")

    plt.figure(figsize=(7, 5))
    for name, _, counts, metrics in runs:
        val_rows = [r for r in metrics if r.get("split") == "val"]
        if val_rows:
            best = min(val_rows, key=lambda r: r["loss"])
            plt.scatter(counts["total_parameters"] / 1e6, best["loss"], s=80, label=name)
    plt.xlabel("Total parameters (millions)")
    plt.ylabel("Best validation loss")
    plt.title("Parameter count versus validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_count_vs_validation_loss.png", dpi=160)
    plt.close()

    summary = make_summary(runs, output_dir / "results.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
