"""Generate Study 3 summary artifacts and figures from validated raw runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_scale import main as aggregate_main
from validate_scale_study import EXPECTED_CONDITIONS, EXPECTED_SEEDS


LABELS = {
    "tied": "Tied",
    "partial": "Partial",
    "untied": "Untied",
    "capacity_control": "Capacity control",
}
COLORS = {"tied": "#4C78A8", "partial": "#59A14F", "untied": "#E15759", "capacity_control": "#79706E"}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def metric_value(entry: dict) -> tuple[str | None, float | None]:
    """Return the first scalar benchmark metric while preserving its name."""
    result = entry.get("result") if isinstance(entry, dict) else None
    if not isinstance(result, dict):
        return None, None
    candidates = []
    for key, value in result.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            candidates.append((str(key), float(value)))
    if not candidates:
        return None, None
    preferred = [item for item in candidates if item[0].split(",")[0] in {"acc", "exact_match", "perplexity", "loglikelihood"}]
    return (preferred or candidates)[0]


def benchmark_summary(payload: dict) -> dict:
    out = {}
    for suite, tasks in payload.get("benchmarks", {}).items():
        out[suite] = {}
        for task, conditions in tasks.items():
            out[suite][task] = {}
            for condition, rows in conditions.items():
                values = []
                metric_names = []
                for row in rows:
                    name, value = metric_value(row)
                    if value is not None:
                        values.append(value)
                        metric_names.append(name)
                out[suite][task][condition] = {
                    "metric": metric_names[0] if metric_names else None,
                    "values": values,
                    "mean": statistics.mean(values) if values else None,
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
    return out


def loss_figure(result: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = list(range(len(EXPECTED_CONDITIONS)))
    for index, condition in enumerate(EXPECTED_CONDITIONS):
        rows = result["conditions"][condition]["final_loss"]["values"]
        ax.scatter([index] * len(rows), rows, color=COLORS[condition], alpha=0.8, zorder=3)
        ax.scatter(index, statistics.mean(rows), color=COLORS[condition], marker="_", s=400, linewidths=2.5, zorder=4)
    ax.set_xticks(x, [LABELS[c] for c in EXPECTED_CONDITIONS], rotation=15, ha="right")
    ax.set_ylabel("Final validation loss (lower is better)")
    ax.set_title("Study 3 scaled language-modeling result")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def pareto_figure(result: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for condition in EXPECTED_CONDITIONS:
        item = result["conditions"][condition]
        x = item["total_parameters"] / 1e6
        y = item["final_loss"]["mean"]
        ax.scatter(x, y, s=75, color=COLORS[condition], label=LABELS[condition])
        ax.annotate(LABELS[condition], (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total trainable parameters (millions)")
    ax.set_ylabel("Mean final validation loss")
    ax.set_title("Quality versus total parameter count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def mechanism_figure(result: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [LABELS[c] for c in EXPECTED_CONDITIONS]
    values = [result["conditions"][c]["mechanism"]["output_to_input_grad_ratio"]["mean"] for c in EXPECTED_CONDITIONS]
    bars = ax.bar(labels, values, color=[COLORS[c] for c in EXPECTED_CONDITIONS])
    ax.axhline(1.0, color="#333333", linewidth=1, linestyle="--")
    ax.set_ylabel("Output / input effective gradient norm")
    ax.set_title("Scaled-study role-gradient pressure")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def markdown(result: dict, benchmark: dict) -> str:
    lines = [
        "# Study 3 scaled-study summary",
        "",
        "Generated from the validated raw Kaggle artifacts. Values are not manually entered.",
        "",
        "| Condition | Final loss mean | Final loss SD | Params | Extra vs tied |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in EXPECTED_CONDITIONS:
        item = result["conditions"][condition]
        lines.append(
            f"| {LABELS[condition]} | {item['final_loss']['mean']:.5f} | {item['final_loss']['std']:.5f} | "
            f"{item['total_parameters']:,} | {item['additional_parameters_vs_tied']:,} |"
        )
    lines += ["", "## Paired final-loss differences versus tied", "", "| Condition | Mean delta | SD |", "|---|---:|---:|"]
    for condition in ("partial", "untied", "capacity_control"):
        item = result["paired_comparisons"][condition]["final_loss_minus_tied"]
        lines.append(f"| {LABELS[condition]} | {item['mean']:.5f} | {item['std']:.5f} |")
    lines += ["", "## Benchmark metric means", ""]
    for suite, tasks in benchmark.items():
        lines += [f"### {suite}", "", "| Task | Metric | " + " | ".join(LABELS[c] for c in EXPECTED_CONDITIONS) + " |", "|---|---|" + "---:|" * len(EXPECTED_CONDITIONS)]
        for task, conditions in tasks.items():
            metric = next((conditions[c]["metric"] for c in EXPECTED_CONDITIONS if conditions[c]["metric"]), "--")
            values = ["--" if conditions[c]["mean"] is None else f"{conditions[c]['mean']:.4f}" for c in EXPECTED_CONDITIONS]
            lines.append(f"| {task} | `{metric}` | " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate_dir = args.output_dir / "aggregate"
    # Keep aggregation in the same machine-readable output directory so every
    # figure and table has one auditable source.
    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["aggregate_scale.py", "--runs_dir", str(args.runs_dir), "--output_dir", str(aggregate_dir)]
        aggregate_main()
    finally:
        sys.argv = old_argv
    result = read(aggregate_dir / "scale_results.json")
    benchmarks = benchmark_summary(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scale_benchmarks.json").write_text(json.dumps(benchmarks, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "scale_summary.md").write_text(markdown(result, benchmarks))
    loss_figure(result, args.output_dir / "final_loss.png")
    pareto_figure(result, args.output_dir / "parameter_efficiency.png")
    mechanism_figure(result, args.output_dir / "gradient_ratio.png")


if __name__ == "__main__":
    main()
