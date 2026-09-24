"""Aggregate validated per-seed cloud studies into publication artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import bootstrap_mean_ci


PRIMARY_CONDITIONS = ["tied", "untied", "partial", "capacity_control", "partial_input", "partial_output"]
SECONDARY_METRICS = (
    "best_val_perplexity",
    "final_val_perplexity",
    "training_tokens",
    "tokens_per_second",
    "training_wall_time_seconds",
    "peak_gpu_memory_mb",
    "estimated_flops_total",
    "estimated_flops_non_embedding",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_run(study_dir: Path, run_dir: Path) -> dict:
    config = read_json(run_dir / "config.json")
    counts = read_json(run_dir / "parameter_counts.json")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    validation = [row for row in metrics if row.get("split") == "val"]
    training = [row for row in metrics if row.get("split") == "train"]
    if not validation:
        raise ValueError(f"missing validation metrics: {run_dir}")
    best = min(validation, key=lambda row: float(row["loss"]))
    final = validation[-1]
    last_train = training[-1] if training else {}
    return {
        "seed": int(config["train"]["seed"]),
        "embedding_type": counts["embedding_type"],
        "run_name": run_dir.name,
        "study_dir": str(study_dir),
        "total_parameters": int(counts["total_parameters"]),
        "trainable_parameters": int(counts["trainable_parameters"]),
        "embedding_parameters": int(counts["embedding_parameters"]),
        "transformer_parameters": int(counts["transformer_parameters"]),
        "best_val_loss": float(best["loss"]),
        "best_val_perplexity": float(best["perplexity"]),
        "best_val_step": int(best["step"]),
        "final_val_loss": float(final["loss"]),
        "final_val_perplexity": float(final["perplexity"]),
        "final_val_step": int(final["step"]),
        "training_tokens": int(final.get("tokens_seen", last_train.get("tokens_seen", 0))),
        # Validation records are emitted at the declared endpoint and carry
        # the cumulative runtime. Prefer them over the last logged train row,
        # which may precede the endpoint when log/eval intervals differ.
        "tokens_per_second": final.get("tokens_per_second", last_train.get("tokens_per_second")),
        "training_wall_time_seconds": final.get("training_wall_time_seconds", last_train.get("training_wall_time_seconds")),
        "peak_gpu_memory_mb": max((float(row.get("peak_gpu_memory_mb", 0.0)) for row in metrics), default=0.0),
        "estimated_flops_total": float(final.get("estimated_flops_total", last_train.get("estimated_flops_total", 0.0))),
        "estimated_flops_non_embedding": float(final.get("estimated_flops_non_embedding", last_train.get("estimated_flops_non_embedding", 0.0))),
    }


def validate_and_load(study_dirs: list[Path]) -> tuple[list[dict], dict]:
    if not study_dirs:
        raise SystemExit("no study directories supplied")
    all_rows: list[dict] = []
    reference_common = None
    reference_commit = None
    expected_seeds: list[int] = []
    expected_conditions: list[str] = []
    for study_dir in study_dirs:
        validation = read_json(study_dir / "study_validation.json")
        if not validation.get("passed"):
            raise SystemExit(f"fairness validation failed: {study_dir}")
        manifest = read_json(study_dir / "study_manifest.json")
        common = manifest.get("common", {})
        if reference_common is None:
            reference_common = common
            expected_conditions = list(manifest.get("embedding_types", []))
            expected_seeds = list(manifest.get("seeds", []))
        elif common != reference_common:
            raise SystemExit(f"shared study configuration differs: {study_dir}")
        if list(manifest.get("embedding_types", [])) != expected_conditions:
            raise SystemExit(f"condition list differs: {study_dir}")
        if len(manifest.get("seeds", [])) != 1:
            raise SystemExit(f"each collected artifact must contain exactly one seed: {study_dir}")
        seed = int(manifest["seeds"][0])
        if seed in expected_seeds and len(study_dirs) > 1:
            pass
        for run_dir in sorted(path for path in study_dir.iterdir() if (path / "config.json").exists()):
            all_rows.append(summarize_run(study_dir, run_dir))
        # The collector copies the kernel's artifact root directly into the
        # destination, so provenance lives alongside study_manifest.json.
        artifact_manifest = study_dir / "artifact_manifest.json"
        if artifact_manifest.exists():
            commit = read_json(artifact_manifest).get("git_commit")
            if reference_commit is None:
                reference_commit = commit
            elif commit != reference_commit:
                raise SystemExit("collected artifact commits differ")

    expected = {(int(seed), condition) for seed in [row["seed"] for row in all_rows] for condition in expected_conditions}
    observed = {(row["seed"], row["embedding_type"]) for row in all_rows}
    if observed != expected:
        raise SystemExit(f"incomplete seed x condition matrix: missing={sorted(expected - observed)} extra={sorted(observed - expected)}")
    if len({row["seed"] for row in all_rows}) != len(study_dirs):
        raise SystemExit("duplicate or missing collected seed artifacts")
    return all_rows, {
        "common": reference_common,
        "conditions": expected_conditions,
        "seeds": sorted({row["seed"] for row in all_rows}),
        "git_commit": reference_commit,
    }


def paired_delta(rows: list[dict], condition: str, metric: str) -> list[float]:
    by_seed = {(row["seed"], row["embedding_type"]): row for row in rows}
    seeds = sorted({row["seed"] for row in rows})
    return [
        float(by_seed[(seed, condition)][metric]) - float(by_seed[(seed, "tied")][metric])
        for seed in seeds
    ]


def summarize(rows: list[dict], metadata: dict) -> dict:
    tied_params = next(row["total_parameters"] for row in rows if row["embedding_type"] == "tied")
    by_condition = {}
    for condition in metadata["conditions"]:
        condition_rows = [row for row in rows if row["embedding_type"] == condition]
        final_losses = [row["final_val_loss"] for row in condition_rows]
        best_losses = [row["best_val_loss"] for row in condition_rows]
        by_condition[condition] = {
            "seed_values": condition_rows,
            "run_count": len(condition_rows),
            "total_parameters": condition_rows[0]["total_parameters"],
            "embedding_parameters": condition_rows[0]["embedding_parameters"],
            "additional_parameters_vs_tied": condition_rows[0]["total_parameters"] - tied_params,
            "mean_final_val_loss": statistics.mean(final_losses),
            "std_final_val_loss": statistics.stdev(final_losses) if len(final_losses) > 1 else 0.0,
            "ci95_final_val_loss": bootstrap_mean_ci(final_losses, seed=4100 + len(condition)),
            "mean_best_val_loss": statistics.mean(best_losses),
            "std_best_val_loss": statistics.stdev(best_losses) if len(best_losses) > 1 else 0.0,
            "ci95_best_val_loss": bootstrap_mean_ci(best_losses, seed=4200 + len(condition)),
        }
        for metric in SECONDARY_METRICS:
            values = [float(row[metric]) for row in condition_rows if row.get(metric) is not None]
            if values:
                by_condition[condition][f"mean_{metric}"] = statistics.mean(values)
                by_condition[condition][f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
                by_condition[condition][f"ci95_{metric}"] = bootstrap_mean_ci(
                    values, seed=7000 + len(condition) + len(metric)
                )

    paired = {}
    for condition in metadata["conditions"]:
        if condition == "tied":
            continue
        for metric in ("final_val_loss", "best_val_loss"):
            deltas = paired_delta(rows, condition, metric)
            paired[f"{condition}_minus_tied_{metric}"] = {
                "condition": condition,
                "metric": metric,
                "seeds": sorted({row["seed"] for row in rows}),
                "deltas": deltas,
                "mean_delta": statistics.mean(deltas),
                "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "ci95": bootstrap_mean_ci(deltas, seed=5100 + len(condition) + len(metric)),
            }

    research_question = {}
    for metric in ("final_val_loss", "best_val_loss"):
        tied = by_condition["tied"][f"mean_{metric}"]
        untied = by_condition["untied"][f"mean_{metric}"]
        partial = by_condition["partial"][f"mean_{metric}"]
        untied_delta = tied - untied
        partial_delta = tied - partial
        paired_entry = paired[f"untied_minus_tied_{metric}"]
        paired_ci = paired_entry["ci95"]
        distinguishable = (
            len(paired_entry["deltas"]) >= 2
            and paired_ci
            and (paired_ci["low"] > 0 or paired_ci["high"] < 0)
        )
        denominator_ok = untied_delta > 1e-8 and bool(distinguishable)
        if denominator_ok:
            recovery_status = "reported"
        elif not distinguishable:
            recovery_status = "statistically_uncertain_or_zero_denominator"
        else:
            recovery_status = "untied_not_better_than_tied"
        research_question[metric] = {
            "tied": tied,
            "partial": partial,
            "untied": untied,
            "untied_improvement_over_tied": untied_delta,
            "partial_improvement_over_tied": partial_delta,
            "partial_to_untied_gap": partial - untied,
            "recovered_fraction": partial_delta / untied_delta if denominator_ok else None,
            "recovery_status": recovery_status,
        }

    return {
        "metadata": metadata,
        "runs": rows,
        "by_condition": by_condition,
        "paired_comparisons": paired,
        "research_question": research_question,
    }


def plot_results(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = result["metadata"]["conditions"]
    by_condition = result["by_condition"]

    plt.figure(figsize=(9, 5))
    for condition in conditions:
        values = [row["final_val_loss"] for row in by_condition[condition]["seed_values"]]
        x = [by_condition[condition]["additional_parameters_vs_tied"] / 1e6] * len(values)
        plt.scatter(x, values, alpha=0.7, label=condition)
        plt.scatter(
            [by_condition[condition]["additional_parameters_vs_tied"] / 1e6],
            [by_condition[condition]["mean_final_val_loss"]],
            marker="D", s=70, edgecolors="black",
        )
    plt.xlabel("Additional parameters versus tied (millions)")
    plt.ylabel("Final validation loss")
    plt.title("Final quality versus parameter overhead")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_efficiency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    means = [by_condition[c]["mean_final_val_loss"] for c in conditions]
    errors = [by_condition[c]["std_final_val_loss"] for c in conditions]
    plt.errorbar(range(len(conditions)), means, yerr=errors, fmt="o", capsize=4)
    plt.xticks(range(len(conditions)), conditions, rotation=25, ha="right")
    plt.ylabel("Final validation loss")
    plt.title("Primary conditions across paired seeds")
    plt.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "final_loss_by_condition.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    for condition in conditions:
        grouped: dict[int, list[float]] = {}
        for row in result["runs"]:
            if row["embedding_type"] != condition:
                continue
            metrics = read_jsonl(Path(row["study_dir"]) / row["run_name"] / "metrics.jsonl")
            for metric in metrics:
                if metric.get("split") == "val":
                    grouped.setdefault(int(metric["step"]), []).append(float(metric["loss"]))
        steps = sorted(grouped)
        means = [statistics.mean(grouped[step]) for step in steps]
        plt.plot(steps, means, label=condition)
    plt.xlabel("Optimizer step")
    plt.ylabel("Validation loss")
    plt.title("Validation loss over training")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "validation_loss_over_steps.png", dpi=180)
    plt.close()


def write_summary(result: dict, path: Path) -> None:
    lines = [
        "# Primary study aggregate",
        "",
        "Generated directly from validated per-seed JSONL artifacts. Lower loss is better.",
        "",
        "| Condition | Seeds | Total params | Extra vs tied | Mean final loss | Mean final PPL | Tokens/s | Wall s | Peak MB | FLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, values in result["by_condition"].items():
        lines.append(
            f"| {condition} | {values['run_count']} | {values['total_parameters']:,} | "
            f"{values['additional_parameters_vs_tied']:,} | {values['mean_final_val_loss']:.6f} | "
            f"{values.get('mean_final_val_perplexity', float('nan')):.4f} | "
            f"{values.get('mean_tokens_per_second', float('nan')):.2f} | "
            f"{values.get('mean_training_wall_time_seconds', float('nan')):.1f} | "
            f"{values.get('mean_peak_gpu_memory_mb', float('nan')):.1f} | "
            f"{values.get('mean_estimated_flops_total', float('nan')):.3e} |"
        )
    lines += ["", "## Paired differences versus tied", "", "| Condition | Metric | Mean delta | 95% interval |", "|---|---|---:|---:|"]
    for values in result["paired_comparisons"].values():
        ci = values["ci95"]
        lines.append(f"| {values['condition']} | {values['metric']} | {values['mean_delta']:.6f} | [{ci['low']:.6f}, {ci['high']:.6f}] |")
    lines += ["", "## Research question", "", "```json", json.dumps(result["research_question"], indent=2, sort_keys=True), "```", ""]
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True, help="collected per-seed study directories")
    parser.add_argument("--output_dir", default="results/primary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study_dirs = [Path(value) for value in args.studies]
    rows, metadata = validate_and_load(study_dirs)
    result = summarize(rows, metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "primary_aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_summary(result, output_dir / "primary_aggregate.md")
    plot_results(result, output_dir / "figures")
    print(json.dumps(result["research_question"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
