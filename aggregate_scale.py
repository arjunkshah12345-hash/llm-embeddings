"""Aggregate validated Study 3 language-model and benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from validate_scale_study import EXPECTED_CONDITIONS, EXPECTED_SEEDS, validate


def percentile_bootstrap(values: list[float], seed: int = 1729, samples: int = 20_000) -> dict:
    if not values:
        return {"low": None, "high": None}
    import random

    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples)]
    means.sort()
    return {"low": means[int(0.025 * samples)], "high": means[int(0.975 * samples) - 1]}


def condition_runs(root: Path, condition: str) -> list[dict]:
    runs = []
    for seed in EXPECTED_SEEDS:
        cell = root / f"scale3_fineweb_20m_seed{seed}_{condition}"
        manifest = json.loads((cell / "artifact_manifest.json").read_text())
        training = json.loads((cell / "training" / condition / "manifest.json").read_text())
        rows = [json.loads(line) for line in (cell / "training" / condition / "metrics.jsonl").read_text().splitlines() if line.strip()]
        vals = [row for row in rows if row.get("split") == "val"]
        final = next(row for row in vals if int(row["step"]) == 19_999)
        best = min(vals, key=lambda row: float(row["loss"]))
        train_rows = [row for row in rows if row.get("split") == "train"]
        mechanism = train_rows[-1] if train_rows else {}
        counts = training["parameter_counts"]
        benchmark = {}
        for suite in ("core", "extended"):
            benchmark[suite] = json.loads((cell / "benchmarks" / f"lm_eval_{suite}.json").read_text())
        runs.append({
            "seed": seed,
            "condition": condition,
            "manifest": manifest,
            "training": training,
            "final": final,
            "best": best,
            "mechanism": mechanism,
            "parameter_counts": counts,
            "benchmarks": benchmark,
        })
    return runs


def summarize(values: list[float]) -> dict:
    return {
        "values": values,
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "descriptive_bootstrap_interval": percentile_bootstrap(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    validation = validate(args.runs_dir)
    if not validation["passed"]:
        raise SystemExit("Study 3 fairness validation failed; refusing to aggregate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {condition: condition_runs(args.runs_dir, condition) for condition in EXPECTED_CONDITIONS}
    tied_by_seed = {row["seed"]: row for row in runs["tied"]}
    by_condition = {}
    for condition, rows in runs.items():
        final_losses = [float(row["final"]["loss"]) for row in rows]
        best_losses = [float(row["best"]["loss"]) for row in rows]
        counts = rows[0]["parameter_counts"]
        tied_counts = runs["tied"][0]["parameter_counts"]
        by_condition[condition] = {
            "final_loss": summarize(final_losses),
            "best_loss": summarize(best_losses),
            "final_perplexity": summarize([float(row["final"]["perplexity"]) for row in rows]),
            "total_parameters": counts["total_parameters"],
            "embedding_parameters": counts["embedding_parameters"],
            "additional_parameters_vs_tied": counts["total_parameters"] - tied_counts["total_parameters"],
            "estimated_flops": rows[0]["final"].get("estimated_flops_total"),
            "tokens_per_second": summarize([float(row["final"]["tokens_per_second"]) for row in rows]),
            "wall_time_seconds": summarize([float(row["final"]["training_wall_time_seconds"]) for row in rows]),
            "peak_gpu_memory_mb": summarize([float(row["final"].get("peak_gpu_memory_mb", 0.0)) for row in rows]),
            "mechanism": {
                key: summarize([float(row["mechanism"].get(key, 0.0)) for row in rows])
                for key in ("input_grad_norm", "output_grad_norm", "output_to_input_grad_ratio", "input_output_grad_cosine", "input_correction_norm", "output_correction_norm", "input_output_correction_cosine")
            },
        }
    paired = {}
    for condition in ("partial", "untied", "capacity_control"):
        deltas = [float(row["final"]["loss"]) - float(tied_by_seed[row["seed"]]["final"]["loss"]) for row in runs[condition]]
        paired[condition] = {"final_loss_minus_tied": summarize(deltas), "by_seed": dict(zip(EXPECTED_SEEDS, deltas))}
        best_deltas = [float(row["best"]["loss"]) - float(tied_by_seed[row["seed"]]["best"]["loss"]) for row in runs[condition]]
        paired[condition]["best_loss_minus_tied"] = summarize(best_deltas)
    benchmarks = {}
    for suite in ("core", "extended"):
        tasks = sorted({task for condition in EXPECTED_CONDITIONS for row in runs[condition] for task in row["benchmarks"][suite]["results"]})
        benchmarks[suite] = {}
        for task in tasks:
            benchmarks[suite][task] = {}
            for condition in EXPECTED_CONDITIONS:
                entries = []
                for row in runs[condition]:
                    entry = row["benchmarks"][suite]["results"].get(task, {})
                    entries.append({"seed": row["seed"], **entry})
                benchmarks[suite][task][condition] = entries
    payload = {
        "study": "scale3_fineweb_20m",
        "validation": validation,
        "conditions": by_condition,
        "paired_comparisons": paired,
        "benchmarks": benchmarks,
        "protocol": "EXPERIMENT_PROTOCOL_SCALE.md",
    }
    (args.output_dir / "scale_results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "scale_runs.json").write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
