"""Aggregate deterministic input/output embedding evaluations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import bootstrap_mean_ci


METRIC_PATHS = {
    "frequency_probe_accuracy": ("frequency_bucket_probe", "accuracy"),
    "frequency_probe_macro_accuracy": ("frequency_bucket_probe", "macro_accuracy"),
    "frequency_neighbor_agreement": ("nearest_neighbors", "mean_frequency_bucket_agreement"),
    "neighbor_cosine": ("nearest_neighbors", "mean_neighbor_cosine"),
    "shape_probe_accuracy": ("token_shape_probe", "accuracy"),
    "shape_probe_macro_accuracy": ("token_shape_probe", "macro_accuracy"),
    "frequency_linear_probe_accuracy": ("frequency_bucket_linear_probe", "accuracy"),
    "frequency_linear_probe_macro_accuracy": ("frequency_bucket_linear_probe", "macro_accuracy"),
    "shape_linear_probe_accuracy": ("token_shape_linear_probe", "accuracy"),
    "shape_linear_probe_macro_accuracy": ("token_shape_linear_probe", "macro_accuracy"),
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_records(study_dirs: list[Path]) -> tuple[list[dict], dict]:
    records: list[dict] = []
    conditions: list[str] | None = None
    seeds: list[int] = []
    commits: set[str] = set()
    for study_dir in study_dirs:
        validation = read_json(study_dir / "study_validation.json")
        if not validation.get("passed"):
            raise SystemExit(f"fairness validation failed: {study_dir}")
        study = read_json(study_dir / "study_manifest.json")
        study_conditions = list(study.get("embedding_types", []))
        if conditions is None:
            conditions = study_conditions
        elif conditions != study_conditions:
            raise SystemExit(f"condition list differs: {study_dir}")
        study_seeds = [int(seed) for seed in study.get("seeds", [])]
        if len(study_seeds) != 1:
            raise SystemExit(f"each artifact must contain one seed: {study_dir}")
        seed = study_seeds[0]
        seeds.append(seed)
        commits.add(str(read_json(study_dir / "artifact_manifest.json").get("git_commit")))
        for run_dir in sorted(path for path in study_dir.iterdir() if (path / "embedding_eval.json").exists()):
            evaluation = read_json(run_dir / "embedding_eval.json")
            for side, metrics in evaluation.get("metrics", {}).items():
                row = {"seed": seed, "condition": evaluation["embedding_type"], "side": side}
                for name, path in METRIC_PATHS.items():
                    value = metrics
                    for key in path:
                        value = value.get(key) if isinstance(value, dict) else None
                    if value is not None:
                        row[name] = float(value)
                for relation, value in metrics.get("pair_evaluation", {}).get("mean_cosine_by_relation", {}).items():
                    row[f"pair_cosine::{relation}"] = float(value)
                records.append(row)
    if len(seeds) != len(set(seeds)):
        raise SystemExit("duplicate representation seed artifacts")
    if len(commits) != 1:
        raise SystemExit("representation artifact commits differ")
    expected = {(seed, condition, side) for seed in seeds for condition in conditions or [] for side in ("input", "output")}
    observed = {(row["seed"], row["condition"], row["side"]) for row in records}
    if observed != expected:
        raise SystemExit(f"incomplete representation matrix: missing={sorted(expected - observed)}")
    return records, {
        "conditions": conditions or [],
        "seeds": sorted(seeds),
        "git_commit": next(iter(commits)),
        "evaluation": "best checkpoint, deterministic probes, both effective embedding sides",
    }


def summarize(records: list[dict], metadata: dict) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["condition"], row["side"])].append(row)
    final: dict[str, dict] = {}
    metric_names = sorted({key for row in records for key in row if key not in {"seed", "condition", "side"}})
    for condition in metadata["conditions"]:
        final[condition] = {}
        for side in ("input", "output"):
            rows = grouped[(condition, side)]
            metrics = {}
            for metric in metric_names:
                values = [float(row[metric]) for row in rows if metric in row]
                if not values:
                    continue
                metrics[metric] = {
                    "seed_values": values,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "ci95": bootstrap_mean_ci(values, seed=7100 + len(metric)),
                }
            final[condition][side] = {"seed_count": len(rows), "metrics": metrics}
    return {"metadata": metadata, "final": final, "records": records}


def plot_metrics(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("frequency_probe_macro_accuracy", "Frequency probe macro accuracy", "frequency_probe.png"),
        ("shape_probe_macro_accuracy", "Token-shape probe macro accuracy", "shape_probe.png"),
        ("frequency_linear_probe_macro_accuracy", "Frozen frequency linear-probe macro accuracy", "frequency_linear_probe.png"),
        ("shape_linear_probe_macro_accuracy", "Frozen token-shape linear-probe macro accuracy", "shape_linear_probe.png"),
        ("frequency_neighbor_agreement", "Frequency-bucket neighbor agreement", "neighbor_agreement.png"),
        ("neighbor_cosine", "Mean nearest-neighbor cosine", "neighbor_cosine.png"),
    ]
    conditions = result["metadata"]["conditions"]
    for metric, title, filename in metrics:
        plt.figure(figsize=(10, 5))
        for side, style in (("input", "o-"), ("output", "s--")):
            values = [
                result["final"].get(condition, {}).get(side, {}).get("metrics", {}).get(metric, {}).get("mean")
                for condition in conditions
            ]
            if any(value is not None for value in values):
                plt.plot(range(len(conditions)), values, style, label=side)
        plt.xticks(range(len(conditions)), conditions, rotation=25, ha="right")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results/primary/embedding_eval")
    args = parser.parse_args()
    records, metadata = load_records([Path(value) for value in args.studies])
    result = summarize(records, metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "embedding_eval_aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_metrics(result, output_dir / "figures")
    print(json.dumps(result["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
