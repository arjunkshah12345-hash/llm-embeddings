"""Combine separately validated study aggregates for robustness reporting."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import bootstrap_mean_ci


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_study_args(values: list[str]) -> dict[str, Path]:
    studies = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"study must use NAME=PATH syntax: {value}")
        name, path = value.split("=", 1)
        if not name or not path:
            raise SystemExit(f"invalid study specification: {value}")
        if name in studies:
            raise SystemExit(f"duplicate study name: {name}")
        studies[name] = Path(path)
    return studies


def aggregate(studies: dict[str, dict]) -> dict:
    comparison = []
    paired_comparisons = {}
    for name, result in studies.items():
        for condition, value in result.get("by_condition", {}).items():
            comparison.append(
                {
                    "study": name,
                    "condition": condition,
                    "seeds": value.get("run_count"),
                    "total_parameters": value.get("total_parameters"),
                    "additional_parameters_vs_tied": value.get("additional_parameters_vs_tied"),
                    "mean_final_val_loss": value.get("mean_final_val_loss"),
                    "ci95_final_val_loss": value.get("ci95_final_val_loss"),
                    "mean_best_val_loss": value.get("mean_best_val_loss"),
                }
            )
        runs = result.get("runs", [])
        by_key = {(int(row["seed"]), row["embedding_type"]): row for row in runs}
        variants = sorted({row["embedding_type"] for row in runs if row["embedding_type"] != "tied"})
        study_pairs = {}
        for variant in variants:
            seeds = sorted(
                seed for seed, condition in by_key
                if condition == "tied" and (seed, variant) in by_key
            )
            for metric in ("final_val_loss", "best_val_loss", "final_val_perplexity", "best_val_perplexity"):
                deltas = [
                    float(by_key[(seed, variant)][metric]) - float(by_key[(seed, "tied")][metric])
                    for seed in seeds
                    if by_key[(seed, variant)].get(metric) is not None
                    and by_key[(seed, "tied")].get(metric) is not None
                ]
                study_pairs[f"{variant}_minus_tied_{metric}"] = {
                    "variant": variant,
                    "metric": metric,
                    "seeds": seeds,
                    "deltas": deltas,
                    "mean_delta": statistics.mean(deltas) if deltas else None,
                    "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0 if deltas else None,
                    "ci95": bootstrap_mean_ci(deltas, seed=8100 + len(name) + len(metric)) if deltas else None,
                }
        paired_comparisons[name] = study_pairs
    return {"studies": studies, "comparison": comparison, "paired_comparisons": paired_comparisons}


def plot(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    studies = list(result["studies"])
    conditions = sorted({row["condition"] for row in result["comparison"]})
    plt.figure(figsize=(10, 5))
    for condition in conditions:
        values = []
        labels = []
        for study in studies:
            row = next((item for item in result["comparison"] if item["study"] == study and item["condition"] == condition), None)
            if row is not None:
                values.append(row["mean_final_val_loss"])
                labels.append(study)
        if values:
            plt.plot(labels, values, "o-", label=condition)
    plt.ylabel("Mean final validation loss")
    plt.title("Validation loss across robustness studies")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output / "robustness_final_loss.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True, metavar="NAME=AGGREGATE_JSON")
    parser.add_argument("--output-dir", default="results/secondary")
    args = parser.parse_args()
    paths = parse_study_args(args.studies)
    studies = {name: read_json(path) for name, path in paths.items()}
    result = aggregate(studies)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "secondary_aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot(result, output_dir / "figures")
    print(json.dumps(result["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
