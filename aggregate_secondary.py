"""Combine separately validated study aggregates for robustness reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    return {"studies": studies, "comparison": comparison}


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
