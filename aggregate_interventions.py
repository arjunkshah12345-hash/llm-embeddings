"""Aggregate paired validation results from mechanism path-ablation studies."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from analyze import bootstrap_mean_ci


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_study(path: Path) -> dict:
    validation = read_json(path / "study_validation.json")
    if not validation.get("passed"):
        raise SystemExit(f"fairness validation failed: {path}")
    manifest = read_json(path / "study_manifest.json")
    conditions = list(manifest.get("embedding_types", []))
    if conditions != ["tied", "partial"]:
        raise SystemExit(f"expected tied/partial intervention study: {path}")
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    rows = []
    by_key: dict[tuple[int, str], dict] = {}
    for run_dir in sorted(p for p in path.iterdir() if p.is_dir() and (p / "metrics.jsonl").exists()):
        config = read_json(run_dir / "config.json")
        seed = int(config["train"]["seed"])
        condition = config["train"]["embedding_type"]
        values = [row for row in read_jsonl(run_dir / "metrics.jsonl") if row.get("split") == "val"]
        if not values:
            raise SystemExit(f"missing validation rows: {run_dir}")
        final = max(values, key=lambda row: int(row["step"]))
        best = min(values, key=lambda row: float(row["loss"]))
        row = {
            "seed": seed,
            "condition": condition,
            "final_step": int(final["step"]),
            "final_val_loss": float(final["loss"]),
            "best_step": int(best["step"]),
            "best_val_loss": float(best["loss"]),
        }
        rows.append(row)
        by_key[(seed, condition)] = row
    expected = {(seed, condition) for seed in seeds for condition in conditions}
    if set(by_key) != expected:
        raise SystemExit(f"incomplete intervention matrix: {path}")

    paired = {}
    for metric in ("final_val_loss", "best_val_loss"):
        deltas = [by_key[(seed, "partial")][metric] - by_key[(seed, "tied")][metric] for seed in sorted(seeds)]
        paired[f"partial_minus_tied_{metric}"] = {
            "metric": metric,
            "seeds": sorted(seeds),
            "deltas": deltas,
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "ci95": bootstrap_mean_ci(deltas, seed=9200 + len(metric)),
        }
    return {
        "metadata": {
            "git_commit": read_json(path / "artifact_manifest.json").get("git_commit"),
            "path_ablation": manifest["common"].get("path_ablation"),
            "ablation_start": manifest["common"].get("ablation_start"),
            "ablation_end": manifest["common"].get("ablation_end"),
            "seeds": sorted(seeds),
        },
        "runs": sorted(rows, key=lambda row: (row["seed"], row["condition"])),
        "paired_comparisons": paired,
    }


def combine_seed_studies(summaries: list[dict]) -> dict:
    if not summaries:
        raise SystemExit("no intervention studies provided")
    metadata = dict(summaries[0]["metadata"])
    seeds = []
    rows = []
    for summary in summaries:
        if summary["metadata"]["path_ablation"] != metadata["path_ablation"]:
            raise SystemExit("intervention path differs")
        seeds.extend(summary["metadata"]["seeds"])
        rows.extend(summary["runs"])
    if len(seeds) != len(set(seeds)):
        raise SystemExit("duplicate intervention seeds")
    metadata["seeds"] = sorted(seeds)
    by_key = {(row["seed"], row["condition"]): row for row in rows}
    expected = {(seed, condition) for seed in metadata["seeds"] for condition in ("tied", "partial")}
    if set(by_key) != expected:
        raise SystemExit("incomplete combined intervention matrix")
    paired = {}
    for metric in ("final_val_loss", "best_val_loss"):
        deltas = [by_key[(seed, "partial")][metric] - by_key[(seed, "tied")][metric] for seed in metadata["seeds"]]
        paired[f"partial_minus_tied_{metric}"] = {
            "metric": metric,
            "seeds": metadata["seeds"],
            "deltas": deltas,
            "mean_delta": statistics.mean(deltas),
            "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "ci95": bootstrap_mean_ci(deltas, seed=9200 + len(metric)),
        }
    return {"metadata": metadata, "runs": sorted(rows, key=lambda row: (row["seed"], row["condition"])), "paired_comparisons": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    studies: dict[str, list[dict]] = {}
    for value in args.studies:
        if "=" not in value:
            raise SystemExit(f"study must use NAME=PATH syntax: {value}")
        name, path = value.split("=", 1)
        if not name:
            raise SystemExit(f"invalid study name: {name}")
        studies.setdefault(name, []).append(summarize_study(Path(path)))
    payload = {"studies": {name: combine_seed_studies(summaries) for name, summaries in studies.items()}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
