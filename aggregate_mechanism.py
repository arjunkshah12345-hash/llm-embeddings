"""Aggregate mechanism metrics from validated per-seed cloud studies."""

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


METRICS = (
    "input_grad_norm",
    "output_grad_norm",
    "output_to_input_grad_ratio",
    "input_output_grad_cosine",
    "shared_input_grad_norm",
    "shared_output_grad_norm",
    "shared_output_to_input_grad_ratio",
    "embedding_cumulative_update_norm",
    "input_correction_norm",
    "output_correction_norm",
    "input_correction_relative_norm",
    "output_correction_relative_norm",
    "input_correction_effective_rank",
    "output_correction_effective_rank",
    "input_correction_top_singular_value",
    "output_correction_top_singular_value",
    "input_output_correction_cosine",
    "input_output_left_subspace_overlap",
    "input_output_right_subspace_overlap",
    "input_correction_shared_cosine",
    "output_correction_shared_cosine",
)

TOKEN_METRICS = (
    "input_token_grad_whitespace_mean",
    "output_token_grad_whitespace_mean",
    "input_token_grad_punctuation_mean",
    "output_token_grad_punctuation_mean",
    "input_token_grad_common_mean",
    "output_token_grad_common_mean",
    "input_token_grad_rare_mean",
    "output_token_grad_rare_mean",
    "input_token_freq_grad_q0_mean",
    "output_token_freq_grad_q0_mean",
    "input_token_freq_grad_q1_mean",
    "output_token_freq_grad_q1_mean",
    "input_token_freq_grad_q2_mean",
    "output_token_freq_grad_q2_mean",
    "input_token_freq_grad_q3_mean",
    "output_token_freq_grad_q3_mean",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_records(study_dirs: list[Path]) -> tuple[list[dict], dict, list[dict]]:
    records: list[dict] = []
    fixed_final_records: list[dict] = []
    conditions: list[str] | None = None
    seeds: list[int] = []
    commits: set[str] = set()
    for study_dir in study_dirs:
        validation = read_json(study_dir / "study_validation.json")
        if not validation.get("passed"):
            raise SystemExit(f"fairness validation failed: {study_dir}")
        manifest = read_json(study_dir / "study_manifest.json")
        study_conditions = list(manifest.get("embedding_types", []))
        if conditions is None:
            conditions = study_conditions
        elif study_conditions != conditions:
            raise SystemExit(f"condition list differs: {study_dir}")
        study_seeds = [int(seed) for seed in manifest.get("seeds", [])]
        if len(study_seeds) != 1:
            raise SystemExit(f"each study artifact must contain one seed: {study_dir}")
        seed = study_seeds[0]
        seeds.append(seed)
        artifact = read_json(study_dir / "artifact_manifest.json")
        commits.add(str(artifact.get("git_commit")))
        for run_dir in sorted(path for path in study_dir.iterdir() if (path / "config.json").exists()):
            config = read_json(run_dir / "config.json")
            condition = config["train"]["embedding_type"]
            rows = [row for row in read_jsonl(run_dir / "metrics.jsonl") if row.get("split") == "train"]
            for row in rows:
                values = {key: row[key] for key in METRICS + TOKEN_METRICS if key in row}
                records.append({"seed": seed, "condition": condition, "step": int(row["step"]), **values})
        mechanism_path = study_dir / "mechanism_metrics.json"
        if mechanism_path.exists():
            mechanism = read_json(mechanism_path)
            for run in mechanism.get("runs", []):
                gradient = dict(run.get("gradient_metrics", {}))
                adapter = dict(run.get("adapter_metrics", {}))
                checkpoint_validation = dict(run.get("checkpoint_validation", {}))
                values = {key: gradient.get(key, adapter.get(key)) for key in METRICS + TOKEN_METRICS}
                if checkpoint_validation.get("embedding_cumulative_update_norm") is not None:
                    values["embedding_cumulative_update_norm"] = checkpoint_validation[
                        "embedding_cumulative_update_norm"
                    ]
                fixed_final_records.append(
                    {
                        "seed": seed,
                        "condition": run["embedding_type"],
                        "step": int(run.get("checkpoint_step", -1)),
                        **{key: value for key, value in values.items() if value is not None},
                    }
                )
    if len(seeds) != len(set(seeds)):
        raise SystemExit("duplicate mechanism seed artifacts")
    if len(commits) != 1:
        raise SystemExit("mechanism artifact commits differ")
    return records, {
        "conditions": conditions or [],
        "seeds": sorted(seeds),
        "git_commit": next(iter(commits)),
        "final_metric_source": "mechanism_metrics.json fixed checkpoint batch when available",
    }, fixed_final_records


def summarize(records: list[dict], metadata: dict, fixed_final_records: list[dict] | None = None) -> dict:
    final_rows: dict[tuple[str, int], dict] = {}
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["condition"], row["step"])].append(row)
        key = (row["condition"], row["seed"])
        if key not in final_rows or row["step"] > final_rows[key]["step"]:
            final_rows[key] = row

    if fixed_final_records:
        for row in fixed_final_records:
            key = (row["condition"], row["seed"])
            if key not in final_rows or row["step"] >= final_rows[key]["step"]:
                final_rows[key] = row

    final: dict[str, dict] = {}
    for condition in metadata["conditions"]:
        rows = [row for (kind, _), row in final_rows.items() if kind == condition]
        metric_summary = {}
        for metric in METRICS + TOKEN_METRICS:
            values = [float(row[metric]) for row in rows if metric in row]
            if not values:
                continue
            metric_summary[metric] = {
                "seed_values": values,
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "ci95": bootstrap_mean_ci(values, seed=7000 + len(metric)),
            }
        final[condition] = {"seed_count": len(rows), "metrics": metric_summary}

    trajectory: dict[str, dict[str, dict[str, float]]] = {}
    for (condition, step), rows in sorted(grouped.items()):
        condition_out = trajectory.setdefault(condition, {})
        step_out = condition_out.setdefault(str(step), {})
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if metric in row]
            if values:
                step_out[metric] = statistics.mean(values)
    return {"metadata": metadata, "final": final, "trajectory": trajectory}


def plot_metric(result: dict, output_dir: Path, metric: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 5))
    plotted = False
    for condition in result["metadata"]["conditions"]:
        points = []
        for step, values in result["trajectory"].get(condition, {}).items():
            if metric in values:
                points.append((int(step), values[metric]))
        if points:
            plotted = True
            points.sort()
            plt.plot([x for x, _ in points], [y for _, y in points], label=condition)
    plt.xlabel("Optimizer step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}.png", dpi=180)
    plt.close()


def plot_final_token_roles(result: dict, output_dir: Path) -> None:
    metrics = [
        ("input_token_grad_common_mean", "Input common"),
        ("output_token_grad_common_mean", "Output common"),
        ("input_token_grad_rare_mean", "Input rare"),
        ("output_token_grad_rare_mean", "Output rare"),
        ("input_token_grad_punctuation_mean", "Input punctuation"),
        ("output_token_grad_punctuation_mean", "Output punctuation"),
        ("input_token_grad_whitespace_mean", "Input whitespace"),
        ("output_token_grad_whitespace_mean", "Output whitespace"),
    ]
    conditions = result["metadata"]["conditions"]
    x = list(range(len(conditions)))
    width = 0.1
    plt.figure(figsize=(12, 5))
    for index, (metric, label) in enumerate(metrics):
        values = [result["final"].get(c, {}).get("metrics", {}).get(metric, {}).get("mean") for c in conditions]
        if any(value is not None for value in values):
            plt.bar([value + (index - len(metrics) / 2) * width for value in x], values, width=width, label=label)
    plt.xticks(x, conditions, rotation=25, ha="right")
    plt.ylabel("Mean row gradient norm")
    plt.title("Final token-class input/output gradient measurements")
    plt.grid(alpha=0.25, axis="y")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "token_role_gradients.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--output_dir", default="results/primary/mechanism")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, metadata, fixed_final_records = load_records([Path(value) for value in args.studies])
    result = summarize(records, metadata, fixed_final_records=fixed_final_records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mechanism_aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    plot_metric(result, output_dir, "output_to_input_grad_ratio", "Output/input embedding gradient ratio", "Output / input gradient norm")
    plot_metric(result, output_dir, "input_output_grad_cosine", "Input/output effective gradient cosine", "Cosine similarity")
    plot_metric(result, output_dir, "input_correction_relative_norm", "Relative input correction norm", "Correction norm / shared norm")
    plot_metric(result, output_dir, "output_correction_relative_norm", "Relative output correction norm", "Correction norm / shared norm")
    plot_metric(result, output_dir, "input_output_correction_cosine", "Input/output correction alignment", "Effective correction cosine")
    plot_final_token_roles(result, output_dir)
    print(json.dumps(result["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
