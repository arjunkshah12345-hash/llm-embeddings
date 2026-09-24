"""Aggregate validated partial-embedding rank-sweep artifacts."""

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


def load_rank_rows(study_dirs: list[Path]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    seen_seeds: set[int] = set()
    commits: set[str] = set()
    for study_dir in study_dirs:
        manifest = read_json(study_dir / "artifact_manifest.json")
        seed = int(manifest["seed"])
        if seed in seen_seeds:
            raise SystemExit(f"duplicate rank seed artifact: {seed}")
        seen_seeds.add(seed)
        commits.add(str(manifest.get("git_commit")))
        summary = read_json(study_dir / "adapter_sweep_summary.json")
        for row in summary.get("conditions", []):
            rows.append({"seed": seed, **row})
    if len(commits) != 1:
        raise SystemExit("rank artifact commits differ")
    if not rows:
        raise SystemExit("rank artifacts contain no conditions")
    return rows, {"seeds": sorted(seen_seeds), "git_commit": next(iter(commits))}


def load_tied_rows(path: Path) -> dict[int, dict]:
    payload = read_json(path)
    return {
        int(row["seed"]): row
        for row in payload.get("runs", [])
        if row.get("embedding_type") == "tied"
    }


def aggregate(rows: list[dict], metadata: dict, tied_rows: dict[int, dict] | None = None) -> dict:
    ranks = sorted({int(row["adapter_rank"]) for row in rows})
    by_rank: dict[int, dict] = {}
    for rank in ranks:
        rank_rows = [row for row in rows if int(row["adapter_rank"]) == rank]
        final_values = [float(row["mean_final_val_loss"]) for row in rank_rows]
        best_values = [float(row["mean_best_val_loss"]) for row in rank_rows]
        entry = {
            "rank": rank,
            "alpha": float(rank_rows[0]["adapter_alpha"]),
            "seed_values": rank_rows,
            "run_count": len(rank_rows),
            "total_parameters": int(rank_rows[0]["total_parameters"]),
            "additional_parameters_vs_tied": int(rank_rows[0]["additional_parameters_vs_tied"]),
            "mean_final_val_loss": statistics.mean(final_values),
            "std_final_val_loss": statistics.stdev(final_values) if len(final_values) > 1 else 0.0,
            "ci95_final_val_loss": bootstrap_mean_ci(final_values, seed=6100 + rank),
            "mean_best_val_loss": statistics.mean(best_values),
            "std_best_val_loss": statistics.stdev(best_values) if len(best_values) > 1 else 0.0,
            "ci95_best_val_loss": bootstrap_mean_ci(best_values, seed=6200 + rank),
        }
        if tied_rows is not None:
            paired = []
            for row in rank_rows:
                seed = int(row["seed"])
                if seed not in tied_rows:
                    raise SystemExit(f"missing tied baseline for rank seed {seed}")
                paired.append(
                    {
                        "seed": seed,
                        "final_delta": float(row["mean_final_val_loss"]) - float(tied_rows[seed]["final_val_loss"]),
                        "best_delta": float(row["mean_best_val_loss"]) - float(tied_rows[seed]["best_val_loss"]),
                    }
                )
            for metric in ("final_delta", "best_delta"):
                values = [item[metric] for item in paired]
                entry[f"paired_{metric}"] = {
                    "seed_values": paired,
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "ci95": bootstrap_mean_ci(values, seed=6300 + rank + len(metric)),
                }
        by_rank[str(rank)] = entry
    return {"metadata": metadata, "by_rank": by_rank}


def plot_frontiers(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(result["by_rank"].values())
    plots = [
        ("additional_parameters_vs_tied", "Extra parameters vs tied", 1e6, "millions", "rank_vs_extra_parameters.png"),
        ("total_parameters", "Total parameters", 1e6, "millions", "rank_vs_total_parameters.png"),
    ]
    for key, xlabel, scale, suffix, filename in plots:
        plt.figure(figsize=(8, 5))
        plt.plot([row[key] / scale for row in rows], [row["mean_final_val_loss"] for row in rows], "o-")
        for row in rows:
            plt.annotate(f"r{row['rank']}", (row[key] / scale, row["mean_final_val_loss"]), xytext=(4, 4), textcoords="offset points")
        plt.xlabel(f"{xlabel} ({suffix})")
        plt.ylabel("Mean final validation loss")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    for key, xlabel, scale, filename in [
        ("mean_estimated_flops_total", "Estimated training FLOPs", 1e12, "rank_vs_flops.png"),
        ("mean_training_wall_time_seconds", "Training wall-clock seconds", 1.0, "rank_vs_wall_clock.png"),
    ]:
        if not all(key in row["seed_values"][0] for row in rows):
            continue
        plt.figure(figsize=(8, 5))
        plt.plot([row["seed_values"][0][key] / scale for row in rows], [row["mean_final_val_loss"] for row in rows], "o-")
        for row in rows:
            plt.annotate(f"r{row['rank']}", (row["seed_values"][0][key] / scale, row["mean_final_val_loss"]), xytext=(4, 4), textcoords="offset points")
        plt.xlabel(f"{xlabel}{' (trillions)' if scale == 1e12 else ''}")
        plt.ylabel("Mean final validation loss")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# Rank sweep aggregate",
        "",
        "Generated from validated per-seed adapter-sweep artifacts. Lower loss is better.",
        "",
        f"Seed coverage: {len(result['metadata']['seeds'])} seed(s) ({', '.join(str(seed) for seed in result['metadata']['seeds'])}). "
        + ("This is exploratory and does not support rank selection." if len(result["metadata"]["seeds"]) < 3 else ""),
        "",
        "| Rank | Seeds | Extra params | Mean final loss | 95% interval | Mean best loss |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["by_rank"].values():
        ci = row["ci95_final_val_loss"]
        lines.append(
            f"| {row['rank']} | {row['run_count']} | {row['additional_parameters_vs_tied']:,} | "
            f"{row['mean_final_val_loss']:.6f} | [{ci['low']:.6f}, {ci['high']:.6f}] | "
            f"{row['mean_best_val_loss']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True)
    parser.add_argument("--tied-results", default="")
    parser.add_argument("--output-dir", default="results/rank_sweep")
    args = parser.parse_args()
    rows, metadata = load_rank_rows([Path(value) for value in args.studies])
    tied_rows = load_tied_rows(Path(args.tied_results)) if args.tied_results else None
    result = aggregate(rows, metadata, tied_rows=tied_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rank_aggregate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_markdown(result, output_dir / "rank_aggregate.md")
    plot_frontiers(result, output_dir / "figures")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
