"""Export compact, checkpoint-free research artifacts for public release."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT_FILES = (
    "artifact_manifest.json",
    "study_manifest.json",
    "study_validation.json",
    "kaggle_hardware.json",
    "mechanism_metrics.json",
    "adapter_sweep_summary.json",
    "adapter_sweep_summary.md",
    "adapter_tradeoff.png",
    "rank_sweep_vs_estimated_flops.png",
    "rank_sweep_vs_extra_parameters.png",
    "rank_sweep_vs_total_parameters.png",
    "rank_sweep_vs_wall_clock.png",
)
RUN_FILES = (
    "config.json",
    "manifest.json",
    "parameter_counts.json",
    "metrics.jsonl",
    "embedding_eval.json",
)


def copy_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def export_study(name: str, source: Path, destination: Path) -> dict:
    if destination.exists():
        raise SystemExit(f"release destination already exists: {destination}")
    if not source.is_dir():
        raise SystemExit(f"study directory does not exist: {source}")
    destination.mkdir(parents=True)
    for filename in ROOT_FILES:
        copy_if_present(source / filename, destination / filename)

    run_names = []
    for run_dir in sorted(path for path in source.iterdir() if path.is_dir() and (path / "config.json").exists()):
        target = destination / "raw_runs" / run_dir.name
        for filename in RUN_FILES:
            copy_if_present(run_dir / filename, target / filename)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            batch_stream = manifest.get("batch_stream")
            if batch_stream:
                (target / "batch_stream_digest.json").write_text(
                    json.dumps(batch_stream, indent=2, sort_keys=True) + "\n"
                )
        run_names.append(run_dir.name)

    # Adapter sweeps expose one validated study directory per rank rather than
    # a flat run directory. Preserve those compact manifests for inspection.
    for substudy in sorted(path for path in source.iterdir() if path.is_dir() and (path / "study_manifest.json").exists()):
        target = destination / "substudies" / substudy.name
        for filename in ("study_manifest.json", "study_validation.json"):
            copy_if_present(substudy / filename, target / filename)

    artifact_manifest = {}
    manifest_path = source / "artifact_manifest.json"
    if manifest_path.exists():
        artifact_manifest = json.loads(manifest_path.read_text())
    return {
        "name": name,
        "experiment_id": artifact_manifest.get("experiment_id"),
        "git_commit": artifact_manifest.get("git_commit"),
        "seed": artifact_manifest.get("seed"),
        "run_names": run_names,
        "checkpoints_exported": False,
    }


def parse_studies(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    seen = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"study must use NAME=PATH syntax: {value}")
        name, path = value.split("=", 1)
        if not name or name in seen:
            raise SystemExit(f"invalid or duplicate study name: {name}")
        seen.add(name)
        parsed.append((name, Path(path)))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", default="results/release_artifacts")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise SystemExit(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    exports = [export_study(name, path, output / name) for name, path in parse_studies(args.studies)]
    (output / "release_manifest.json").write_text(
        json.dumps({"studies": exports, "checkpoints_exported": False}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(exports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
