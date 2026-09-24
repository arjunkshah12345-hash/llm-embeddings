"""Run a fair matrix of embedding experiments with one shared configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_study import validate_study


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_is_complete(run_dir: Path, steps: int) -> bool:
    validation = [row for row in read_jsonl(run_dir / "metrics.jsonl") if row.get("split") == "val"]
    return bool(validation) and max(int(row.get("step", -1)) for row in validation) >= steps - 1


def upsert_run(manifest: dict, entry: dict) -> None:
    key = (int(entry["seed"]), entry["embedding_type"])
    runs = [
        existing
        for existing in manifest.get("runs", [])
        if (int(existing["seed"]), existing["embedding_type"]) != key
    ]
    runs.append(entry)
    manifest["runs"] = runs


def validate_resume_common(previous_common: dict, requested_common: dict) -> None:
    """Require an exact study configuration when resuming optimizer state.

    In particular, the step horizon controls the cosine learning-rate schedule.
    Changing it would make a resumed run incomparable with a fresh run at the
    requested horizon, so longer studies must use a new output directory.
    """
    for key, value in requested_common.items():
        if key == "steps":
            if int(previous_common.get(key, -1)) != int(value):
                raise SystemExit(
                    "Cannot resume study with a different step horizon; "
                    "launch a fresh study directory for a new training schedule"
                )
        elif previous_common.get(key) != value:
            raise SystemExit(f"Cannot resume study: manifest common setting {key} differs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/study-001")
    parser.add_argument("--dataset", choices=["wikitext2", "tiny_shakespeare", "fixture"], default="wikitext2")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337])
    parser.add_argument(
        "--embedding_types",
        nargs="+",
        choices=["tied", "untied", "partial", "partial_input", "partial_output", "capacity_control"],
        default=["tied", "untied", "partial"],
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter_rank", type=int, default=8)
    parser.add_argument("--adapter_alpha", type=float, default=8.0)
    parser.add_argument("--capacity_control_width", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--path_ablation", choices=["none", "stop_input", "stop_output"], default="none")
    parser.add_argument("--ablation_start", type=int, default=0)
    parser.add_argument("--ablation_end", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="allow existing run directories to be replaced")
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        help="resume incomplete runs from optimizer_last.pt and skip completed runs in an existing study",
    )
    parser.add_argument(
        "--save_optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write optimizer checkpoints for exact resume (default: true)",
    )
    parser.add_argument("--skip_analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume_existing:
        raise SystemExit("--overwrite and --resume_existing are mutually exclusive")
    if args.ablation_start < 0 or args.ablation_end < 0:
        raise SystemExit("ablation_start and ablation_end must be non-negative")
    if args.ablation_end and args.ablation_end <= args.ablation_start:
        raise SystemExit("ablation_end must be greater than ablation_start when non-zero")
    if args.path_ablation != "none" and args.ablation_start >= args.steps:
        raise SystemExit("ablation_start must be smaller than steps when path_ablation is enabled")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common_keys = (
        "dataset", "data_dir", "device", "steps", "batch_size", "grad_accum_steps",
        "block_size", "n_layer", "n_head", "n_embd", "dropout", "adapter_rank",
        "adapter_alpha", "capacity_control_width", "learning_rate", "min_learning_rate", "warmup_steps",
        "weight_decay", "grad_clip", "eval_interval", "eval_batches", "log_interval", "save_interval",
        "path_ablation", "ablation_start", "ablation_end",
    )
    common = {key: getattr(args, key) for key in common_keys}
    new_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "script": "train.py",
        "common": common,
        "seeds": args.seeds,
        "embedding_types": args.embedding_types,
        "save_optimizer": bool(args.save_optimizer),
        "runs": [],
    }
    manifest_path = output_dir / "study_manifest.json"
    if manifest_path.exists():
        if not args.resume_existing and not args.overwrite:
            raise SystemExit(f"Study manifest already exists: {manifest_path}; use --overwrite or --resume_existing")
        if args.resume_existing:
            study_manifest = json.loads(manifest_path.read_text())
            if study_manifest.get("seeds") != new_manifest["seeds"]:
                raise SystemExit("Cannot resume study: manifest seeds differ from requested configuration")
            if study_manifest.get("embedding_types") != new_manifest["embedding_types"]:
                raise SystemExit("Cannot resume study: manifest embedding_types differ from requested configuration")
            if bool(study_manifest.get("save_optimizer", True)) != bool(new_manifest["save_optimizer"]):
                raise SystemExit("Cannot resume study with a different optimizer-checkpoint policy")
            previous_common = study_manifest.get("common", {})
            validate_resume_common(previous_common, common)
            study_manifest["common"] = common
        else:
            study_manifest = new_manifest
    else:
        if args.resume_existing:
            raise SystemExit(f"Cannot resume missing study manifest: {manifest_path}")
        study_manifest = new_manifest
    write_json(manifest_path, study_manifest)

    repo_root = Path(__file__).resolve().parent
    for seed in args.seeds:
        for embedding_type in args.embedding_types:
            run_name = f"seed{seed}_{embedding_type}"
            run_dir = output_dir / run_name
            resume_path = None
            if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
                if run_is_complete(run_dir, args.steps):
                    print(f"skip completed run={run_name}", flush=True)
                    upsert_run(
                        study_manifest,
                        {"run_name": run_name, "seed": seed, "embedding_type": embedding_type, "command": ["preserved"]},
                    )
                    write_json(manifest_path, study_manifest)
                    continue
                if not args.resume_existing:
                    raise SystemExit(f"Run directory already contains files: {run_dir}; use --overwrite or --resume_existing")
                resume_path = run_dir / "optimizer_last.pt"
                if not resume_path.exists():
                    raise SystemExit(f"Cannot resume incomplete run without {resume_path}")
            command = [
                sys.executable, "train.py", "--embedding_type", embedding_type,
                "--dataset", args.dataset, "--data_dir", args.data_dir,
                "--output_dir", str(output_dir), "--run_name", run_name,
                "--device", args.device, "--seed", str(seed),
            ]
            for key in common_keys:
                if key in {"dataset", "data_dir", "device"}:
                    continue
                command.extend([f"--{key}", str(getattr(args, key))])
            if resume_path is not None:
                command.extend(["--resume", str(resume_path)])
            if not args.save_optimizer:
                command.append("--no-save_optimizer")
            print(f"\n=== {run_name} ===", flush=True)
            subprocess.run(command, cwd=repo_root, check=True)
            upsert_run(study_manifest, {"run_name": run_name, "seed": seed, "embedding_type": embedding_type, "command": command})
            write_json(manifest_path, study_manifest)

    validation = validate_study(output_dir)
    write_json(output_dir / "study_validation.json", validation)
    if not validation["passed"]:
        print(json.dumps(validation, indent=2), file=sys.stderr)
        raise SystemExit("Study fairness validation failed; refusing to analyze an invalid comparison")

    if not args.skip_analysis:
        analysis_dir = output_dir / "analysis"
        subprocess.run(
            [sys.executable, "analyze.py", "--runs_dir", str(output_dir), "--output_dir", str(analysis_dir)],
            cwd=repo_root,
            check=True,
        )
    print(f"Completed {len(study_manifest['runs'])} runs under {output_dir}")


if __name__ == "__main__":
    main()
