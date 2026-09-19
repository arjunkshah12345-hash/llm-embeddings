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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="runs/study-001")
    parser.add_argument("--dataset", choices=["wikitext2", "tiny_shakespeare"], default="wikitext2")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1337])
    parser.add_argument("--embedding_types", nargs="+", choices=["tied", "untied", "partial"], default=["tied", "untied", "partial"])
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
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true", help="allow existing run directories to be replaced")
    parser.add_argument("--skip_analysis", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common_keys = (
        "dataset", "data_dir", "device", "steps", "batch_size", "grad_accum_steps",
        "block_size", "n_layer", "n_head", "n_embd", "dropout", "adapter_rank",
        "adapter_alpha", "learning_rate", "min_learning_rate", "warmup_steps",
        "weight_decay", "grad_clip", "eval_interval", "eval_batches", "log_interval", "save_interval",
    )
    study_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "script": "train.py",
        "common": {key: getattr(args, key) for key in common_keys},
        "seeds": args.seeds,
        "embedding_types": args.embedding_types,
        "runs": [],
    }
    manifest_path = output_dir / "study_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Study manifest already exists: {manifest_path}; use --overwrite to start a new study")
    write_json(manifest_path, study_manifest)

    repo_root = Path(__file__).resolve().parent
    for seed in args.seeds:
        for embedding_type in args.embedding_types:
            run_name = f"seed{seed}_{embedding_type}"
            run_dir = output_dir / run_name
            if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite:
                raise SystemExit(f"Run directory already contains files: {run_dir}; use --overwrite to replace it")
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
            print(f"\n=== {run_name} ===", flush=True)
            subprocess.run(command, cwd=repo_root, check=True)
            study_manifest["runs"].append({"run_name": run_name, "seed": seed, "embedding_type": embedding_type, "command": command})
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
