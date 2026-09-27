"""Run the frozen Study 3 training matrix on a Google Colab GPU.

This runner performs training only. It deliberately leaves benchmark
evaluation out of the Colab path because the current scaled benchmark outputs
are undertrained and the primary missing evidence is matched validation-loss
replication. Point ``--output-root`` at Google Drive so a runtime disconnect
does not discard checkpoints.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


CONDITIONS = ("tied", "partial", "untied", "capacity_control")
SEEDS = (1337, 2027, 31415)
STEPS = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def run(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    if not args.conditions:
        raise SystemExit("at least one condition is required")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Study 3 training requires a Colab GPU runtime; CUDA is unavailable")

    repo = Path(__file__).resolve().parents[1]
    root = args.output_root.expanduser().resolve()
    data_dir = root / "data"
    training_root = root / f"seed{args.seed}" / "training"
    training_root.mkdir(parents=True, exist_ok=True)

    runner_manifest = {
        "kind": "scale3_colab_training",
        "protocol": "EXPERIMENT_PROTOCOL_SCALE.md",
        "seed": args.seed,
        "conditions": args.conditions,
        "steps": STEPS,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "started_at_unix": time.time(),
    }
    (root / f"seed{args.seed}_runner_manifest.json").write_text(
        json.dumps(runner_manifest, indent=2, sort_keys=True) + "\n"
    )

    common = [
        sys.executable,
        "train.py",
        "--dataset",
        "fineweb_edu",
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(training_root),
        "--device",
        "cuda",
        "--seed",
        str(args.seed),
        "--steps",
        str(STEPS),
        "--batch_size",
        "1",
        "--grad_accum_steps",
        "2",
        "--block_size",
        "512",
        "--n_layer",
        "12",
        "--n_head",
        "12",
        "--n_embd",
        "768",
        "--adapter_rank",
        "8",
        "--adapter_alpha",
        "8",
        "--capacity_control_width",
        "532",
        "--learning_rate",
        "3e-4",
        "--min_learning_rate",
        "3e-5",
        "--warmup_steps",
        "500",
        "--weight_decay",
        "0.1",
        "--grad_clip",
        "1.0",
        "--eval_interval",
        "500",
        "--eval_batches",
        "32",
        "--log_interval",
        "500",
        "--save_interval",
        "2500",
        "--save_optimizer",
    ]

    for condition in args.conditions:
        run_dir = training_root / condition
        resume = run_dir / "optimizer_last.pt"
        if run_dir.exists() and not args.resume_existing:
            raise SystemExit(
                f"{run_dir} already exists; use --resume-existing to continue from its optimizer checkpoint"
            )
        command = [*common, "--embedding_type", condition, "--run_name", condition]
        if args.resume_existing and resume.exists():
            command.extend(["--resume", str(resume)])
        run(command, repo)

    runner_manifest["finished_at_unix"] = time.time()
    (root / f"seed{args.seed}_runner_manifest.json").write_text(
        json.dumps(runner_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"completed Study 3 Colab seed={args.seed} output={root}", flush=True)


if __name__ == "__main__":
    main()
