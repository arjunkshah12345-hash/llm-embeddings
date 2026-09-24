"""Generate and submit reproducible private Kaggle experiment kernels."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OWNER = "aks1321"

PRIMARY = {
    "runner": "sweep",
    "experiment_id": "primary_10k",
    "dataset": "wikitext2",
    "steps": 10_000,
    "batch_size": 2,
    "block_size": 256,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "adapter_rank": 8,
    "adapter_alpha": 8,
    "eval_interval": 200,
    "eval_batches": 20,
    "log_interval": 50,
    "save_interval": 1_000,
    "seeds": [1337, 2027, 31415],
    "embedding_types": ["tied", "untied", "partial", "capacity_control", "partial_input", "partial_output"],
}

VALIDATION = {
    "runner": "sweep",
    "experiment_id": "validation_fixture",
    "dataset": "fixture",
    "steps": 3,
    "batch_size": 1,
    "block_size": 8,
    "n_layer": 1,
    "n_head": 1,
    "n_embd": 16,
    "adapter_rank": 2,
    "adapter_alpha": 2,
    "eval_interval": 1,
    "eval_batches": 1,
    "log_interval": 1,
    "save_interval": 3,
    "seeds": [1337],
    "embedding_types": ["tied", "untied", "partial", "capacity_control", "partial_input", "partial_output"],
}

RANK_SWEEP = {
    "runner": "adapter_sweep",
    "experiment_id": "rank_sweep_10k",
    "dataset": "wikitext2",
    "steps": 10_000,
    "batch_size": 2,
    "block_size": 256,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "ranks": [1, 2, 4, 8, 16, 32],
    "alphas": [8.0],
    "eval_interval": 200,
    "eval_batches": 20,
    "log_interval": 50,
    "save_interval": 1_000,
    "seeds": [1337],
}

LONG = {
    "runner": "sweep",
    "experiment_id": "long_50k",
    "dataset": "wikitext2",
    "steps": 50_000,
    "batch_size": 2,
    "block_size": 256,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "adapter_rank": 8,
    "adapter_alpha": 8,
    "eval_interval": 500,
    "eval_batches": 20,
    "log_interval": 100,
    "save_interval": 5_000,
    "seeds": [1337, 2027, 31415],
    "embedding_types": ["tied", "untied", "partial", "capacity_control"],
}

SMALL_SCALE = {
    "runner": "sweep",
    "experiment_id": "small_scale_10k",
    "dataset": "wikitext2",
    "steps": 10_000,
    "batch_size": 2,
    "block_size": 256,
    "n_layer": 4,
    "n_head": 4,
    "n_embd": 256,
    "adapter_rank": 8,
    "adapter_alpha": 8,
    "eval_interval": 200,
    "eval_batches": 20,
    "log_interval": 50,
    "save_interval": 1_000,
    "seeds": [1337, 2027, 31415],
    "embedding_types": ["tied", "untied", "partial", "capacity_control"],
}

SECOND_DATASET = {
    "runner": "sweep",
    "experiment_id": "tiny_shakespeare_10k",
    "dataset": "tiny_shakespeare",
    "steps": 10_000,
    "batch_size": 2,
    "block_size": 256,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "adapter_rank": 8,
    "adapter_alpha": 8,
    "eval_interval": 200,
    "eval_batches": 20,
    "log_interval": 50,
    "save_interval": 1_000,
    "seeds": [1337, 2027, 31415],
    "embedding_types": ["tied", "untied", "partial", "capacity_control"],
}


RUN_TEMPLATE = r'''"""Generated Kaggle kernel for the llm-embeddings study."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


COMMIT = "__COMMIT__"
CONFIG = json.loads(r"""__CONFIG_JSON__""")
SEED = __SEED__
OWNER = "__OWNER__"
KERNEL_SLUG = "__KERNEL_SLUG__"
SOURCE_URL = "https://github.com/arjunkshah12345-hash/llm-embeddings.git"
WORK = Path("/kaggle/working")
SOURCE = WORK / "llm-embeddings-source"
STUDY = WORK / f"{CONFIG['experiment_id']}_seed{SEED}"
DATA = WORK / "llm-embeddings-data"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def capture(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False).stdout
    except OSError as exc:
        return f"unavailable: {exc}"


def main() -> None:
    started = time.time()
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    run(["git", "clone", SOURCE_URL, str(SOURCE)])
    run(["git", "checkout", COMMIT], cwd=SOURCE)
    actual_commit = capture(["git", "rev-parse", "HEAD"], cwd=SOURCE).strip()
    if actual_commit != COMMIT:
        raise RuntimeError(f"source commit mismatch: expected {COMMIT}, got {actual_commit}")

    dependency_file = "requirements-lock.txt" if (SOURCE / "requirements-lock.txt").exists() else "requirements.txt"
    run([sys.executable, "-m", "pip", "install", "-r", dependency_file, "--quiet"], cwd=SOURCE)
    if CONFIG["runner"] == "sweep":
        run([
            sys.executable, "sweep.py",
            "--output_dir", str(STUDY),
            "--dataset", CONFIG["dataset"],
            "--data_dir", str(DATA),
            "--device", "cuda",
            "--seeds", str(SEED),
            "--embedding_types", *CONFIG["embedding_types"],
            "--steps", str(CONFIG["steps"]),
            "--batch_size", str(CONFIG["batch_size"]),
            "--block_size", str(CONFIG["block_size"]),
            "--n_layer", str(CONFIG["n_layer"]),
            "--n_head", str(CONFIG["n_head"]),
            "--n_embd", str(CONFIG["n_embd"]),
            "--adapter_rank", str(CONFIG["adapter_rank"]),
            "--adapter_alpha", str(CONFIG["adapter_alpha"]),
            "--eval_interval", str(CONFIG["eval_interval"]),
            "--eval_batches", str(CONFIG["eval_batches"]),
            "--log_interval", str(CONFIG["log_interval"]),
            "--save_interval", str(CONFIG["save_interval"]),
            "--no-save_optimizer",
        ], cwd=SOURCE)

        run([
            sys.executable, "mechanism_eval.py", "--runs_dir", str(STUDY),
            "--checkpoint", "last.pt", "--device", "cuda",
            "--output", str(STUDY / "mechanism_metrics.json"),
        ], cwd=SOURCE)
        pair_file = SOURCE / "eval" / "semantic_pairs.jsonl"
        for run_dir in sorted(path for path in STUDY.iterdir() if path.is_dir() and (path / "config.json").exists()):
            run([sys.executable, "embedding_eval.py", "--run_dir", str(run_dir), "--checkpoint", "best.pt", "--side", "both", "--pairs", str(pair_file)], cwd=SOURCE)
    elif CONFIG["runner"] == "adapter_sweep":
        run([
            sys.executable, "adapter_sweep.py",
            "--output_dir", str(STUDY),
            "--dataset", CONFIG["dataset"],
            "--data_dir", str(DATA),
            "--device", "cuda",
            "--seeds", str(SEED),
            "--ranks", *[str(rank) for rank in CONFIG["ranks"]],
            "--alphas", *[str(alpha) for alpha in CONFIG["alphas"]],
            "--steps", str(CONFIG["steps"]),
            "--batch_size", str(CONFIG["batch_size"]),
            "--block_size", str(CONFIG["block_size"]),
            "--n_layer", str(CONFIG["n_layer"]),
            "--n_head", str(CONFIG["n_head"]),
            "--n_embd", str(CONFIG["n_embd"]),
            "--eval_interval", str(CONFIG["eval_interval"]),
            "--eval_batches", str(CONFIG["eval_batches"]),
            "--log_interval", str(CONFIG["log_interval"]),
            "--save_interval", str(CONFIG["save_interval"]),
            "--no-save_optimizer",
        ], cwd=SOURCE)
    else:
        raise RuntimeError(f"unknown runner: {CONFIG['runner']}")

    hardware = {
        "python": sys.version,
        "platform": platform.platform(),
        "pip_freeze": capture([sys.executable, "-m", "pip", "freeze"]),
        "nvidia_smi": capture(["nvidia-smi"]),
    }
    (STUDY / "kaggle_hardware.json").write_text(json.dumps(hardware, indent=2) + "\n")
    manifest = {
        "experiment_id": CONFIG["experiment_id"],
        "kernel": f"{OWNER}/{KERNEL_SLUG}",
        "seed": SEED,
        "git_commit": actual_commit,
        "source_url": SOURCE_URL,
        "config": CONFIG,
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "study_dir": STUDY.name,
        "checkpoint_policy": "used inside Kaggle for diagnostics, removed from collected bundle",
    }
    (STUDY / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in STUDY.rglob("*.pt"):
        path.unlink()
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
'''


def metadata(kernel_id: str, title: str) -> dict:
    return {
        "id": kernel_id,
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["primary", "validation", "rank", "long", "small_scale", "second_dataset"],
        default="primary",
    )
    parser.add_argument("--commit", default="", help="frozen source commit; defaults to the current checkout HEAD")
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="explicitly replace an existing Kaggle kernel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = {
        "primary": PRIMARY,
        "validation": VALIDATION,
        "rank": RANK_SWEEP,
        "long": LONG,
        "small_scale": SMALL_SCALE,
        "second_dataset": SECOND_DATASET,
    }
    config = dict(profiles[args.profile])
    commit = args.commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    seeds = args.seeds or config["seeds"]
    generated_root = ROOT / "kaggle" / "generated"
    generated_root.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        slug = f"llm-embeddings-{config['experiment_id'].replace('_', '-')}-seed{seed}"
        kernel_id = f"{args.owner}/{slug}"
        kernel_dir = generated_root / slug
        if kernel_dir.exists():
            shutil.rmtree(kernel_dir)
        kernel_dir.mkdir()
        rendered = RUN_TEMPLATE.replace("__COMMIT__", commit)
        rendered = rendered.replace("__CONFIG_JSON__", json.dumps(config, sort_keys=True))
        rendered = rendered.replace("__SEED__", str(seed))
        rendered = rendered.replace("__OWNER__", args.owner)
        rendered = rendered.replace("__KERNEL_SLUG__", slug)
        (kernel_dir / "run.py").write_text(rendered)
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata(kernel_id, slug), indent=2) + "\n"
        )
        print(kernel_id)
        if not args.dry_run:
            if not args.overwrite:
                status = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
                if status.returncode == 0:
                    raise SystemExit(f"Kaggle kernel already exists: {kernel_id}; use a new slug or --overwrite")
            subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], check=True)


if __name__ == "__main__":
    main()
