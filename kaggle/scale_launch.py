"""Submit one frozen Study 3 data probe or one condition/seed Kaggle job."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OWNER = "aks1321"
EXPERIMENT_ID = "scale3_fineweb_20m"
DATASET_SOURCE = "aks1321/llm-embeddings-scale3-data"
DATASET_SLUG = "llm-embeddings-scale3-data"
MODEL = {
    "block_size": 512,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "adapter_rank": 8,
    "adapter_alpha": 8,
    "capacity_control_width": 532,
}
TRAIN = {
    "dataset": "fineweb_edu",
    "steps": 20_000,
    "batch_size": 1,
    "grad_accum_steps": 2,
    "learning_rate": 3e-4,
    "min_learning_rate": 3e-5,
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "eval_interval": 500,
    "eval_batches": 32,
    "log_interval": 500,
    "save_interval": 2_500,
    "benchmark_batch_size": 1,
    "benchmark_suites": ["core", "extended"],
}
CONDITIONS = ("tied", "partial", "untied", "capacity_control")
SEEDS = (1337, 2027, 31415)


def kaggle_executable() -> str:
    candidates = [
        os.environ.get("KAGGLE_CLI"),
        shutil.which("kaggle"),
        "/opt/homebrew/bin/kaggle",
        "/usr/local/bin/kaggle",
        "/usr/bin/kaggle",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(Path(candidate))
    raise SystemExit("Kaggle CLI executable not found; install it or set KAGGLE_CLI")


def slug(condition: str, seed: int, suffix: str = "") -> str:
    # Kaggle normalizes kernel IDs by replacing underscores in titles with
    # hyphens.  Use the canonical URL slug here so status/collection can find
    # capacity_control jobs after submission.
    kernel_condition = condition.replace("_", "-")
    return f"llm-embeddings-scale3-{kernel_condition}-seed{seed}{suffix}"


RUN_TEMPLATE = r'''"""Generated Kaggle kernel for Study 3."""
from __future__ import annotations
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

COMMIT = "__COMMIT__"
MODE = "__MODE__"
CONDITION = "__CONDITION__"
SEED = __SEED__
OWNER = "__OWNER__"
KERNEL_SLUG = "__KERNEL_SLUG__"
CONFIG = json.loads(r"""__CONFIG_JSON__""")
SOURCE_URL = "https://github.com/arjunkshah12345-hash/llm-embeddings.git"
WORK = Path("/kaggle/working")
SOURCE = WORK / "llm-embeddings-source"
STUDY = WORK / f"{CONFIG['experiment_id']}-seed{SEED}-{CONDITION}"
DATA = WORK / "scale_data"
DATA_INPUT = Path("/kaggle/input/__DATASET_SLUG__")

def run(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)

def capture(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False).stdout
    except OSError as exc:
        return f"unavailable: {exc}"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> None:
    started = time.time()
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    run(["git", "clone", SOURCE_URL, str(SOURCE)])
    run(["git", "checkout", COMMIT], cwd=SOURCE)
    actual_commit = capture(["git", "rev-parse", "HEAD"], cwd=SOURCE).strip()
    if actual_commit != COMMIT:
        raise RuntimeError(f"source commit mismatch: expected {COMMIT}, got {actual_commit}")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements-scale-lock.txt", "--quiet"], cwd=SOURCE)
    if MODE in {"probe", "data_bundle"}:
        run([sys.executable, "scale_probe.py"], cwd=SOURCE)
        output_root = WORK / ("scale_probe" if MODE == "probe" else "scale_data_bundle")
        output_root.mkdir(parents=True, exist_ok=True)
        if MODE == "data_bundle":
            shutil.copytree(DATA, output_root / "fineweb_edu")
        manifest = {"kind": "scale_probe" if MODE == "probe" else "scale_data_bundle", "experiment_id": "scale3_probe" if MODE == "probe" else "scale3_data_bundle", "git_commit": actual_commit, "kernel": f"{OWNER}/{KERNEL_SLUG}"}
        (output_root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if (WORK / "scale_data").exists():
            shutil.rmtree(WORK / "scale_data")
        if SOURCE.exists():
            shutil.rmtree(SOURCE)
        return

    output_dir = STUDY / "training"
    if DATA_INPUT.exists() and (DATA_INPUT / "fineweb_edu").is_dir():
        data_dir = DATA_INPUT
    elif DATA_INPUT.exists() and all((DATA_INPUT / f"{split}.pt").exists() for split in ("train", "val", "test")):
        # Kaggle may flatten a directory uploaded as a dataset zip.  Present
        # that read-only mount through the layout expected by TokenDataset;
        # symlinks avoid copying the 160 MB token window for every job.
        cached_root = DATA / "fineweb_edu"
        cached_root.mkdir(parents=True, exist_ok=True)
        for name in ("metadata.json", "source_manifest.json", "train.pt", "val.pt", "test.pt"):
            source = DATA_INPUT / name
            if source.exists():
                link = cached_root / name
                if not link.exists():
                    link.symlink_to(source)
        data_dir = DATA
    else:
        data_dir = DATA
    run([
        sys.executable, "train.py",
        "--embedding_type", CONDITION,
        "--dataset", CONFIG["train"]["dataset"],
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--run_name", CONDITION,
        "--device", "cuda",
        "--seed", str(SEED),
        "--steps", str(CONFIG["train"]["steps"]),
        "--batch_size", str(CONFIG["train"]["batch_size"]),
        "--grad_accum_steps", str(CONFIG["train"]["grad_accum_steps"]),
        "--block_size", str(CONFIG["model"]["block_size"]),
        "--n_layer", str(CONFIG["model"]["n_layer"]),
        "--n_head", str(CONFIG["model"]["n_head"]),
        "--n_embd", str(CONFIG["model"]["n_embd"]),
        "--adapter_rank", str(CONFIG["model"]["adapter_rank"]),
        "--adapter_alpha", str(CONFIG["model"]["adapter_alpha"]),
        "--capacity_control_width", str(CONFIG["model"]["capacity_control_width"]),
        "--learning_rate", str(CONFIG["train"]["learning_rate"]),
        "--min_learning_rate", str(CONFIG["train"]["min_learning_rate"]),
        "--warmup_steps", str(CONFIG["train"]["warmup_steps"]),
        "--weight_decay", str(CONFIG["train"]["weight_decay"]),
        "--grad_clip", str(CONFIG["train"]["grad_clip"]),
        "--eval_interval", str(CONFIG["train"]["eval_interval"]),
        "--eval_batches", str(CONFIG["train"]["eval_batches"]),
        "--log_interval", str(CONFIG["train"]["log_interval"]),
        "--save_interval", str(CONFIG["train"]["save_interval"]),
        "--no-save_optimizer",
    ], cwd=SOURCE)
    run_dir = output_dir / CONDITION
    checkpoint = run_dir / "last.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"missing final checkpoint: {checkpoint}")
    for suite in CONFIG["train"]["benchmark_suites"]:
        run([
            sys.executable, "scale_lm_eval.py",
            "--checkpoint", str(checkpoint),
            "--output_dir", str(STUDY / "benchmarks"),
            "--suite", suite,
            "--device", "cuda",
            "--batch_size", str(CONFIG["train"]["benchmark_batch_size"]),
        ], cwd=SOURCE)
    run_manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest = {
        "kind": "scale3_run",
        "experiment_id": CONFIG["experiment_id"],
        "kernel": f"{OWNER}/{KERNEL_SLUG}",
        "seed": SEED,
        "embedding_type": CONDITION,
        "git_commit": actual_commit,
        "source_url": SOURCE_URL,
        "protocol": "EXPERIMENT_PROTOCOL_SCALE.md",
        "config": CONFIG,
        "run_manifest": run_manifest,
        "checkpoint": {"name": checkpoint.name, "sha256": sha256(checkpoint), "step": 19999},
        "started_at_unix": started,
        "finished_at_unix": time.time(),
        "hardware": {"python": sys.version, "platform": platform.platform(), "pip_freeze": capture([sys.executable, "-m", "pip", "freeze"]), "nvidia_smi": capture(["nvidia-smi"])},
    }
    STUDY.mkdir(parents=True, exist_ok=True)
    (STUDY / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for path in STUDY.rglob("*.pt"):
        path.unlink()
    if DATA.exists():
        shutil.rmtree(DATA)
    if SOURCE.exists():
        shutil.rmtree(SOURCE)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
'''


def metadata(kernel_id: str, title: str, dataset_sources: list[str] | None = None) -> dict:
    return {
        "id": kernel_id,
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mode", choices=["probe", "data_bundle", "train"], required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="tied")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--slug-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.commit = subprocess.check_output(
            ["git", "rev-parse", f"{args.commit}^{{commit}}"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"cannot resolve --commit {args.commit!r} in the local checkout") from exc
    if args.mode == "probe":
        kernel_slug = f"llm-embeddings-scale3-probe{args.slug_suffix}"
    elif args.mode == "data_bundle":
        kernel_slug = f"llm-embeddings-scale3-data-bundle{args.slug_suffix}"
    else:
        if args.seed not in SEEDS:
            raise SystemExit(f"seed must be one of {SEEDS}")
        kernel_slug = slug(args.condition, args.seed, args.slug_suffix)
    config = {"experiment_id": EXPERIMENT_ID, "model": MODEL, "train": TRAIN}
    generated = ROOT / "kaggle" / "generated" / kernel_slug
    if generated.exists():
        shutil.rmtree(generated)
    generated.mkdir(parents=True)
    rendered = RUN_TEMPLATE.replace("__COMMIT__", args.commit)
    rendered = rendered.replace("__MODE__", args.mode)
    rendered = rendered.replace("__CONDITION__", args.condition)
    rendered = rendered.replace("__SEED__", str(args.seed))
    rendered = rendered.replace("__OWNER__", args.owner)
    rendered = rendered.replace("__KERNEL_SLUG__", kernel_slug)
    rendered = rendered.replace("__DATASET_SLUG__", DATASET_SLUG)
    rendered = rendered.replace("__CONFIG_JSON__", json.dumps(config, sort_keys=True))
    (generated / "run.py").write_text(rendered)
    kernel_id = f"{args.owner}/{kernel_slug}"
    dataset_sources = [DATASET_SOURCE] if args.mode == "train" else []
    (generated / "kernel-metadata.json").write_text(json.dumps(metadata(kernel_id, kernel_slug, dataset_sources), indent=2) + "\n")
    print(kernel_id)
    if args.dry_run:
        return
    kaggle = kaggle_executable()
    status = subprocess.run([kaggle, "kernels", "status", kernel_id], capture_output=True, text=True)
    if status.returncode == 0:
        raise SystemExit(f"Kaggle kernel already exists: {kernel_id}; use --slug-suffix")
    subprocess.run([kaggle, "kernels", "push", "-p", str(generated)], check=True)


if __name__ == "__main__":
    main()
