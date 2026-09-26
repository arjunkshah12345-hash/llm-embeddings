"""Submit, monitor, collect, and gate the frozen Study 3 Kaggle matrix."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from scale_launch import CONDITIONS, EXPERIMENT_ID, OWNER, SEEDS, kaggle_executable, slug


ROOT = Path(__file__).resolve().parents[1]


def status(kaggle: str, kernel: str) -> str:
    result = subprocess.run([kaggle, "kernels", "status", kernel], capture_output=True, text=True, check=False)
    output = result.stdout + "\n" + result.stderr
    if result.returncode:
        if "not found" in output.lower() or "permission 'kernels.get' was denied" in output.lower():
            return "MISSING"
        raise RuntimeError(output.strip())
    upper = output.upper()
    if any(x in upper for x in ("RUNNING", "QUEUED", "STARTING")):
        return "RUNNING"
    if any(x in upper for x in ("COMPLETE", "SUCCEEDED")):
        return "COMPLETE"
    if any(x in upper for x in ("ERROR", "FAILED", "CANCEL")):
        return "FAILED"
    return "UNKNOWN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stage", choices=["probe", "train"], required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--output-root", default="cloud_artifacts/scale3")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--slug-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        args.commit = subprocess.check_output(
            ["git", "rev-parse", f"{args.commit}^{{commit}}"], cwd=ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"cannot resolve --commit {args.commit!r} in the local checkout") from exc
    kaggle = kaggle_executable()
    jobs = []
    if args.stage == "probe":
        jobs.append(("probe", 1337, f"{args.owner}/llm-embeddings-scale3-probe{args.slug_suffix}"))
    else:
        for seed in args.seeds:
            for condition in args.conditions:
                jobs.append((condition, seed, f"{args.owner}/{slug(condition, seed, args.slug_suffix)}"))
    if args.dry_run:
        for condition, seed, kernel in jobs:
            print(f"{condition}\t{seed}\t{kernel}")
        return

    remaining = []
    active = []
    completed = []
    for condition, seed, kernel in jobs:
        state = status(kaggle, kernel)
        if state == "COMPLETE":
            completed.append((condition, seed, kernel))
        elif state == "MISSING":
            remaining.append((condition, seed, kernel))
        else:
            active.append((condition, seed, kernel))

    def submit(job: tuple[str, int, str]) -> None:
        condition, seed, _kernel = job
        command = [
            os.environ.get("PYTHON", "python3"),
            "kaggle/scale_launch.py",
            "--commit", args.commit,
            "--mode", args.stage,
            "--owner", args.owner,
            "--seed", str(seed),
        ]
        if args.stage == "train":
            command.extend(["--condition", condition])
        if args.slug_suffix:
            command.append(f"--slug-suffix={args.slug_suffix}")
        subprocess.run(command, cwd=ROOT, check=True)

    # Kaggle currently allows two active GPU sessions for this account. Keep
    # the queue explicit so a quota rejection cannot be mistaken for a job.
    max_active = 2
    while active or remaining:
        while remaining and len(active) < max_active:
            job = remaining.pop(0)
            submit(job)
            state = status(kaggle, job[2])
            if state == "MISSING":
                raise SystemExit(f"Kaggle submission did not create a job: {job[2]}")
            active.append(job)
        states = {f"{condition}:{seed}": status(kaggle, kernel) for condition, seed, kernel in active}
        print(states, flush=True)
        failed = [name for name, state in states.items() if state == "FAILED"]
        if failed:
            raise SystemExit(f"Kaggle jobs failed: {', '.join(failed)}")
        finished = [job for job in active if states[f"{job[0]}:{job[1]}"] == "COMPLETE"]
        completed.extend(finished)
        active = [job for job in active if states[f"{job[0]}:{job[1]}"] != "COMPLETE"]
        if active or remaining:
            time.sleep(args.poll_seconds)

    if args.stage == "probe":
        destination = ROOT / args.output_root / "probe"
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            os.environ.get("PYTHON", "python3"), "kaggle/collect.py",
            "--kernel", completed[0][2], "--destination", str(destination),
            "--expected-commit", args.commit, "--expected-experiment", "scale3_probe",
        ], cwd=ROOT, check=True)
        return

    for condition, seed, kernel in completed:
        destination = ROOT / args.output_root / f"{EXPERIMENT_ID}_seed{seed}_{condition}"
        if destination.exists():
            continue
        subprocess.run([
            os.environ.get("PYTHON", "python3"), "kaggle/collect.py",
            "--kernel", kernel, "--destination", str(destination),
            "--expected-commit", args.commit, "--expected-experiment", EXPERIMENT_ID,
        ], cwd=ROOT, check=True)
    subprocess.run([
        os.environ.get("PYTHON", "python3"), "validate_scale_study.py",
        "--runs_dir", str(ROOT / args.output_root),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
