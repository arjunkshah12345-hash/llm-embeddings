"""Launch, monitor, and collect a matched Kaggle profile matrix.

This script only submits and collects Kaggle kernels. Model training happens in
the submitted kernels, never in the local process.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OWNER = "aks1321"
PROFILE_EXPERIMENTS = {
    "primary": "primary_10k",
    "rank": "rank_sweep_10k",
    "long": "long_50k",
    "small_scale": "small_scale_10k",
    "second_dataset": "tiny_shakespeare_10k",
    "mechanism_stop_input": "mechanism_stop_input_10k",
    "mechanism_stop_output": "mechanism_stop_output_10k",
}


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


def kernel_slug(profile: str, seed: int, suffix: str = "") -> str:
    return f"llm-embeddings-{PROFILE_EXPERIMENTS[profile].replace('_', '-')}-seed{seed}{suffix}"


def classify_status(returncode: int, output: str) -> str:
    """Normalize Kaggle's status output, including its private-kernel 403."""
    if returncode:
        lowered = output.lower()
        if "not found" in lowered or "permission 'kernels.get' was denied" in lowered:
            return "MISSING"
        raise RuntimeError(output.strip() or "Kaggle status command failed")
    upper = output.upper()
    if any(value in upper for value in ("RUNNING", "QUEUED", "STARTING")):
        return "RUNNING"
    if any(value in upper for value in ("COMPLETE", "SUCCEEDED")):
        return "COMPLETE"
    if any(value in upper for value in ("ERROR", "FAILED", "CANCEL")):
        return "FAILED"
    return "UNKNOWN"


def kernel_status(kaggle: str, kernel: str) -> str:
    result = subprocess.run(
        [kaggle, "kernels", "status", kernel],
        capture_output=True,
        text=True,
        check=False,
    )
    return classify_status(result.returncode, result.stdout + "\n" + result.stderr)


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def destination_for(root: Path, profile: str, seed: int, suffix: str = "") -> Path:
    return root / f"{profile}_seed{seed}{suffix}"


def launch_command(profile: str, commit: str, owner: str, seed: int, suffix: str = "") -> list[str]:
    command = [
        os.environ.get("PYTHON", "python3"), "kaggle/launch.py",
        "--profile", profile, "--commit", commit,
        "--owner", owner, "--seeds", str(seed),
    ]
    if suffix:
        # argparse interprets a separate value beginning with '-' as another
        # option. Attach rerun suffixes to preserve values such as '-e8'.
        command.append(f"--slug-suffix={suffix}")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILE_EXPERIMENTS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--commit", required=True, help="immutable repository commit used by every kernel")
    parser.add_argument("--owner", default=OWNER)
    parser.add_argument("--output-root", default="cloud_artifacts")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--slug-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_seconds < 1:
        raise SystemExit("--poll-seconds must be positive")
    kaggle = kaggle_executable()
    output_root = ROOT / args.output_root
    jobs = []
    for profile in args.profiles:
        for seed in args.seeds:
            slug = kernel_slug(profile, seed, args.slug_suffix)
            jobs.append((profile, seed, f"{args.owner}/{slug}"))

    if args.dry_run:
        for profile, seed, kernel in jobs:
            print(f"{profile}\t{seed}\t{kernel}\t{destination_for(output_root, profile, seed, args.slug_suffix)}")
        return

    pending_jobs = []
    for profile, seed, kernel in jobs:
        destination = destination_for(output_root, profile, seed, args.slug_suffix)
        if destination.exists():
            validation = destination / "study_validation.json"
            artifact = destination / "artifact_manifest.json"
            if validation.exists() and artifact.exists():
                validation_payload = json.loads(validation.read_text())
                artifact_payload = json.loads(artifact.read_text())
                if not validation_payload.get("passed"):
                    raise SystemExit(f"collected artifact failed validation: {destination}")
                if artifact_payload.get("git_commit") != args.commit:
                    raise SystemExit(f"collected artifact commit differs from requested commit: {destination}")
                if artifact_payload.get("experiment_id") != PROFILE_EXPERIMENTS[profile]:
                    raise SystemExit(f"collected artifact profile differs from requested profile: {destination}")
                print(f"skip collected {profile} seed={seed}: {destination}", flush=True)
                continue
            raise SystemExit(f"destination exists without validation: {destination}; inspect before replacing")
        pending_jobs.append((profile, seed, kernel))
        if kernel_status(kaggle, kernel) == "MISSING":
            run(launch_command(profile, args.commit, args.owner, seed, args.slug_suffix))

    pending = set(pending_jobs)
    while pending:
        states = {f"{profile}:{seed}": kernel_status(kaggle, kernel) for profile, seed, kernel in sorted(pending)}
        print(states, flush=True)
        failed = [key for key, state in states.items() if state == "FAILED"]
        if failed:
            raise SystemExit(f"Kaggle jobs failed: {', '.join(failed)}")
        pending = {
            item for item in pending
            if states[f"{item[0]}:{item[1]}"] != "COMPLETE"
        }
        if pending:
            time.sleep(args.poll_seconds)

    for profile, seed, kernel in pending_jobs:
        destination = destination_for(output_root, profile, seed, args.slug_suffix)
        if destination.exists():
            continue
        run([
            os.environ.get("PYTHON", "python3"), "kaggle/collect.py",
            "--kernel", kernel,
            "--destination", str(destination),
            "--expected-commit", args.commit,
            "--expected-experiment", PROFILE_EXPERIMENTS[profile],
        ])


if __name__ == "__main__":
    main()
