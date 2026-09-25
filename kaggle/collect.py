"""Collect and verify one completed Kaggle artifact bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def kaggle_executable() -> str:
    """Return an existing Kaggle CLI executable, even if PATH has a stale shim."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="owner/kernel-slug")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--expected-experiment", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_transient_download_error(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "connection broken",
            "incompleteread",
            "connection reset",
            "connection aborted",
            "maxretryerror",
            "name resolution",
            "temporary failure",
            "timed out",
            "502 bad gateway",
            "503 service unavailable",
        )
    )


def download_kernel_output(kaggle: str, kernel: str, output: Path) -> None:
    command = [kaggle, "kernels", "output", kernel, "-p", str(output), "--force"]
    for attempt in range(5):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.stdout.strip():
            print(result.stdout, end="")
        if result.returncode == 0:
            if result.stderr.strip():
                print(result.stderr, end="")
            return
        error = (result.stdout + "\n" + result.stderr).strip()
        if attempt == 4 or not is_transient_download_error(error):
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        print(f"transient Kaggle output download failure; retrying ({attempt + 1}/5)", flush=True)
        time.sleep(30)


def main() -> None:
    args = parse_args()
    kaggle = kaggle_executable()
    destination = Path(args.destination)
    if destination.exists():
        if not args.overwrite:
            raise SystemExit(f"destination already exists: {destination}; use --overwrite to replace it")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llm-embeddings-kaggle-") as temp:
        output = Path(temp) / "output"
        download_kernel_output(kaggle, args.kernel, output)
        manifests = list(output.rglob("artifact_manifest.json"))
        if len(manifests) != 1:
            raise SystemExit(f"expected one artifact_manifest.json, found {len(manifests)}")
        manifest = json.loads(manifests[0].read_text())
        if args.expected_commit and manifest.get("git_commit") != args.expected_commit:
            raise SystemExit(
                f"artifact commit mismatch: expected {args.expected_commit}, got {manifest.get('git_commit')}"
            )
        if args.expected_experiment and manifest.get("experiment_id") != args.expected_experiment:
            raise SystemExit(
                f"artifact experiment mismatch: expected {args.expected_experiment}, got {manifest.get('experiment_id')}"
            )
        source_root = manifests[0].parent
        shutil.copytree(source_root, destination)
    print(json.dumps(json.loads((destination / "artifact_manifest.json").read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
