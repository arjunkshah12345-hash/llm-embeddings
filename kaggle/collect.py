"""Collect and verify one completed Kaggle artifact bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="owner/kernel-slug")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--expected-experiment", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = Path(args.destination)
    if destination.exists():
        if not args.overwrite:
            raise SystemExit(f"destination already exists: {destination}; use --overwrite to replace it")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llm-embeddings-kaggle-") as temp:
        output = Path(temp) / "output"
        subprocess.run(["kaggle", "kernels", "output", args.kernel, "-p", str(output), "--force"], check=True)
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
