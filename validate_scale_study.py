"""Strict fairness and completeness gate for Study 3 collected artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_CONDITIONS = ("tied", "partial", "untied", "capacity_control")
EXPECTED_SEEDS = (1337, 2027, 31415)
EXPECTED_STEPS = 20_000
EXPECTED_FINAL_STEP = EXPECTED_STEPS - 1
EXPECTED_TOKENS = 20_480_000
TRAINING_SOURCE_FILES = (
    "config.py",
    "data.py",
    "model.py",
    "scale_lm_eval.py",
    "token_classes.py",
    "train.py",
    "requirements-scale-lock.txt",
)
EXPECTED_BENCHMARK_TASKS = {
    "core": (
        "lambada_open",
        "hellaswag",
        "piqa",
        "winogrande",
        "arc_easy",
        "arc_challenge",
        "sciq",
        "openbookqa",
        "boolq",
        "commonsense_qa",
    ),
    "extended": ("mmlu", "truthfulqa_mc1", "triviaqa", "gsm8k"),
}


def read_json(path: Path):
    return json.loads(path.read_text())


def training_source_fingerprint(commit: str) -> str:
    """Hash substantive training files at a recorded repository commit."""
    digest = hashlib.sha256()
    for filename in TRAINING_SOURCE_FILES:
        content = subprocess.check_output(["git", "show", f"{commit}:{filename}"])
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def final_metric(run_dir: Path, split: str, step: int) -> dict:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    matches = [row for row in rows if row.get("split") == split and int(row.get("step", -1)) == step]
    if len(matches) != 1:
        raise ValueError(f"{run_dir}: expected one {split} metric at step {step}, got {len(matches)}")
    return matches[0]


def validate(root: Path, conditions=EXPECTED_CONDITIONS, seeds=EXPECTED_SEEDS) -> dict:
    errors: list[str] = []
    cells = {}
    for seed in seeds:
        for condition in conditions:
            cell_root = root / f"scale3_fineweb_20m_seed{seed}_{condition}"
            manifest_path = cell_root / "artifact_manifest.json"
            if not manifest_path.exists():
                errors.append(f"missing {seed}/{condition}: {cell_root}")
                continue
            manifest = read_json(manifest_path)
            if manifest.get("kind") != "scale3_run":
                errors.append(f"{cell_root}: wrong artifact kind")
            if manifest.get("seed") != seed or manifest.get("embedding_type") != condition:
                errors.append(f"{cell_root}: seed/condition metadata mismatch")
            training_manifest = cell_root / "training" / condition / "manifest.json"
            metrics_path = cell_root / "training" / condition / "metrics.jsonl"
            if not training_manifest.exists() or not metrics_path.exists():
                errors.append(f"{cell_root}: missing training manifest or metrics")
                continue
            training = read_json(training_manifest)
            try:
                final_val = final_metric(cell_root / "training" / condition, "val", EXPECTED_FINAL_STEP)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if int(final_val.get("tokens_seen", -1)) != EXPECTED_TOKENS:
                errors.append(f"{cell_root}: final token exposure is {final_val.get('tokens_seen')}")
            stream = training.get("batch_stream", {})
            if not stream.get("digest") or int(stream.get("token_count", -1)) != EXPECTED_TOKENS:
                errors.append(f"{cell_root}: missing or incomplete actual batch stream digest")
            dataset = training.get("dataset", {})
            cells[(seed, condition)] = {
                "root": cell_root,
                "manifest": manifest,
                "training": training,
                "final_val": final_val,
                "dataset": dataset,
                "stream": stream,
            }

    for seed in seeds:
        reference = cells.get((seed, conditions[0]))
        if reference is None:
            continue
        ref_dataset = json.dumps(reference["dataset"], sort_keys=True)
        ref_stream = reference["stream"].get("digest")
        ref_tokens = reference["final_val"].get("tokens_seen")
        for condition in conditions[1:]:
            cell = cells.get((seed, condition))
            if cell is None:
                continue
            if json.dumps(cell["dataset"], sort_keys=True) != ref_dataset:
                errors.append(f"seed {seed}: dataset metadata differs for {condition}")
            if cell["stream"].get("digest") != ref_stream:
                errors.append(f"seed {seed}: sampled token stream differs for {condition}")
            if cell["final_val"].get("tokens_seen") != ref_tokens:
                errors.append(f"seed {seed}: token exposure differs for {condition}")

    commits = {cell["manifest"].get("git_commit") for cell in cells.values()}
    fingerprints = {}
    for commit in commits:
        try:
            fingerprints[commit] = training_source_fingerprint(commit)
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"cannot fingerprint training source commit {commit}: {exc}")
    if len(set(fingerprints.values())) != 1:
        errors.append(f"substantive training source differs across commits: {fingerprints}")
    configs = {json.dumps(cell["manifest"].get("config"), sort_keys=True) for cell in cells.values()}
    if len(configs) != 1:
        errors.append("Study 3 configs differ across cells")
    for cell in cells.values():
        for suite in ("core", "extended"):
            benchmark = cell["root"] / "benchmarks" / f"lm_eval_{suite}.json"
            if not benchmark.exists():
                errors.append(f"{cell['root']}: missing {suite} benchmark output")
            else:
                payload = read_json(benchmark)
                if payload.get("harness", {}).get("commit") != "ddd67220430a2470529f25fd5c05a576ca1057a0":
                    errors.append(f"{cell['root']}: benchmark harness commit mismatch")
                expected_tasks = list(EXPECTED_BENCHMARK_TASKS[suite])
                if payload.get("tasks") != expected_tasks:
                    errors.append(f"{cell['root']}: {suite} task list differs from frozen protocol")
                results = payload.get("results", {})
                for task in expected_tasks:
                    entry = results.get(task)
                    if not isinstance(entry, dict) or entry.get("status") != "complete" or not entry.get("result"):
                        errors.append(f"{cell['root']}: {suite}/{task} did not complete")

    payload = {
        "study": "scale3_fineweb_20m",
        "expected_conditions": list(conditions),
        "expected_seeds": list(seeds),
        "expected_steps": EXPECTED_STEPS,
        "expected_final_step": EXPECTED_FINAL_STEP,
        "expected_tokens": EXPECTED_TOKENS,
        "cell_count": len(cells),
        "passed": not errors and len(cells) == len(conditions) * len(seeds),
        "errors": errors,
        "git_commits": sorted(commits),
        "training_source_fingerprints": fingerprints,
    }
    (root / "study_validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", required=True, type=Path)
    args = parser.parse_args()
    payload = validate(args.runs_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
