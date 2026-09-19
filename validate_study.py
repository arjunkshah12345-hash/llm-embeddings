"""Validate the comparability contract for a completed sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config["model"])
    train = dict(config["train"])
    variable_keys = {"seed", "run_name", "embedding_type", "output_dir"}
    return {
        "model": model,
        "train": {key: value for key, value in train.items() if key not in variable_keys},
        "device": config.get("device"),
    }


def _final_tokens(metrics: list[dict[str, Any]]) -> int | None:
    validation = [row for row in metrics if row.get("split") == "val" and "tokens_seen" in row]
    if validation:
        return int(validation[-1]["tokens_seen"])
    training = [row for row in metrics if row.get("split") == "train" and "tokens_seen" in row]
    return int(training[-1]["tokens_seen"]) if training else None


def validate_study(runs_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = runs_dir / "study_manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "errors": [f"missing {manifest_path}"], "warnings": []}

    study = read_json(manifest_path)
    seeds = [int(seed) for seed in study.get("seeds", [])]
    embedding_types = list(study.get("embedding_types", []))
    expected = {(seed, kind) for seed in seeds for kind in embedding_types}
    entries = study.get("runs", [])
    seen: dict[tuple[int, str], dict[str, Any]] = {}
    for entry in entries:
        key = (int(entry["seed"]), entry["embedding_type"])
        if key in seen:
            errors.append(f"duplicate study entry for seed={key[0]} embedding_type={key[1]}")
        seen[key] = entry

    missing = sorted(expected - set(seen))
    unexpected = sorted(set(seen) - expected)
    for seed, kind in missing:
        errors.append(f"missing run for seed={seed} embedding_type={kind}")
    for seed, kind in unexpected:
        errors.append(f"unexpected run for seed={seed} embedding_type={kind}")

    canonical_configs: dict[str, dict[str, Any]] = {}
    dataset_hashes: dict[str, dict[str, str]] = {}
    token_counts: dict[str, int] = {}
    for entry in entries:
        run_name = entry.get("run_name", "")
        run_dir = runs_dir / run_name
        required = ["config.json", "manifest.json", "parameter_counts.json", "metrics.jsonl"]
        missing_files = [name for name in required if not (run_dir / name).exists()]
        if missing_files:
            errors.append(f"{run_name}: missing {', '.join(missing_files)}")
            continue
        try:
            config = read_json(run_dir / "config.json")
            run_manifest = read_json(run_dir / "manifest.json")
            metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"{run_name}: unreadable run metadata ({exc})")
            continue

        try:
            canonical = _canonical_config(config)
        except KeyError as exc:
            errors.append(f"{run_name}: config missing {exc.args[0]}")
            continue
        canonical_configs[run_name] = canonical

        hashes = config.get("dataset", {}).get("sha256", {})
        manifest_hashes = run_manifest.get("dataset", {}).get("sha256", {})
        if hashes != manifest_hashes:
            errors.append(f"{run_name}: config and manifest dataset hashes differ")
        dataset_hashes[run_name] = hashes

        tokens = _final_tokens(metrics)
        if tokens is None or tokens <= 0:
            errors.append(f"{run_name}: no positive token count in metrics")
        else:
            token_counts[run_name] = tokens

    if canonical_configs:
        reference_name, reference = next(iter(canonical_configs.items()))
        for run_name, config in canonical_configs.items():
            if config != reference:
                errors.append(f"{run_name}: shared model/training configuration differs from {reference_name}")

    unique_hashes = {json.dumps(value, sort_keys=True) for value in dataset_hashes.values()}
    if len(unique_hashes) > 1:
        errors.append("dataset SHA-256 hashes differ across runs")

    by_seed: dict[int, list[tuple[str, int]]] = {}
    for entry in entries:
        run_name = entry.get("run_name", "")
        if run_name in token_counts:
            by_seed.setdefault(int(entry["seed"]), []).append((entry["embedding_type"], token_counts[run_name]))
    token_tolerance: dict[str, dict[str, float | int]] = {}
    for seed, rows in by_seed.items():
        values = [tokens for _, tokens in rows]
        if len(values) < len(embedding_types):
            continue
        minimum, maximum = min(values), max(values)
        allowed_delta = max(1, maximum * 0.01)
        token_tolerance[str(seed)] = {
            "minimum": minimum,
            "maximum": maximum,
            "delta": maximum - minimum,
            "allowed_delta": allowed_delta,
        }
        if maximum - minimum > allowed_delta:
            errors.append(f"seed={seed}: token exposure differs by more than 1% across embedding types")

    expected_common = study.get("common", {})
    if expected_common and canonical_configs:
        reference_train = next(iter(canonical_configs.values()))["train"]
        reference_model = next(iter(canonical_configs.values()))["model"]
        actual_common = {
            "dataset": reference_train.get("dataset"),
            "data_dir": reference_train.get("data_dir"),
            "device": reference_train.get("device"),
            "steps": reference_train.get("steps"),
            "batch_size": reference_train.get("batch_size"),
            "grad_accum_steps": reference_train.get("grad_accum_steps"),
            "block_size": reference_model.get("block_size"),
            "n_layer": reference_model.get("n_layer"),
            "n_head": reference_model.get("n_head"),
            "n_embd": reference_model.get("n_embd"),
            "dropout": reference_model.get("dropout"),
            "adapter_rank": reference_model.get("adapter_rank"),
            "adapter_alpha": reference_model.get("adapter_alpha"),
            "learning_rate": reference_train.get("learning_rate"),
            "min_learning_rate": reference_train.get("min_learning_rate"),
            "warmup_steps": reference_train.get("warmup_steps"),
            "weight_decay": reference_train.get("weight_decay"),
            "grad_clip": reference_train.get("grad_clip"),
            "eval_interval": reference_train.get("eval_interval"),
            "eval_batches": reference_train.get("eval_batches"),
            "log_interval": reference_train.get("log_interval"),
            "save_interval": reference_train.get("save_interval"),
        }
        for key, expected_value in expected_common.items():
            if key in actual_common and actual_common[key] != expected_value:
                errors.append(f"study common setting {key!r}={expected_value!r} disagrees with run metadata {actual_common[key]!r}")

    if not entries:
        errors.append("study manifest contains no runs")
    if entries and len(canonical_configs) < len(entries):
        warnings.append("some run metadata could not be included in the configuration comparison")

    return {
        "passed": not errors,
        "runs_dir": str(runs_dir),
        "expected_run_count": len(expected),
        "observed_run_count": len(entries),
        "dataset_hashes": dataset_hashes,
        "token_counts": token_counts,
        "token_tolerance_by_seed": token_tolerance,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = validate_study(Path(args.runs_dir))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
