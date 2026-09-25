# Reproducibility

This repository contains a completed, cloud-trained study of tied, untied, and low-rank partially tied input/output embeddings. Training is performed in Kaggle kernels; local commands collect artifacts, validate comparisons, regenerate analyses, run tests, and compile the paper.

## Environment

```bash
python3 -m pip install -r requirements.txt
```

The Kaggle reference versions are pinned in [requirements-lock.txt](requirements-lock.txt). The lock records the Python 3.12.13 reference environment and core package versions. `requirements.txt` remains the portable installation entry point for compatible CPU/CUDA platforms.

## Data and protocol

The main corpus is WikiText-2-raw-v1 with GPT-2 BPE tokenization. Split URLs and SHA-256 values are pinned in [data.py](data.py). Tiny Shakespeare uses a pinned source commit and hash with deterministic disjoint splits. Every run records source metadata, split hashes, tokenizer metadata, configuration, package versions, git commit, sampled-offset digest, parameter counts, metrics, throughput, memory, and estimated FLOPs.

The main 50k comparison is:

- six layers, six heads, width 384, block size 256;
- tied, untied, rank-8 partial, and parameter-matched capacity control;
- seeds 1337, 2027, and 31415;
- 50,000 fresh optimizer steps, batch size 2, warmup 100, cosine decay from `3e-4` to `3e-5`;
- identical tokenizer, optimizer, validation windows, data order, token budget, and initialization protocol within each seed.

The preserved 10k Study 1 adds input-only and output-only conditions. The confirmatory rank study uses ranks 1, 2, 4, 8, 16, and 32 for three seeds. Robustness uses a smaller WikiText-2 model and Tiny Shakespeare. Mechanism profiles run matched tied/partial path interventions with input or output gradients stopped for steps `[0, 5000)`.

## Kaggle training commands

Use the source commits recorded in the final artifacts. These commands submit cloud kernels and collect compact outputs; they do not train locally:

```bash
python3 kaggle/orchestrate.py --profiles long --seeds 1337 2027 31415 \
  --commit 56f6360ad93eadff2c6ad1406a4805264549747f
python3 kaggle/orchestrate.py --profiles rank --seeds 1337 2027 31415 \
  --commit e8aa5e87b07810596940df5f66b5c92c7d825569
python3 kaggle/orchestrate.py --profiles small_scale second_dataset \
  --seeds 1337 2027 31415 --commit 77923885b40f9e80c0f0a0f6fb6e94e9bdfc51db
python3 kaggle/orchestrate.py \
  --profiles mechanism_stop_input mechanism_stop_output \
  --seeds 1337 2027 31415 --commit 77923885b40f9e80c0f0a0f6fb6e94e9bdfc51db
```

The 50k study must be launched in a fresh output directory. Do not resume a 10k run with a changed schedule horizon. The launcher rejects unsafe step-horizon changes when resuming.

## Validation and aggregation

Validate each collected seed artifact before combining it:

```bash
python3 validate_study.py --runs_dir cloud_artifacts/long_seed1337_exact56f
python3 validate_study.py --runs_dir cloud_artifacts/long_seed2027
python3 validate_study.py --runs_dir cloud_artifacts/long_seed31415
```

The validator requires the complete seed-by-condition matrix, matching configs and pinned data hashes, finite metrics, a validation record at the exact declared final step, and exact equality of actual token exposure and sampled-offset digests. A failed gate is a failed study and must not be analyzed around.

Regenerate the primary, rank, robustness, and mechanism artifacts with:

```bash
python3 aggregate_results.py \
  --studies cloud_artifacts/long_seed1337_exact56f \
            cloud_artifacts/long_seed2027 \
            cloud_artifacts/long_seed31415 \
  --output_dir results/long_50k
python3 aggregate_rank.py \
  --studies cloud_artifacts/rank_seed1337_e8 \
            cloud_artifacts/rank_seed2027 \
            cloud_artifacts/rank_seed31415_e8 \
  --tied-results results/primary/primary_aggregate.json \
  --output-dir results/rank_sweep
python3 aggregate_secondary.py \
  --studies small_scale=results/small_scale/primary_aggregate.json \
            tiny_shakespeare=results/tiny_shakespeare/primary_aggregate.json \
  --output-dir results/secondary
python3 aggregate_mechanism.py \
  --studies cloud_artifacts/mechanism_stop_input_seed1337 \
            cloud_artifacts/mechanism_stop_input_seed2027 \
            cloud_artifacts/mechanism_stop_input_seed31415 \
  --output_dir results/mechanism_stop_input
python3 aggregate_mechanism.py \
  --studies cloud_artifacts/mechanism_stop_output_seed1337 \
            cloud_artifacts/mechanism_stop_output_seed2027 \
            cloud_artifacts/mechanism_stop_output_seed31415 \
  --output_dir results/mechanism_stop_output
python3 aggregate_interventions.py \
  --studies stop_input=cloud_artifacts/mechanism_stop_input_seed1337 \
            stop_input=cloud_artifacts/mechanism_stop_input_seed2027 \
            stop_input=cloud_artifacts/mechanism_stop_input_seed31415 \
  --output results/mechanism_interventions_input.json
python3 aggregate_interventions.py \
  --studies stop_output=cloud_artifacts/mechanism_stop_output_seed1337 \
            stop_output=cloud_artifacts/mechanism_stop_output_seed2027 \
            stop_output=cloud_artifacts/mechanism_stop_output_seed31415 \
  --output results/mechanism_interventions_output.json
```

Export compact provenance and raw JSONL metrics without checkpoints:

```bash
python3 export_release_artifacts.py \
  --studies long_seed1337=cloud_artifacts/long_seed1337_exact56f \
            long_seed2027=cloud_artifacts/long_seed2027 \
            long_seed31415=cloud_artifacts/long_seed31415 \
  --output-dir results/release_artifacts/long_50k_final
```

## Rebuild the paper

```bash
python3 paper/generate_figures.py --output-dir results/long_50k/figures
python3 paper/generate_tables.py \
  --primary results/long_50k/primary_aggregate.json \
  --rank results/rank_sweep/rank_aggregate.json \
  --mechanism results/long_50k/mechanism/mechanism_aggregate.json \
  --secondary results/secondary/secondary_aggregate.json \
  --embedding results/long_50k/embedding_eval/embedding_eval_aggregate.json \
  --interventions-input results/mechanism_interventions_input.json \
  --interventions-output results/mechanism_interventions_output.json \
  --output-dir paper/generated
python3 paper/package_arxiv.py --output-dir arxiv
(cd paper && tectonic --keep-logs main.tex)
(cd arxiv && tectonic --keep-logs main.tex)
```

All tables and figures are generated from JSON/JSONL artifacts. The release index is [`results/release_artifacts/final_release_manifest.json`](results/release_artifacts/final_release_manifest.json). The arXiv directory is a self-contained source bundle; it contains no datasets, checkpoints, credentials, or absolute local paths.

## Verification

```bash
python3 -m py_compile $(git ls-files '*.py')
python3 -m pytest -q
python3 -m compileall -q .
git diff --check
```

The full GitHub Actions workflow repeats compilation, the complete pytest suite, and the synthetic sweep → validation → analysis pipeline. The synthetic fixture is explicitly non-training and must never be used as research evidence.

The smallest smoke command remains documented in the repository history and is also available in `docs/initial-sanity-check.md`; it exists only to check wiring. It is too short to support a scientific claim.
