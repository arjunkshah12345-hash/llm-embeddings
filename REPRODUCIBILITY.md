# Reproducibility

This document describes the smallest reproducible run and the checks required before comparing embedding variants. Generated data, checkpoints, metrics, and plots belong in ignored directories such as `data/` and `runs/`; they are not part of the source repository.

## Environment

Install the Python dependencies from a clean checkout:

```bash
python3 -m pip install -r requirements.txt
```

The substantive Phase 2 runs use the exact versions recorded in [requirements-lock.txt](requirements-lock.txt). The lock records the core Python packages and the Python version used for the reference environment; `requirements.txt` remains the portable installation entry point for other CPU/CUDA platforms.

The code requires Python 3.9 or newer and PyTorch 2.1 or newer. CUDA, Apple MPS, and CPU are supported; `--device auto` selects CUDA, then MPS, then CPU.

## Smallest end-to-end experiment

From the repository root:

```bash
python3 sweep.py \
  --output_dir runs/smoke \
  --dataset wikitext2 --device cpu --seeds 1337 \
  --steps 8 --batch_size 1 --block_size 32 \
  --n_layer 1 --n_head 1 --n_embd 16 \
  --adapter_rank 2 --adapter_alpha 2 \
  --warmup_steps 2 --eval_interval 4 --eval_batches 2 \
  --log_interval 2 --save_interval 8 --no-save_optimizer
```

This trains tied, untied, and partial models with the same seed and token budget. The sweep validates the matrix before running analysis. The run is intentionally too short and too small to support a scientific claim.

To inspect the comparison manually:

```bash
python3 validate_study.py --runs_dir runs/smoke
python3 analyze.py --runs_dir runs/smoke --output_dir runs/smoke/analysis
```

## Baseline study

The first substantive study should use the shared configuration in [docs/phase2-baseline.md](docs/phase2-baseline.md), beginning with 10,000 steps and at least three seeds. Keep the same dataset, model flags, optimizer flags, and step count for every embedding type. Use `--resume_existing` only when the study was created with optimizer checkpoints enabled.

Every run records:

- exact model and training configuration;
- dataset split hashes and token counts;
- tokenizer and package metadata;
- git commit and command line;
- parameter counts, training tokens, speed, memory, and approximate FLOPs;
- train/validation metrics and embedding diagnostics.

`validate_study.py` refuses incomplete or unfair comparisons. It checks the full seed-by-embedding matrix, shared configuration, pinned source metadata and dataset hashes, finite loss/perplexity, a validation record at exactly the configured final step, and exactly equal token exposure. Treat a failed validation as a failed experiment rather than analyzing around it.

## Data

The default corpus is WikiText-2-raw-v1, with each split downloaded from the pinned URLs and verified against the SHA-256 values declared in [`data.py`](data.py). The source is the raw variant described by the [Salesforce WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext), rather than the preprocessed word-level files used by the old PyTorch example. Tiny Shakespeare is downloaded from a pinned `char-rnn` commit, verified by SHA-256, and split deterministically into disjoint contiguous train, validation, and test portions. Dataset source metadata and final file hashes are written into each run manifest.

The repository also provides a deterministic `fixture` dataset for CI integration tests. It is synthetic and must never be used as research evidence.

Do not commit downloaded data, model checkpoints, optimizer states, generated plots, or run directories. The repository `.gitignore` covers these artifacts.

## Verification

Run the source checks before sharing changes:

```bash
python3 -m py_compile config.py data.py token_classes.py model.py train.py evaluate.py analyze.py sweep.py adapter_sweep.py embedding_eval.py mechanism_eval.py validate_study.py test_model.py test_data.py test_token_classes.py test_study.py test_sweep.py test_analyze.py test_adapter_sweep.py test_embedding_eval.py
python3 -m pytest -q
```

The GitHub Actions workflow repeats compilation and the test suite on pushes and pull requests. Results from short smoke tests belong in the exploratory record; they must not be described as evidence for the main hypothesis.

After a validated study, the checkpoint-level mechanism report can be regenerated with:

```bash
python3 mechanism_eval.py --runs_dir runs/phase2-pilot-10k --checkpoint last.pt
```

This command refuses to analyze a study whose fairness validator did not pass. The fixed-batch report complements the stepwise JSONL measurements and is written to `mechanism_metrics.json`.

Input and output representation probes can be run together from a validated checkpoint:

```bash
python3 embedding_eval.py \
  --run_dir runs/phase2-pilot-10k/seed1337_partial \
  --checkpoint last.pt --side both \
  --pairs eval/semantic_pairs.jsonl \
  --output runs/phase2-pilot-10k/seed1337_partial/embedding_eval.json
```

The pair file is a small, predeclared single-token probe. It is an exploratory representation diagnostic, not a substitute for a downstream task.

Rank sweeps default to compact checkpoints without optimizer state to keep exploratory artifacts small. Pass `--save_optimizer` only when an interrupted rank condition must be resumed.
