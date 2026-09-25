# Phase 2 — baseline training study

The final Phase 2 record contains the preserved 10k Study 1 and a fresh-from-step-zero 50k primary study. The 10k endpoint is retained as preliminary context; the 50k endpoint is the primary result used by the paper.

## Shared configuration

Hold these fixed across all variants and seeds:

| Setting | Value |
|---|---|
| Dataset | WikiText-2-raw-v1, GPT-2 BPE; pinned split URLs and SHA-256 values in `data.py` |
| Seeds | `1337 2027 31415` |
| Steps | `10000` Study 1 and fresh `50000` primary |
| Batch size × grad accum | `2 × 1` |
| Block size | `256` |
| Transformer | `n_layer=6`, `n_head=6`, `n_embd=384` (~30M tied) |
| Adapter | rank `8`, alpha `8` |
| LR schedule | `3e-4` → `3e-5`, warmup `100` |
| Eval | every `200` steps, `20` fixed validation batches |

Token exposure must match exactly across the three embedding types. `sweep.py` writes `study_validation.json` and refuses to analyze the study if the matrix, shared settings, pinned dataset metadata, final validation step, or token exposure fail validation.

## Commands

Pilot (three seeds × three models, 10k steps):

```bash
python3 sweep.py \
  --output_dir runs/phase2-pilot-10k \
  --steps 10000 \
  --seeds 1337 2027 31415 \
  --batch_size 2 \
  --block_size 256 \
  --n_layer 6 --n_head 6 --n_embd 384 \
  --adapter_rank 8 \
  --eval_interval 200 \
  --eval_batches 20 \
  --log_interval 50 \
  --save_interval 1000 \
  --warmup_steps 100
```

The released 50k study was launched from step zero in a fresh output directory.
Do not use `--resume_existing` to change the step horizon: the cosine
learning-rate schedule depends on the declared total step count. The final
50k aggregate is under `results/long_50k/` and its compact source artifacts are
under `results/release_artifacts/long_50k/`.

Smaller disk-friendly smoke before the pilot (not Phase 2 evidence):

```bash
python3 sweep.py \
  --output_dir runs/phase2-tooling-check \
  --steps 50 \
  --seeds 1337 \
  --batch_size 1 \
  --block_size 64 \
  --n_layer 2 --n_head 2 --n_embd 64 \
  --eval_interval 25 \
  --eval_batches 2 \
  --log_interval 10 \
  --save_interval 50 \
  --warmup_steps 5
```

## Reporting checklist

Before publishing any claim:

1. Confirm `study_validation.json` has `"passed": true` and `study_manifest.json` lists the intended common flags for every run.
2. Confirm dataset SHA-256 hashes match across manifests.
3. Confirm all logged train/validation losses and perplexities are finite.
4. Report mean ± std of best and final validation loss/perplexity by embedding type.
5. Report deterministic bootstrap 95% intervals from `results.json` when multiple seeds are available.
6. Report paired seed-level deltas versus tied from `results.json`.
7. Report parameters, extra-vs-tied, tokens/s, peak memory, and `estimated_flops_total`.
8. Plot loss vs tokens and loss vs estimated FLOPs separately.
9. Do not tune one variant on validation and then compare it to untuned variants.

## Disk note

Default ~30M checkpoints are large once optimizer state is included. Keep `optimizer_last.pt` only for active runs; compact `best.pt` / `last.pt` are enough for analysis. Free disk before starting the 10k pilot.

Use `--no-save_optimizer` for a non-resumable pilot when disk is limited.
