# Phase 2 — baseline training study

This is the first token-matched comparison that can be treated as evidence, not instrumentation.

## Shared configuration

Hold these fixed across all variants and seeds:

| Setting | Value |
|---|---|
| Dataset | WikiText-2 (raw), GPT-2 BPE |
| Seeds | `1337 2027 31415` |
| Steps | `10000` (pilot) then `50000` if signal holds |
| Batch size × grad accum | `2 × 1` |
| Block size | `256` |
| Transformer | `n_layer=6`, `n_head=6`, `n_embd=384` (~30M tied) |
| Adapter | rank `8`, alpha `8` |
| LR schedule | `3e-4` → `3e-5`, warmup `100` |
| Eval | every `200` steps, `20` fixed validation batches |

Token budget must match within 1% across the three embedding types. `sweep.py` writes `study_validation.json` and refuses to analyze the study if the matrix, shared settings, dataset hashes, or token exposure fail validation.

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

Smaller disk-friendly smoke before the pilot (not publishable as Phase 2 evidence):

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
3. Report mean ± std of best and final validation loss/perplexity by embedding type.
4. Report deterministic bootstrap 95% intervals from `results.json` when multiple seeds are available.
5. Report paired seed-level deltas versus tied from `results.json`.
6. Report parameters, extra-vs-tied, tokens/s, peak memory, and `estimated_flops_total`.
7. Plot loss vs tokens and loss vs estimated FLOPs separately.
8. Do not tune one variant on validation and then compare it to untuned variants.

## Disk note

Default ~30M checkpoints are large once optimizer state is included. Keep `optimizer_last.pt` only for active runs; compact `best.pt` / `last.pt` are enough for analysis. Free disk before starting the 10k pilot.
