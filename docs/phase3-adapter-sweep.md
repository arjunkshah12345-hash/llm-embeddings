# Phase 3 — adapter budget sweep

The released confirmatory rank study uses ranks 1, 2, 4, 8, 16, and 32 with
three seeds at 10,000 steps. Its aggregate and Pareto plots are under
`results/rank_sweep/`. The rank study does not identify a stable improvement
over tied embeddings; the exploratory tooling records below are retained for
historical reproduction.

Phase 3 tests whether rank 8 is a useful operating point or an arbitrary choice. The runner creates one validated study per rank/scaling condition and then produces a top-level parameter-quality summary.

## Tooling check

Run a small matrix before allocating a longer budget:

```bash
python3 adapter_sweep.py \
  --output_dir runs/adapter-tooling-check \
  --ranks 1 2 \
  --alphas 4 8 \
  --seeds 1337 \
  --steps 50 \
  --batch_size 1 \
  --block_size 64 \
  --n_layer 2 --n_head 2 --n_embd 64 \
  --eval_interval 25 \
  --eval_batches 2 \
  --log_interval 10 \
  --save_interval 50 \
  --warmup_steps 5
```

The command writes one directory per condition, each with its own `study_validation.json`, analysis output, and raw run metrics. The top-level directory contains `adapter_sweep_summary.json`, `adapter_sweep_summary.md`, and `adapter_tradeoff.png`.

## Research sweep

The historical command used to launch the predeclared rank set was:

```bash
python3 adapter_sweep.py \
  --output_dir runs/adapter-study-001 \
  --ranks 1 2 4 8 16 32 \
  --alphas 4 8 16 \
  --seeds 1337 2027 31415 \
  --steps 10000 \
  --batch_size 2 \
  --block_size 256 \
  --n_layer 6 --n_head 6 --n_embd 384 \
  --eval_interval 200 \
  --eval_batches 20 \
  --log_interval 50 \
  --save_interval 1000 \
  --warmup_steps 100
```

The released analysis does not select a winning rank from these validation
results. All ranks are reported with paired seed differences and parameter,
FLOP, and wall-clock costs.
