# Phase 3 — adapter budget sweep

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

After the tooling check, run the predeclared rank set with the same model and training settings as Phase 2:

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

Do not select a rank from this sweep using the same validation result that is later presented as a final comparison. Freeze the adapter family and rank using the predeclared gate, then evaluate the selected configuration on held-out test data and representation probes.
