# Partially tied language-model embeddings

This repository is a small, reproducible experiment for comparing three GPT-style decoder-only language models:

- `tied`: one matrix is used for input embeddings and output logits.
- `untied`: separate input and output matrices.
- `partial`: one shared matrix plus independent low-rank corrections for the input and output roles.

The experiment keeps the transformer, tokenizer, dataset, optimizer, schedule, batch size, seed, and token budget the same across runs. The architectural difference is the embedding module.

The long-term research roadmap is in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). The initial six-step sanity check is documented in [docs/initial-sanity-check.md](docs/initial-sanity-check.md). Phase 2 baseline study commands are in [docs/phase2-baseline.md](docs/phase2-baseline.md), Phase 3 adapter-budget commands are in [docs/phase3-adapter-sweep.md](docs/phase3-adapter-sweep.md), and Phase 5 input evaluation is in [docs/phase5-input-evaluation.md](docs/phase5-input-evaluation.md). The first representation pilot is recorded in [docs/initial-representation-pilot.md](docs/initial-representation-pilot.md).

The prepared repository and remote publishing steps are documented in [docs/publishing.md](docs/publishing.md).

## Setup

```bash
python3 -m pip install -r requirements.txt
```

The first training run downloads the public WikiText-2 raw train, validation, and test files into `data/wikitext2/`. The GPT-2 BPE vocabulary is loaded through `tiktoken`.

## Run the experiment

The default model is about 30M parameters when tied: six transformer blocks, width 384, six heads, and a GPT-2 vocabulary. The untied model is about 49M parameters. Partial tying uses rank-8 adapters by default and adds roughly 0.8M embedding parameters.

For a short sanity check:

```bash
python3 train.py --embedding_type tied   --steps 20 --batch_size 1 --block_size 128 --eval_interval 10 --eval_batches 4
python3 train.py --embedding_type untied --steps 20 --batch_size 1 --block_size 128 --eval_interval 10 --eval_batches 4
python3 train.py --embedding_type partial --steps 20 --batch_size 1 --block_size 128 --eval_interval 10 --eval_batches 4
python3 analyze.py --runs_dir runs --output_dir analysis
```

For a useful first run, increase `--steps` to 1,000–5,000 and use the same flags for all three models. Each command writes to `runs/<embedding_type>/` unless `--run_name` is provided.

```bash
python3 train.py --embedding_type tied
python3 train.py --embedding_type untied
python3 train.py --embedding_type partial --adapter_rank 8
python3 analyze.py
```

For a multi-seed study, use `sweep.py`. It forwards one shared configuration to every model/seed combination, refuses to overwrite completed runs unless asked, writes `study_manifest.json`, validates the complete matrix, dataset hashes, shared settings, and token exposure, and analyzes the full study only after that gate passes:

```bash
python3 sweep.py --output_dir runs/study-001 --steps 2000 --seeds 1337 2027 31415
```

If a study is interrupted, rerun it with the same settings and `--resume_existing`; completed runs are skipped and incomplete runs resume from their latest optimizer checkpoint:

```bash
python3 sweep.py --output_dir runs/study-001 --steps 10000 --seeds 1337 2027 31415 --resume_existing
```

For the predeclared partial-adapter rank/scaling sweep:

```bash
python3 adapter_sweep.py --output_dir runs/adapter-study-001 --ranks 1 2 4 8 16 32 --alphas 4 8 16 --seeds 1337 2027 31415
```

Evaluate a saved checkpoint directly:

```bash
python3 evaluate.py --run_dir runs/partial --checkpoint best.pt
```

Evaluate the effective input representation with deterministic intrinsic probes:

```bash
python3 embedding_eval.py --run_dir runs/partial --checkpoint best.pt
```

## Outputs

Each run directory contains:

- `config.json`: exact model, data, optimizer, and seed settings;
- `manifest.json`: git commit, command, runtime, dataset hashes, and parameter manifest;
- `parameter_counts.json`: total, transformer, and embedding parameter counts;
- `metrics.jsonl`: training/validation losses, best/final perplexity, training-only throughput, wall-clock time, memory, estimated FLOPs (6ND rule), gradient decomposition, and adapter norms;
- `study_validation.json`: the sweep fairness gate and its token/hash/config checks;
- `last.pt` and `best.pt`: compact CPU model checkpoints without optimizer state (evaluation and comparison);
- `optimizer_last.pt`: full resume checkpoint with AdamW state, RNG, and token counters (written by default; disable with `--no-save_optimizer`).

Resume an interrupted run:

```bash
python3 train.py --embedding_type partial --resume runs/partial/optimizer_last.pt --steps 5000
```

Validation uses deterministic fixed token windows, so all model variants and repeated evaluations see the same validation examples.

`analyze.py` creates:

- training and validation loss plots;
- validation loss versus estimated training FLOPs;
- parameter count versus validation loss;
- input/output embedding gradient norms and their output-to-input ratio (`gradient_norms_and_ratio.png`);
- input/output gradient alignment (`gradient_alignment.png`);
- embedding update size and cumulative update path (`embedding_updates.png`);
- partial-model correction norms;
- effective rank of the learned corrections (`correction_effective_rank.png`);
- `results_summary.md` and `results.json`, including per-run and per-embedding-type aggregates with deterministic bootstrap 95% intervals when multiple seeds are available.

## Gradient measurement

The gradient decomposition is done without changing the training objective. For a logged batch, the code evaluates the same logits twice:

1. The output weight is detached, so the gradient reaching the input-side embedding parameters comes through the transformer.
2. The hidden state is detached, so the gradient reaching the output-side parameters comes directly from next-token prediction.

This makes the input/output pressure comparable even when the two roles share a parameter. The ordinary combined gradient is still used for the optimizer update.

For the partial model, `shared_input_grad_norm` and `shared_output_grad_norm` isolate the two pressures on the shared matrix. `input_correction_norm` and `output_correction_norm` measure the effective low-rank corrections, while `input_correction_parameter_count` and `output_correction_parameter_count` report their trainable parameter cost.

## Fairness and limitations

All three models use the same GPT-2 BPE tokenizer, WikiText-2 split, sampling procedure, context length, transformer size, initialization seed, optimizer, cosine schedule, and number of training tokens. The partial model is initialized with zero effective corrections, so it starts from the same shared embedding as the tied model.

The first run is a signal check, not a definitive claim. The default single seed and short runs should be followed by longer training, multiple seeds, larger models, and downstream representation tests if the loss/parameter curve is promising.

## Development checks

```bash
python3 -m py_compile config.py data.py model.py train.py evaluate.py analyze.py sweep.py adapter_sweep.py embedding_eval.py validate_study.py test_model.py test_study.py test_sweep.py test_adapter_sweep.py test_embedding_eval.py
python3 -m pytest -q
```
