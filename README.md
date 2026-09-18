# Partially tied language-model embeddings

This repository is a small, reproducible experiment for comparing three GPT-style decoder-only language models:

- `tied`: one matrix is used for input embeddings and output logits.
- `untied`: separate input and output matrices.
- `partial`: one shared matrix plus independent low-rank corrections for the input and output roles.

The experiment keeps the transformer, tokenizer, dataset, optimizer, schedule, batch size, seed, and token budget the same across runs. The architectural difference is the embedding module.

The long-term research roadmap is in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). The initial six-step sanity check is documented in [docs/initial-sanity-check.md](docs/initial-sanity-check.md).

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

For a multi-seed study, use `sweep.py`. It forwards one shared configuration to every model/seed combination, refuses to overwrite completed runs unless asked, writes `study_manifest.json`, and analyzes the full study at the end:

```bash
python3 sweep.py --output_dir runs/study-001 --steps 2000 --seeds 1337 2027 31415
```

Evaluate a saved checkpoint directly:

```bash
python3 evaluate.py --run_dir runs/partial --checkpoint best.pt
```

## Outputs

Each run directory contains:

- `config.json`: exact model, data, optimizer, and seed settings;
- `manifest.json`: git commit, command, runtime, dataset hashes, and parameter manifest;
- `parameter_counts.json`: total, transformer, and embedding parameter counts;
- `metrics.jsonl`: training/validation losses, best/final perplexity, training-only throughput, wall-clock time, memory, gradient decomposition, and adapter norms;
- `last.pt` and `best.pt`: CPU model checkpoints. The first version intentionally omits AdamW state to keep local artifacts compact; checkpoints are for evaluation and comparison rather than exact mid-run resume.

Validation uses deterministic fixed token windows, so all model variants and repeated evaluations see the same validation examples.

`analyze.py` creates:

- training and validation loss plots;
- parameter count versus validation loss;
- input/output embedding gradient norms and their output-to-input ratio (`gradient_norms_and_ratio.png`);
- partial-model correction norms;
- `results_summary.md` and `results.json`, including per-run and per-embedding-type aggregates.

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
python3 -m py_compile config.py data.py model.py train.py evaluate.py analyze.py sweep.py test_model.py
python3 -m pytest -q
```
