# Historical 30M mechanism smoke (pre-cleanup)

This historical run used the pre-cleanup dataset source and is retained only as
an instrumentation record. It is not comparable to the pinned WikiText-2-raw-v1
study and is not evidence for a general performance claim.

## Reproduction

Run from the repository root:

```bash
python3 sweep.py \
  --output_dir runs/phase2_mechanism_smoke_20260919 \
  --dataset wikitext2 --data_dir data --device auto \
  --seeds 1337 --steps 5 --batch_size 1 --block_size 64 \
  --n_layer 6 --n_head 6 --n_embd 384 \
  --adapter_rank 8 --adapter_alpha 8 \
  --learning_rate 0.0003 --min_learning_rate 0.00003 \
  --warmup_steps 2 --eval_interval 5 --eval_batches 2 \
  --log_interval 5 --save_interval 5 --no-save_optimizer
```

The run used the GPT-2 tokenizer and the then-current WikiText-2 files whose
hashes are stored in `study_validation.json`. Optimizer checkpoints were disabled to keep
the short smoke run small; compact `best.pt` and `last.pt` checkpoints were
written for all three models.

## Observed output

| Model | Total parameters | Embedding parameters | Extra vs tied | Final validation loss | Validation perplexity |
|---|---:|---:|---:|---:|---:|
| tied | 29,950,080 | 19,298,688 | 0 | 10.2284 | 27,678.9 |
| partial, rank 8 | 30,760,336 | 20,108,944 | 810,256 | 10.2254 | 27,594.5 |
| untied | 49,248,768 | 38,597,376 | 19,298,688 | 10.2244 | 27,566.4 |

The validator reported identical 320-token exposure for all three runs and no
configuration, dataset-hash, or metric errors. The partial model was 0.0031
validation-loss units below tied and 0.0010 above untied. The output/input
embedding-gradient ratio at the logged training step was about 1.487 for all
three variants. These values come from one seed and five optimizer steps, so
they only confirm that the measurements and analysis pipeline work.

The raw run directory is ignored by git under `runs/`; its generated plots
include `gradient_norms_and_ratio.png`, `gradient_alignment.png`,
`token_type_gradients.png`, `embedding_updates.png`, and the partial correction
plots.
