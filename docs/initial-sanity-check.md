# Historical initial sanity check (pre-cleanup)

This historical run used the pre-cleanup dataset source. It used six optimizer steps, one sample per batch, context length 64, two validation batches per checkpoint, seed 1337, and the same 384-wide, six-layer, six-head transformer for all variants. It ran on Apple MPS and is retained as an instrumentation record only.

| Variant | Total parameters | Embedding parameters | Extra vs tied | Best logged validation loss | Best logged perplexity |
|---|---:|---:|---:|---:|---:|
| tied | 29,950,080 | 19,298,688 | 0 | 10.2695 | 28,840.35 |
| partial, rank 8 | 30,760,336 | 20,108,944 | 810,256 | 10.2658 | 28,732.49 |
| untied | 49,248,768 | 38,597,376 | 19,298,688 | 10.2667 | 28,759.05 |

The measured output-to-input embedding gradient ratio increased through the short run and finished at approximately 8.71 for tied, 8.73 for partial, and 8.86 for untied. Partial corrections became nonzero and finished at about 0.18% of the shared matrix norm on the input side and 0.21% on the output side.

These numbers validate the training, checkpoint, analysis, and gradient instrumentation. They are not a performance conclusion: six steps are far too short, the validation sample is small, and the differences are within the range where seeds and longer training can change the ranking.

Recreate the run with:

```bash
python3 train.py --embedding_type tied --output_dir sanity_runs2 --steps 6 --batch_size 1 --block_size 64 --eval_interval 2 --eval_batches 2 --log_interval 1 --save_interval 6 --warmup_steps 2
python3 train.py --embedding_type untied --output_dir sanity_runs2 --steps 6 --batch_size 1 --block_size 64 --eval_interval 2 --eval_batches 2 --log_interval 1 --save_interval 6 --warmup_steps 2
python3 train.py --embedding_type partial --output_dir sanity_runs2 --steps 6 --batch_size 1 --block_size 64 --eval_interval 2 --eval_batches 2 --log_interval 1 --save_interval 6 --warmup_steps 2
python3 analyze.py --runs_dir sanity_runs2 --output_dir sanity_analysis
```
