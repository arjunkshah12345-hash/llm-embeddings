# Study 3 scaled follow-up status

**Status: exploratory and incomplete.**

The frozen scaled protocol targets a 12-layer, 12-head, width-768 decoder with GPT-2 BPE, block size 512, FineWeb-Edu sample-10BT token cache, rank 8 / alpha 8 partial corrections, 20,000 optimizer steps, and seeds 1337, 2027, and 31415. All model training ran in Kaggle.

Kaggle completed the four matched seed-1337 conditions below. Each run consumed 20,480,000 training tokens, passed the same token-stream digest, and produced complete core and extended lm-evaluation-harness outputs. The remaining seed submissions were rejected before launch with `Maximum weekly GPU quota of 30.00 hours reached.` These results therefore do not pass the repository's multi-seed final-study gate and must not be used as a confirmatory scale claim.

| Condition | Final validation loss | Perplexity | Parameters | Extra vs tied |
|---|---:|---:|---:|---:|
| tied | 5.005524 | 149.235 | 123,963,648 | 0 |
| partial | 4.989441 | 146.854 | 124,780,048 | 816,400 |
| untied | 5.022585 | 151.803 | 162,561,024 | 38,597,376 |
| Capacity control | 5.011638 | 150.151 | 124,780,800 | 817,152 |

Within this single seed, partial minus tied final loss was -0.016082; untied minus tied was +0.017061; capacity control minus tied was +0.006115. Those are descriptive observations, not evidence of a scaled-model conclusion.

The compact machine-readable artifacts are in [`results/scale3_seed1337/`](../results/scale3_seed1337/). The existing small-model paper and its conclusions remain unchanged. The scaled study should be resumed only after Kaggle quota is available, then run through the strict three-seed validator before updating the paper.

## Resume command

From a clean checkout with Kaggle credentials, launch the frozen matrix with:

```bash
python3 kaggle/orchestrate_scale.py --stage train --commit <new-study-commit> --seeds 1337 2027 31415 --conditions tied partial untied capacity_control
```

Do not mix these artifacts with the older scale attempts that used the pre-revision LAMBADA task name.
