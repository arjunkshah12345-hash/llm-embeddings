# Phase 5 — input-representation evaluation

`embedding_eval.py` provides the first deterministic intrinsic evaluation of the effective input matrix. It is designed to answer whether role-specific input corrections change measurable token representation structure, independently of language-model validation loss.

Run it on a saved checkpoint:

```bash
python3 embedding_eval.py \
  --run_dir runs/partial \
  --checkpoint best.pt \
  --max_tokens 512 \
  --neighbors 5
```

The output includes:

- a fixed nearest-centroid probe for four frequency quantiles, reporting accuracy, macro accuracy, and per-class accuracy;
- a fixed nearest-centroid probe for coarse token-shape classes (whitespace, punctuation, short, and longer tokens), with macro accuracy to expose class imbalance;
- frequency-bucket agreement and mean cosine similarity among deterministic nearest neighbors sampled across frequency buckets.

The probe split is token-ID modulo five, with one remainder held out. This is an intrinsic diagnostic, not a semantic benchmark. Positive claims about meaning require a predeclared pair set or downstream task and should be reported separately from these diagnostics.
