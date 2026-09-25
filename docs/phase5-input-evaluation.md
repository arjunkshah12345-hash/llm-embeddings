# Phase 5 — input-representation evaluation

`embedding_eval.py` provides a deterministic intrinsic evaluation of the effective input matrix. The completed 50k primary and robustness aggregates are under each study's `embedding_eval/` directory. These measurements answer whether role-specific input corrections change token representation structure independently of language-model validation loss.

Run it on a saved checkpoint:

```bash
python3 embedding_eval.py \
  --run_dir runs/partial \
  --checkpoint best.pt \
  --max_tokens 512 \
  --neighbors 5
```

Once a pair set has been predeclared, add it as JSONL. Each row must contain `left` and `right` strings that each encode to one GPT-2 token, or explicit `left_token_id` and `right_token_id` integers; an optional `relation` groups the report:

```json
{"left_token_id": 123, "right_token_id": 456, "relation": "morphological"}
```

```bash
python3 embedding_eval.py --run_dir runs/partial --pairs research/pairs.jsonl
```

The output includes:

- a fixed nearest-centroid probe for four frequency quantiles, reporting accuracy, macro accuracy, and per-class accuracy;
- a fixed nearest-centroid probe for coarse token-shape classes (whitespace, punctuation, short, and longer tokens), with macro accuracy to expose class imbalance;
- frequency-bucket agreement and mean cosine similarity among deterministic nearest neighbors sampled across frequency buckets.
- optional cosine scores for every predeclared pair and mean cosine by relation.

The probe split is token-ID modulo five, with one remainder held out. This is an intrinsic diagnostic, not a semantic benchmark. Positive claims about meaning require a predeclared pair set or downstream task and should be reported separately from these diagnostics.
