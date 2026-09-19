# Exploratory representation pilot

This report applies `embedding_eval.py` to the existing three-model pilot. It is an instrumentation check, not a representation claim: each checkpoint was trained for only 30 steps with seed 1337, batch size 1, and context length 64.

| Variant | Frequency probe accuracy | Frequency majority baseline | Neighbor bucket agreement | Mean neighbor cosine | Shape probe accuracy |
|---|---:|---:|---:|---:|---:|
| tied | 0.2482 | 0.2583 | 0.2609 | 0.2205 | 0.6210 |
| partial | 0.2479 | 0.2583 | 0.2613 | 0.2206 | 0.6210 |
| untied | 0.2456 | 0.2583 | 0.2527 | 0.1912 | 0.2510 |

The frequency probe is at or below its majority baseline, and nearest-neighbor frequency agreement is close to the chance level implied by the bucket distribution. The shape probe is not interpretable as a positive result because its majority baseline is 0.9658; the token-shape labels are highly imbalanced and need a balanced or macro-averaged evaluation before use.

Reproduce the measurements after running the pilot:

```bash
for variant in tied partial untied; do
  python3 embedding_eval.py \
    --run_dir "runs/phase2_pilot_20260919/seed1337_${variant}" \
    --checkpoint best.pt \
    --max_tokens 512 \
    --neighbors 5
done
```

The next representation phase should use longer checkpoints, frequency-stratified splits, macro-averaged probes, and a predeclared token-pair or downstream task set.
