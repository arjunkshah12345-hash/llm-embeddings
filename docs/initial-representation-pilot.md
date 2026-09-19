# Exploratory representation pilot

This report applies `embedding_eval.py` to the existing three-model pilot. It is an instrumentation check, not a representation claim: each checkpoint was trained for only 30 steps with seed 1337, batch size 1, and context length 64.

| Variant | Frequency accuracy | Frequency macro accuracy | Frequency majority baseline | Neighbor bucket agreement | Mean neighbor cosine | Shape accuracy | Shape macro accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| tied | 0.2482 | 0.2483 | 0.2583 | 0.2609 | 0.2205 | 0.6210 | 0.4736 |
| partial | 0.2479 | 0.2479 | 0.2583 | 0.2613 | 0.2206 | 0.6210 | 0.4736 |
| untied | 0.2456 | 0.2457 | 0.2583 | 0.2527 | 0.1912 | 0.2510 | 0.2505 |

The frequency probe is at or below its majority baseline, and nearest-neighbor frequency agreement is close to the chance level implied by the bucket distribution. Macro accuracy makes the shape result visible despite its 0.9658 majority baseline; the tied and partial checkpoints are identical at this very short horizon, while the untied result is near chance. None of this is a semantic claim: the checkpoints are too short and the shape labels are only coarse diagnostics.

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
