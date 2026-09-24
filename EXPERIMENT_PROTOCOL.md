# Frozen primary experiment protocol

**Protocol status:** frozen before substantive results. The primary endpoint and
condition list must not be changed in response to intermediate validation loss.
Later work is labelled secondary or exploratory.

## Question

Does a shared vocabulary embedding with independent low-rank input and output
corrections recover part of the quality benefit of fully untied embeddings at a
small fraction of the additional parameter cost?

The effective matrices are:

```text
W_input  = W_shared + (alpha / rank) A_input B_input^T
W_output = W_shared + (alpha / rank) A_output B_output^T
```

## Primary comparison

The primary study contains six conditions, all run by the same sweep:

| Condition | Definition |
|---|---|
| `tied` | One shared input/output matrix |
| `untied` | Separate input and output matrices initialized identically |
| `partial` | Shared matrix plus independent rank-8 input/output corrections |
| `capacity_control` | Tied embeddings plus a parameter-matched residual capacity adapter |
| `partial_input` | Shared matrix plus input correction only |
| `partial_output` | Shared matrix plus output correction only |

`capacity_control` is included in the primary matrix. It tests whether a partial
model wins because capacity is placed on the two embedding roles rather than
because it simply has more trainable parameters.

## Fixed configuration

- Dataset: WikiText-2-raw-v1, using the SHA-256-pinned files in `data.py`.
- Tokenizer: GPT-2 BPE from `tiktoken`.
- Seeds: `1337`, `2027`, `31415`.
- Horizon: 10,000 optimizer steps.
- Transformer: 6 layers, 6 heads, width 384, context 256.
- Batch size: 2; gradient accumulation: 1.
- Adapter rank: 8; adapter alpha: 8.
- Optimizer: AdamW; learning rate `3e-4` to `3e-5`; warmup 100 steps; weight decay 0.1; gradient clipping 1.0.
- Validation: 20 deterministic validation batches every 200 steps, with a required record at step 9,999.
- Logging: training loss and diagnostics every 50 steps; compact checkpoints every validation interval; optimizer checkpoints disabled for the primary run because the horizon is fixed.
- Device: Kaggle GPU. No model training is performed locally.

The six conditions use the same sampled windows for each seed. The run manifest
records the actual sampled-offset SHA-256 digest and the validator compares it
across conditions and against the deterministic expected stream.

## Primary metrics and statistics

The primary endpoint is final validation loss at step 9,999. Best validation
loss, final perplexity, training loss, tokens/sec, wall time, peak memory,
parameter counts, and estimated FLOPs are secondary reported metrics.

For every condition and seed, retain the individual value, mean, sample standard
deviation, paired difference against `tied`, and deterministic percentile
bootstrap 95% interval over paired seeds. Report the untied improvement over
tied, partial improvement over tied, and partial-to-untied gap. Report a
recovered fraction only when the paired tied-vs-untied difference is
distinguishable from zero and its denominator is meaningfully nonzero.

No run is excluded for an unfavorable result. A run is invalid only when the
fairness validator fails, metrics are incomplete/non-finite, or the cloud job
does not produce the required artifact manifest; invalid runs are reported and
repaired or rerun rather than silently removed.

## Secondary and exploratory work

- A fresh 50,000-step study is secondary and is launched from step zero if the
  primary result is unresolved or warrants longer training. A 10k study is
  never resumed into a 50k schedule because the cosine learning-rate horizon
  would differ.
- Rank sweep: ranks 1, 2, 4, 8, 16, and 32, with the same fixed model and
  dataset. Seed count and horizon are recorded per sweep and are not used to
  rewrite the primary endpoint.
- Interventional path ablations, representation evaluations, scale replication,
  and second-dataset replication are secondary studies.

## Reproducibility gate

The primary study is analyzable only when every seed × condition cell exists,
all configs and dataset hashes match, final validation is at exactly step
9,999, token exposure is identical, actual sampled-offset digests agree, and
the generated analysis reads the raw JSONL outputs. The frozen protocol,
source commit, environment metadata, Kaggle hardware metadata, and commands
are retained with the cloud artifacts.
