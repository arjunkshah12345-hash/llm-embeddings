# Research plan

## Question

Does a decoder-only language model retain most of the validation-loss benefit of separate input and output embeddings when those matrices share a large base and receive small, role-specific corrections?

The primary comparison is:

1. `tied`: one matrix used for token lookup and output logits;
2. `untied`: separate input and output matrices;
3. `partial`: one shared matrix plus low-rank input and output corrections.

The central result should be reported both at a matched transformer size and as a parameter-efficiency curve. No outcome is assumed in advance.

## Phase 0 — Reproducible baseline

Status: implemented.

- GPT-style decoder-only transformer with configurable depth, width, heads, context length, and adapter rank.
- GPT-2 BPE tokenizer over WikiText-2 raw train/validation/test splits.
- Common seed, initialization, optimizer, schedule, context length, batch size, and token budget.
- JSONL metrics, CPU checkpoints, evaluation, plots, and a concise result summary.
- Input/output gradient decomposition using an input-path loss and an output-path loss.
- Low-rank partial adapters initialized to zero effective correction.

Exit condition: all three variants train, save checkpoints, report finite losses/perplexities, and produce non-empty gradient and correction metrics. This is satisfied by the local sanity run.

## Phase 1 — Experimental hardening

Goal: eliminate measurement ambiguity before spending compute.

Status: implemented.

- Add a manifest containing the exact dataset hashes, tokenizer version, package versions, model config, and git commit. (Implemented.)
- Add deterministic fixed validation windows alongside sampled validation batches. (Implemented; evaluation uses fixed windows.)
- Verify that the input/output gradient decomposition matches finite-difference checks on tiny models. (Implemented.)
- Add tests for tied parameter identity, untied initialization, partial zero initialization, checkpoint round trips, and token accounting. (Implemented.)
- Separate optimizer-state checkpoints (`optimizer_last.pt`) from compact evaluation checkpoints (`last.pt` / `best.pt`). Resume with `--resume path/to/optimizer_last.pt`.
- Record wall-clock time, peak allocated memory, FLOPs estimates (`estimated_flops_*` via the 6ND rule), and tokens per second in a common metrics schema.
- Add a sweep runner that records the exact shared configuration and run matrix. (Implemented.)
- Enforce the comparison gate before analysis: complete seed × embedding matrix, identical dataset hashes and shared settings, and token exposure within one percent. (Implemented in `validate_study.py` and called by `sweep.py`.)

Gate: no comparison is published unless all three runs use the same data manifest and the same token budget within one percent.

## Phase 2 — Baseline training study

Goal: determine whether the loss signal survives beyond the six-step smoke test.

Status: ready to run (see [docs/phase2-baseline.md](docs/phase2-baseline.md)).

- Run 10k–100k steps on WikiText-2 with at least three seeds.
- Keep the transformer configuration fixed across variants and repeat the analysis at two model sizes in the 20M–50M range.
- Report mean, standard deviation, best validation loss, final validation loss, perplexity, throughput, memory, FLOPs, and parameter count.
- Use confidence intervals or bootstrap intervals for differences between variants.
- Compare equal training tokens first; add compute-matched results as a separate analysis because untied output projections have different cost.

Gate: continue only if the partial model is consistently closer to untied than tied, or if the gradient/representation measurements show a clear independent signal.

## Phase 3 — Adapter design and budget sweep

Goal: map the quality/parameter tradeoff rather than overfit to rank 8.

Status: runner implemented in `adapter_sweep.py`; evidence pending the longer study.

- Sweep ranks 1, 2, 4, 8, 16, and 32.
- Sweep adapter scaling separately from rank.
- Compare low-rank additive corrections with a small number of alternatives: per-token diagonal gates, shared low-rank corrections with separate scalars, and a bottleneck residual adapter.
- Plot validation loss against extra parameters and against additional training FLOPs.
- Keep each adapter family initialized to the same effective function where possible.

Gate: select the simplest adapter family on the Pareto frontier and freeze it before downstream representation tests.

## Phase 4 — Understand the mechanism

Goal: test whether the shared matrix is actually receiving unequal role pressure and whether the correction directions explain the difference.

- Track input-side, output-side, and combined gradient norms for the shared matrix.
- Track update norms, cumulative parameter displacement, cosine similarity of input/output gradients, and the output-to-input ratio over training. (Gradient cosine and embedding update-path logging/plotting are implemented.)
- Measure correction norm, rank utilization, singular values, and alignment between corrections and the shared matrix.
- Compare token-frequency buckets and token types such as punctuation, whitespace, common words, and rare words.
- Run ablations that stop gradients through the input or output path for controlled intervals.

Gate: call the mechanism supported only if the pattern replicates across seeds and is not explained by token frequency, optimizer state, or a logging artifact.

## Phase 5 — Input-representation evaluation

Goal: test whether partial tying improves the usefulness of input embeddings independently of language-model loss.

Status: intrinsic frequency/shape probes implemented in `embedding_eval.py`; semantic and downstream evidence pending.

- Evaluate nearest-neighbor structure with frequency-matched token probes.
- Measure similarity on morphological, lexical, and semantic token-pair sets where appropriate.
- Train frozen-input linear probes for token metadata and contextual tasks.
- Compare input embedding quality at equal validation loss and equal parameter budget.
- Keep output-side evaluations separate so the two roles are not conflated.

Gate: require a predeclared evaluation set and report all metrics, including negative or null results.

## Phase 6 — Scale and robustness

Goal: establish whether the effect is real beyond one dataset and small model.

- Repeat on WikiText-103, TinyStories, and one larger public corpus with documented licensing and preprocessing.
- Test model sizes below 20M, 20M–50M, and above 100M where hardware allows.
- Use at least five seeds for the final comparison and report failed or interrupted runs.
- Check sensitivity to tokenizer vocabulary size, context length, normalization, dropout, optimizer, and learning-rate schedule.
- Add compute-matched and memory-matched comparisons.

Gate: make a positive claim only when the direction and practical effect size are stable across datasets, sizes, and seeds.

## Phase 7 — Reproducible release

Goal: make the result independently checkable.

- Freeze code and configs by git commit.
- Publish dataset download instructions, hashes, environment lock information, run manifests, raw JSONL metrics, plots, and analysis scripts.
- Add a one-command reproduction path for the smallest result and documented commands for the larger study.
- Write a methods report that separates exploratory analyses from preregistered comparisons.
- Preserve negative results and explain all deviations from the plan.

## Reporting rules

- Report validation loss and perplexity, not only the best checkpoint.
- Report total parameters, embedding parameters, extra parameters over tied, and transformer parameters separately.
- Report matched-transformer and parameter-efficiency views together.
- Treat a six-step sanity run as an instrumentation check, never as evidence of a general effect.
- Do not tune one variant using validation results and then compare it with untuned variants.
- Record seeds, data hashes, package versions, and exact commands for every result.
