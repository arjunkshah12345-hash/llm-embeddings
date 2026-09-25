# Research plan and final interpretation

## Question

Does a decoder-only language model retain a useful quality benefit from separate input and output embeddings when those matrices share a large base and receive small, role-specific low-rank corrections?

The fixed comparison is:

1. `tied`: one matrix for token lookup and output logits;
2. `untied`: independent input and output matrices initialized from the same values;
3. `partial`: one shared matrix plus independent low-rank input and output corrections.

No outcome was assumed in advance. The release reports both matched Transformer size and parameter-efficiency views.

## Completed studies

### Study 1: 10k primary endpoint

The preserved three-seed WikiText-2 study used the full six-condition matrix: tied, untied, rank-8 partial, parameter-matched capacity control, input-only partial, and output-only partial. All fairness checks passed. This remains a preliminary horizon and is retained as historical context rather than combined with the 50k primary result.

### 50k primary comparison

Fresh-from-step-zero 50k runs used seeds 1337, 2027, and 31415 with tied, untied, rank-8 partial, and capacity control. The exact cosine schedule was fixed at the 50k horizon. All variants saw exactly the same sampled token streams within each seed, and the final validation record was present at the declared final step.

The result is negative for the original performance hypothesis: tied is best, partial is worse than tied by a paired mean final-loss difference of 0.03928, and untied is worse by 0.35877. A recovery fraction is undefined because the untied denominator is not a positive quality benefit.

### Rank and budget studies

Ranks 1, 2, 4, 8, 16, and 32 were replicated across the same three seeds under a frozen 10k protocol. No rank consistently improves tied. Rank 32 is closest in mean final loss, but its uncertainty interval includes no reliable difference. The parameter-matched tied control in the 50k study also underperforms tied, while remaining worse than partial, so the result is not explained by a simple parameter-count advantage.

### Robustness

The central comparison was repeated for a smaller WikiText-2 Transformer and for Tiny Shakespeare, each with three seeds. Neither replication gives a reliable partial-tying improvement over tied. The Tiny Shakespeare capacity control shows late overfitting, so final-checkpoint results are reported alongside best-checkpoint values where relevant.

## Mechanism and representation measurements

The code measures input and output pressure in effective vocabulary-by-width matrix space, avoiding arbitrary low-rank factor-basis effects. It records gradient norms and ratios, effective gradient cosines, cumulative embedding displacement, correction norms, effective rank, singular values, shared-matrix alignment, token classes, matched-frequency buckets, and one-sided path interventions. Input and output frozen representation probes remain separate from language-modeling loss.

The 50k primary mechanism result does not show persistent output-gradient dominance: the output/input ratio is near one and effective input/output gradient cosine is near zero at the final measurement. Partial corrections become distinct, especially on the output side, but their weak alignment and nonzero movement do not yield a validation-loss gain. In the three-seed 10k path interventions, stopping input-path gradients changed partial minus tied final loss by `+0.00037` (95% interval `[-0.00676, +0.00798]`), while stopping output-path gradients changed it by `-0.00618` (interval `[-0.01474, +0.01011]`); neither intervention has a stable paired effect. The path-intervention artifacts are reported as mechanism evidence and are not used to retrofit the performance claim.

## Final conclusion

Under the tested small-model, WikiText-2/Tiny Shakespeare, 10k/50k, three-seed protocols:

- full untying is not beneficial and is strongly worse in the 50k primary study;
- low-rank partial tying is much cheaper than untying but does not recover a quality advantage over tied;
- rank choice does not reveal a stable improvement;
- role-specific corrections and role-level gradient measurements show measurable specialization without a corresponding language-modeling benefit.

The resulting paper is a negative/mechanistic study of when tied embeddings are sufficient, not a claim that partial tying is a generally superior architecture.

## Reproducibility gate

The code, pinned data sources, frozen environment, Kaggle launch/collection scripts, exact seed manifests, fairness outputs, raw compact metrics, aggregate JSON, plots, generated tables, and independently compilable arXiv bundle are part of the release. Large checkpoints and datasets remain excluded. Headline numbers are regenerated from machine-readable artifacts and checked against per-seed outputs before release.
