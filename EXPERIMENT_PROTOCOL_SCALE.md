# Study 3: Scale and broad-corpus replication protocol

This protocol is frozen before inspecting any Study 3 result. It is a
predeclared generalization study of the completed small-model experiments.
The existing 10k and 50k studies, their manifests, and their paper inputs are
preserved as separate studies.

## Question and endpoints

The primary question is whether the small-model conclusion survives at a
substantially larger decoder-only Transformer scale, after training on broad
text, and on zero-shot language-model benchmarks:

> Does a shared embedding base with small role-specific low-rank corrections
> produce a meaningful quality or capability benefit over exact tying, and does
> it approach a fully untied model at much lower parameter cost?

The primary language-model endpoint is final validation cross-entropy at the
declared final optimizer step. Best validation loss is reported as a secondary
training diagnostic. The primary downstream endpoint is the complete frozen
core benchmark table; individual task metrics remain the estimands and are not
collapsed into an unpredeclared intelligence score.

## Model and embedding conditions

All conditions use the same decoder-only Transformer:

| Setting | Value |
| --- | --- |
| Layers | 12 |
| Attention heads | 12 |
| Width | 768 |
| Head dimension | 64 |
| Context length | 512 |
| Tokenizer | GPT-2 BPE (`tiktoken` `gpt2`) |
| Dropout | 0 |
| Vocabulary | 50,257 |
| Partial rank | 8 |
| Partial alpha | 8 |
| Partial scale | `s = alpha / rank = 1` |

The four predeclared conditions are `tied`, `partial`, `untied`, and
`capacity_control`. Partial tying uses

\[
W_{input}=W_{shared}+sA_iB_i^T, \qquad
W_{output}=W_{shared}+sA_oB_o^T.
\]

The untied model starts with an output copy of the tied input initialization.
Partial corrections start with zero effective matrices. The capacity control
is a tied model with a zero-initialized, bias-free residual bottleneck MLP on
the final hidden state. Its width is chosen by the existing parameter-matching
rule, which gives approximately the same added parameter count as rank-8
partial tying without role-specific embedding parameters.

## Data

Training uses the `HuggingFaceFW/fineweb-edu` `sample-10BT` configuration at
revision `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`. The repository records the
source shard manifest and file object IDs in `data.py`. Kaggle streams this
immutable revision once per job and materializes a compact deterministic token
window locally in the job workspace. No FineWeb data is committed to Git.

The study consumes exactly 20,480,000 training tokens, 262,144 validation
tokens, and 262,144 test tokens from the source stream. The consumed training
window is far smaller than the declared 10B-token sample and is not cycled.
GPT-2 BPE tokenization is applied to the dataset `text` field. The validation
and test token arrays are disjoint contiguous windows following the training
window. Validation batches are deterministic and identical across conditions.

## Training schedule

Every condition within a seed receives the same sampled token windows, data
arrays, initialization seed, optimizer, schedule, and endpoint:

| Setting | Value |
| --- | --- |
| Seeds | 1337, 2027, 31415 |
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Minimum learning rate | 3e-5 |
| Warmup | 500 optimizer steps |
| Schedule | cosine decay through step 19,999 |
| Weight decay | 0.1 |
| Gradient clipping | global norm 1.0 |
| Micro-batch | 1 sequence |
| Gradient accumulation | 2 micro-batches |
| Optimizer steps | 20,000 |
| Tokens per step | 1,024 |
| Total training tokens | 20,480,000 |
| Validation interval | every 500 steps and step 19,999 |
| Validation batches | 32 fixed batches |
| Primary checkpoint | final checkpoint at step 19,999 |

Training starts from step zero. No Study 1 or Study 2 checkpoint is resumed.
Each condition/seed is an independent Kaggle job, while the deterministic
materialized token window and seed pair the data stream across jobs. The
fairness validator fails on any mismatch in dataset hashes, token-stream
digest, exposure, configuration, or final validation step.

## Benchmarks

The benchmark harness is EleutherAI `lm-evaluation-harness` v0.4.13 at commit
`ddd67220430a2470529f25fd5c05a576ca1057a0`. The model is evaluated as a base
language model with no fine-tuning, instruction tuning, chat template, or
benchmark-specific training. The benchmark task list is frozen before result
inspection.

The core suite is:

`lambada_open`, `hellaswag`, `piqa`, `winogrande`, `arc_easy`,
`arc_challenge`, `sciq`, `openbookqa`, `boolq`, `commonsense_qa`.

The extended suite is:

`mmlu`, `truthfulqa_mc1`, `triviaqa`, `gsm8k`.

The optional appendix suite is `mmlu_pro`, `bbh`, `gpqa`, `musr`, and
`math`/`math_algebra` tasks supported by the pinned harness. Unsupported or
resource-prohibitive tasks are recorded as such; they are never silently
dropped. All evaluated checkpoints use zero-shot settings (`num_fewshot=0`),
the same precision and batch size, and the final checkpoint. LAMBADA is
reported separately with its exact harness metric name and per-seed values.

## Measurements and reporting

Each run records final and best validation loss/perplexity, training loss,
parameters, embedding parameters, extra parameters versus tied, token count,
estimated FLOPs, wall time, tokens/second, peak GPU memory, gradient norms,
effective gradient cosine, output/input ratio, correction norms/effective
rank/singular values, and the input/output correction alignment where the
condition supports it. The scale mechanism interval is intentionally less
frequent than the small-model interval to keep effective-matrix diagnostics
computationally bounded.

The analysis reports each seed, mean, standard deviation, paired differences
versus tied, and descriptive paired bootstrap intervals. With three seeds these
intervals describe the observed matched runs; they are not large-sample
inferential claims. No benchmark percentages are averaged into a single score.

The scale comparison is a replication across the existing approximately 30M
study and this approximately 124M tied-model study. It is not presented as a
scaling law. The paper will follow the observed result, including a null,
negative, or mixed outcome.

