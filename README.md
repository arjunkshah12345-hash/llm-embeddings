# Partially tied language-model embeddings

> **Research status:** Experimental. The implementation and measurement pipeline are complete, and the first three-seed 10k-step primary study is available as preliminary evidence. Fresh 50k training, rank replication, intervention, scale, and dataset studies are still required before making a final performance claim.

This project studies a parameter-sharing choice in decoder-only language models. A language model uses token vectors at the input and a vocabulary projection at the output, but those roles need not require exactly the same representation.

The central question is whether a model can retain most of the parameter savings of weight tying while giving the two roles a small amount of independent capacity:

\[
W_{\text{input}} = W_{\text{shared}} + \Delta_{\text{input}}, \qquad
W_{\text{output}} = W_{\text{shared}} + \Delta_{\text{output}}.
\]

Here each correction is a learned low-rank factorization, so the experiment compares:

- **Tied:** one matrix is used for both token lookup and output logits.
- **Untied:** input and output matrices are separate.
- **Partial:** one shared matrix plus independent low-rank input and output corrections.

Rather than choosing between fully tied and fully untied embeddings, this project studies whether a large shared base plus small role-specific low-rank residuals can recover much of the flexibility of untying at a fraction of the parameter cost. The result is open: the repository is designed to measure the tradeoff fairly, including cases where partial tying does not help.

## Quick start

```bash
python3 -m pip install -r requirements.txt
python3 sweep.py \
  --output_dir runs/smoke \
  --dataset wikitext2 --device cpu --seeds 1337 \
  --steps 8 --batch_size 1 --block_size 32 \
  --n_layer 1 --n_head 1 --n_embd 16 \
  --adapter_rank 2 --adapter_alpha 2 \
  --warmup_steps 2 --eval_interval 4 --eval_batches 2 \
  --log_interval 2 --save_interval 8 --no-save_optimizer
```

The command trains all three variants with one shared configuration, validates the comparison, and writes compact checkpoints, JSONL metrics, and plots under `runs/smoke/`. It downloads the pinned WikiText-2-raw-v1 source and the GPT-2 BPE vocabulary on first use. The full reproduction workflow is in [REPRODUCIBILITY.md](REPRODUCIBILITY.md); the frozen primary study is specified in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), with Kaggle launch and collection tooling under [`kaggle/`](kaggle/).

## Model and experiment

The default model is a small GPT-style causal Transformer: six layers, width 384, six attention heads, and a GPT-2 vocabulary, about 30M parameters when tied. Partial tying uses rank-8 corrections by default. For vocabulary size \(V\), hidden width \(D\), and adapter rank \(r\), its extra stored parameters over tied are \(2r(V+D)\).

The partial implementation uses the factorized operations directly during training:

```text
input:  shared[token_ids] + scale * input_A[token_ids] @ input_B.T
output: hidden @ shared.T + scale * (hidden @ output_B) @ output_A.T
```

The full correction matrices are reconstructed only for analysis metrics and explicit evaluation. At zero effective correction, partial tying is functionally identical to tied. Untied input and output matrices start from the same initialized values, while the Transformer, optimizer, schedule, seed, tokenizer, data exposure, and validation windows remain shared across variants.

The default dataset is the pinned WikiText-2-raw-v1 corpus tokenized with GPT-2 BPE. Tiny Shakespeare is also available with a pinned source commit and deterministic, disjoint 90/5/5 character splits. Every run records dataset source metadata and hashes, configuration, package versions, git commit, parameter counts, throughput, memory, approximate FLOPs, losses, and embedding-gradient measurements.

## Measurements

The training metrics include:

- train and validation loss/perplexity;
- total, Transformer, shared, correction, and embedding parameter counts;
- training speed, wall time, peak allocated GPU memory, and approximate FLOPs;
- input-side and output-side effective embedding-matrix gradient norms, cosine alignment, and their ratio;
- token-class gradient means for whitespace, punctuation, common, rare, and other tokens;
- partial-correction norms, relative norms, effective ranks, top singular values, and alignment with the shared matrix;
- controlled ablations that stop input or output gradients for selected step intervals.

`validate_study.py` is a hard comparison gate. It requires the complete seed-by-variant matrix, matching configurations and pinned dataset metadata, finite metrics, an observed validation record at exactly the declared final step, and exactly equal token exposure. `analyze.py` produces loss, parameter-efficiency, FLOP, gradient, update, token-class, and correction plots plus paired seed-level summaries.

The gradient decomposition evaluates two counterfactual losses: one detaches the output weights to measure pressure arriving through the input path, and the other detaches the hidden states to measure direct output-prediction pressure. The role-level norms and cosine are computed in effective vocabulary-by-width matrix space, so they do not depend on arbitrary low-rank factor rotations. The `shared_*` fields isolate the two pressures on the shared matrix. The normal training path remains factorized; full correction matrices are used only for analysis.

## Current evidence

The first frozen primary study used WikiText-2-raw-v1, GPT-2 BPE, the 6-layer/384-width Transformer, six conditions, and seeds 1337, 2027, and 31415 for 10,000 steps. The fairness validator passed for every seed. Mean final validation loss was 5.3741 for tied, 5.3774 for rank-8 partial, and 5.4388 for untied; partial added 810,256 parameters over tied, while untied added 19,298,688. In this preliminary horizon, untied was worse than tied, so no recovery fraction is reported. The partial-versus-tied paired difference was uncertain across the three seeds. These values are a reproducible pilot endpoint, not a claim that partial tying works; fresh 50k runs and robustness studies are in progress. The machine-readable aggregate is [results/primary/primary_aggregate.json](results/primary/primary_aggregate.json).

The tracked 30M-class pilot used one seed and five optimizer steps against the pre-cleanup dataset source. It verifies that the three variants train, produce finite metrics, expose equal token counts, and generate the analysis artifacts. It is explicitly an instrumentation check and is documented in [docs/initial-mechanism-smoke.md](docs/initial-mechanism-smoke.md); it is not evidence that partial tying improves language modeling or is comparable with the pinned raw-source study.

A provisional manuscript draft is in [`paper/main.tex`](paper/main.tex), with generated tables and references under [`paper/`](paper/). It is intentionally labeled preliminary and will be regenerated from the final cloud aggregates.

Substantive final validation still requires longer equal-token runs, multi-seed rank comparisons, intervention studies, and tests at additional sizes and datasets. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

## Related work

Weight tying was introduced as a practical and theoretical parameter-sharing method for language models by Press and Wolf and by Inan, Khosravi, and Socher. Press and Wolf also compared the input and output roles and reported that the tied matrix evolves more like the output embedding in their settings: [Press & Wolf, EACL 2017](https://aclanthology.org/E17-2025/) and [Inan et al., ICLR 2017](https://arxiv.org/abs/1611.01462).

Several papers study ways to relax or reinterpret the equality constraint. Gulordava, Aina, and Boleda decouple the hidden state from word-embedding prediction while retaining a compact architecture: [EMNLP 2018](https://aclanthology.org/D18-1323/). Pappas, Miculicich, and Henderson propose a structure-aware output layer that generalizes hard tying in neural machine translation: [WMT 2018](https://aclanthology.org/W18-6308/). Chung et al. study decoupled input and output embedding dimensions and show that extra output capacity can matter for pretrained representations: [ICLR 2021](https://openreview.net/forum?id=xpFFI_NtgpW). Derby, Miller, and Devereux analyze differences between input and output representations in neural language models: [CoNLL 2020](https://aclanthology.org/2020.conll-1.36/).

Adaptive input representations use variable-capacity factorization for vocabulary efficiency and can be paired with an adaptive softmax: [Baevski and Auli, ICLR 2019](https://openreview.net/forum?id=ByxZX20qFQ). That is related parameterization work, but it does not test role-specific residuals on a shared full-width base.

More recent work makes the role distinction especially relevant. Bertolotti and Cazzola connect tying to the distributional hypothesis and distinguish semantic input structure from contextual output structure: [ICML 2024](https://proceedings.mlr.press/v235/bertolotti24a.html). Lopardo et al. report evidence that tied embeddings can be biased toward the output space and link that bias to output-gradient dominance: [Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.2027/). Other recent alternatives take different routes, including a compact learned input representation with an untied output head in [Leviathan (Batley & Saha, 2026)](https://arxiv.org/abs/2601.22040) and a pseudo-inverse-consistent shared interface in [Pseudo-Inverse Tying (Gu et al., 2026)](https://arxiv.org/abs/2602.04556).

This project is related to those efforts but does not claim a new general solution from its current pilots. Its specific controlled comparison is a shared vocabulary-sized base with separate low-rank residuals for the input and output paths, evaluated against tied and fully untied models with the same Transformer and training exposure.

## Limitations and roadmap

The current evidence is still small: it uses GPT-2 BPE, WikiText-2, one 10k endpoint, one primary adapter family, and limited representation probes. The FLOP numbers are estimates based on dense-matmul conventions and factorized projection costs, not hardware-independent measurements. A positive result would require replication across training lengths, model sizes, datasets, and evaluation types. A null result is also useful because it would bound the value of role-specific corrections.

The roadmap covers baseline scaling, adapter-budget comparisons, mechanism checks, input-representation evaluation, robustness, and reproducible release artifacts. It is maintained in [RESEARCH_PLAN.md](RESEARCH_PLAN.md), with detailed phase notes under [`docs/`](docs/).

## License

Released under the [MIT License](LICENSE).
