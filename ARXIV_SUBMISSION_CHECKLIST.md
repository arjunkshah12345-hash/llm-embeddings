# arXiv submission checklist

This checklist is for the final research release. The current repository still
contains provisional results, so the paper must not be submitted until the
longer and secondary Kaggle studies are complete and the manuscript has been
regenerated from their machine-readable outputs.

## Submission metadata

- **Working title:** Between Tied and Untied: Low-Rank Role-Specific Embeddings for Language Models
- **Author:** Arjun Shah
- **Affiliation:** Use the author's truthful affiliation, or omit affiliation if none is being claimed.
- **Primary category:** `cs.LG`
- **Possible cross-list:** `cs.CL`, only if the final paper retains a substantial language-modeling or NLP contribution.
- **Peer-review status:** This is an open research preprint; do not imply peer review or acceptance.

## Abstract and claims

Regenerate the abstract after the final aggregates exist. Every number in it
must come from the checked-in machine-readable result artifacts. State whether
the evidence supports, weakens, or fails to support the partial-tying
hypothesis. Do not report a recovered fraction when the tied-versus-untied
denominator is not meaningful.

## Source bundle

- Exact upload directory: `arxiv/`
- Main source: `arxiv/main.tex`
- Bibliography: `arxiv/references.bib`
- Generated tables: `arxiv/generated/`
- Figures: `arxiv/figures/`
- No absolute local paths, checkpoints, datasets, credentials, or generated
  cache files may be included.

## Verification before upload

1. Complete the final Kaggle studies and collect their manifests, raw compact
   metrics, validation outputs, and dataset hashes.
2. Re-run the fairness validator for every final study.
3. Regenerate all tables, macros, figures, and the arXiv bundle from raw
   machine-readable artifacts.
4. Compile independently from the bundle:

   ```bash
   cd arxiv
   tectonic main.tex
   ```

5. Inspect every page of the resulting PDF for clipped text, unreadable
   legends, bad table breaks, missing references, and layout warnings.
6. Check every headline number against the raw JSON outputs and confirm that
   the repository commit, environment lock, seeds, training budgets, and
   dataset hashes are recorded.
7. Confirm the README, paper, abstract, and comments field make only claims
   supported by the completed studies.

## Comments field suggestion

Use a short factual description such as: “Experimental study of low-rank
role-specific corrections to tied input/output embeddings in small decoder-only
language models; code and reproducibility artifacts included.” Add page and
figure counts only after the final PDF is generated.

## Final decision gate

Do not label the project **READY FOR ARXIV** until the final multi-seed main
comparison, longer-training decision, rank/parameter-efficiency analysis, and
at least the selected robustness studies are complete. If the evidence is
negative or dataset-dependent, state that directly in the title, abstract, and
conclusion.
