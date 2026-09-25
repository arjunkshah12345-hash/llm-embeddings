# Phase 4 — mechanism checks

These checks ask whether input and output roles push the shared matrix differently, and whether that push is just a token-frequency artifact. The completed 50k primary and three-seed 10k path-intervention artifacts are under `results/long_50k/mechanism/`, `results/mechanism_stop_input/`, `results/mechanism_stop_output/`, and the two intervention JSON files in `results/`. They support a mechanistic analysis but do not turn gradient differences into a performance claim.

## Token classes

On each logged training step the trainer records the mean embedding-row gradient for tokens present in that batch. The input-side value is grouped by input tokens; the output-side value is grouped by target tokens:

- whitespace and punctuation, from the decoded surface form;
- common: the 1,000 highest-count remaining training tokens;
- rare: remaining tokens seen at most once;
- other: everything left.

Keys look like `output_token_grad_rare_mean`. `analyze.py` plots the output-side means in `token_type_gradients.png` when those keys exist.

The logged side-gradient norms use two counterfactual losses: the input-side loss detaches output weights, and the output-side loss detaches the hidden states. `input_grad_norm`, `output_grad_norm`, and `input_output_grad_cosine` are computed in effective vocabulary-by-width embedding-matrix space. This avoids comparing arbitrary low-rank factor coordinates, which can change under rotations that leave the represented correction unchanged. The `shared_*` metrics isolate the two pressures on the shared matrix. These diagnostics are measurement-only and do not alter the optimizer update; the default study uses zero dropout so the diagnostic replay is deterministic.

## Path ablations

Stop one role's gradient for a step range without changing the loss:

```bash
python3 train.py --embedding_type tied --path_ablation stop_input --ablation_start 0 --ablation_end 200 --steps 400 --run_name tied_stop_input
python3 train.py --embedding_type tied --path_ablation stop_output --ablation_start 0 --ablation_end 200 --steps 400 --run_name tied_stop_output
```

`stop_input` detaches the token lookup, so input embedding parameters get no gradient on those steps. `stop_output` detaches the output matrix, so output embedding parameters get no gradient while the input path still trains. Outside `[start, end)` training is normal. `ablation_end 0` means through the last step.

Keep these runs out of the tied/partial/untied comparison. They are ablations, not the primary experiment.

When a path ablation is active, the ordinary gradient diagnostics still describe the normal two-path decomposition for the logged batch. The `path_ablation` field identifies the actual training path used for that optimizer step.
