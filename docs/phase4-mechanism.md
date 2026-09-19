# Phase 4 — mechanism checks

These checks ask whether input and output roles push the shared matrix differently, and whether that push is just a token-frequency artifact. They are instrumentation. A six-step run is not evidence.

## Token classes

On each logged training step the trainer records the mean embedding-row gradient for tokens present in that batch. The input-side value is grouped by input tokens; the output-side value is grouped by target tokens:

- whitespace and punctuation, from the decoded surface form;
- common: the 1,000 highest-count remaining training tokens;
- rare: remaining tokens seen at most once;
- other: everything left.

Keys look like `output_token_grad_rare_mean`. `analyze.py` plots the output-side means in `token_type_gradients.png` when those keys exist.

## Path ablations

Stop one role's gradient for a step range without changing the loss:

```bash
python3 train.py --embedding_type tied --path_ablation stop_input --ablation_start 0 --ablation_end 200 --steps 400 --run_name tied_stop_input
python3 train.py --embedding_type tied --path_ablation stop_output --ablation_start 0 --ablation_end 200 --steps 400 --run_name tied_stop_output
```

`stop_input` detaches the token lookup, so input embedding parameters get no gradient on those steps. `stop_output` detaches the output matrix, so output embedding parameters get no gradient while the input path still trains. Outside `[start, end)` training is normal. `ablation_end 0` means through the last step.

Keep these runs out of the tied/partial/untied comparison. They are ablations, not the primary experiment.
