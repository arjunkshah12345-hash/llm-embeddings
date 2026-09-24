# Primary study aggregate

Generated directly from validated per-seed JSONL artifacts. Lower loss is better.

| Condition | Seeds | Total params | Extra vs tied | Mean final loss | Std. dev. | Mean best loss |
|---|---:|---:|---:|---:|---:|---:|
| tied | 3 | 30,023,808 | 0 | 5.374129 | 0.021395 | 5.373816 |
| untied | 3 | 49,322,496 | 19,298,688 | 5.438763 | 0.027742 | 5.438269 |
| partial | 3 | 30,834,064 | 810,256 | 5.377364 | 0.028314 | 5.377000 |
| capacity_control | 3 | 30,834,816 | 811,008 | 5.366543 | 0.021415 | 5.366543 |
| partial_input | 3 | 30,428,936 | 405,128 | 5.381567 | 0.024074 | 5.381145 |
| partial_output | 3 | 30,428,936 | 405,128 | 5.373475 | 0.031600 | 5.373475 |

## Paired differences versus tied

| Condition | Metric | Mean delta | 95% interval |
|---|---|---:|---:|
| untied | final_val_loss | 0.064634 | [0.056694, 0.068797] |
| untied | best_val_loss | 0.064453 | [0.055212, 0.069735] |
| partial | final_val_loss | 0.003235 | [-0.005557, 0.008260] |
| partial | best_val_loss | 0.003184 | [-0.005557, 0.008107] |
| capacity_control | final_val_loss | -0.007586 | [-0.027700, 0.008819] |
| capacity_control | best_val_loss | -0.007273 | [-0.026762, 0.008819] |
| partial_input | final_val_loss | 0.007438 | [0.004855, 0.010269] |
| partial_input | best_val_loss | 0.007329 | [0.004855, 0.010269] |
| partial_output | final_val_loss | -0.000653 | [-0.011484, 0.008784] |
| partial_output | best_val_loss | -0.000341 | [-0.011484, 0.008784] |

## Research question

```json
{
  "best_val_loss": {
    "partial": 5.377000490824381,
    "partial_improvement_over_tied": -0.003184469540912893,
    "partial_to_untied_gap": -0.06126851240793929,
    "recovered_fraction": null,
    "recovery_status": "untied_not_better_than_tied",
    "tied": 5.373816021283468,
    "untied": 5.43826900323232,
    "untied_improvement_over_tied": -0.06445298194885218
  },
  "final_val_loss": {
    "partial": 5.377363888422648,
    "partial_improvement_over_tied": -0.0032352447509760296,
    "partial_to_untied_gap": -0.061399086316426654,
    "recovered_fraction": null,
    "recovery_status": "untied_not_better_than_tied",
    "tied": 5.374128643671672,
    "untied": 5.438762974739075,
    "untied_improvement_over_tied": -0.06463433106740268
  }
}
```
