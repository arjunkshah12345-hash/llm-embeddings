# Primary study aggregate

Generated directly from validated per-seed JSONL artifacts. Lower loss is better.

| Condition | Seeds | Total params | Extra vs tied | Mean final loss | Mean final PPL | Tokens/s | Wall s | Peak MB | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied | 3 | 30,023,808 | 0 | 5.068562 | 158.9514 | 10362.97 | 2472.4 | 1716.8 | 4.612e+15 |
| untied | 3 | 49,322,496 | 19,298,688 | 5.427333 | 227.6409 | 9363.14 | 2736.1 | 2012.1 | 4.612e+15 |
| partial | 3 | 30,834,064 | 810,256 | 5.107840 | 165.3255 | 9152.17 | 2799.8 | 1873.2 | 4.674e+15 |
| capacity_control | 3 | 30,834,816 | 811,008 | 5.279322 | 196.2498 | 10083.74 | 2540.6 | 1739.0 | 4.736e+15 |

## Paired differences versus tied

| Condition | Metric | Mean delta | 95% interval |
|---|---|---:|---:|
| untied | final_val_loss | 0.358771 | [0.315596, 0.381454] |
| untied | best_val_loss | 0.182308 | [0.174257, 0.194245] |
| untied | final_val_perplexity | 68.689576 | [59.177235, 74.441187] |
| untied | best_val_perplexity | 28.094453 | [26.691075, 30.587374] |
| partial | final_val_loss | 0.039278 | [0.017152, 0.067284] |
| partial | best_val_loss | 0.041901 | [0.035354, 0.053284] |
| partial | final_val_perplexity | 6.374157 | [2.773007, 10.933280] |
| partial | best_val_perplexity | 5.999872 | [5.045794, 7.566511] |
| capacity_control | final_val_loss | 0.210760 | [0.197664, 0.218492] |
| capacity_control | best_val_loss | 0.103508 | [0.082959, 0.126085] |
| capacity_control | final_val_perplexity | 37.298430 | [34.853651, 39.142969] |
| capacity_control | best_val_perplexity | 15.368660 | [11.958555, 19.171670] |

## Research question

```json
{
  "best_val_loss": {
    "partial": 4.986158219973246,
    "partial_improvement_over_tied": -0.041901369889576934,
    "partial_to_untied_gap": -0.14040679931640643,
    "recovered_fraction": null,
    "recovery_status": "untied_not_better_than_tied",
    "tied": 4.944256850083669,
    "untied": 5.126565019289653,
    "untied_improvement_over_tied": -0.18230816920598336
  },
  "final_val_loss": {
    "partial": 5.10783992211024,
    "partial_improvement_over_tied": -0.039278268814086914,
    "partial_to_untied_gap": -0.31949274937311767,
    "recovered_fraction": null,
    "recovery_status": "untied_not_better_than_tied",
    "tied": 5.068561653296153,
    "untied": 5.427332671483358,
    "untied_improvement_over_tied": -0.3587710181872046
  }
}
```
