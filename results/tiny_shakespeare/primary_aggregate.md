# Primary study aggregate

Generated directly from validated per-seed JSONL artifacts. Lower loss is better.

| Condition | Seeds | Total params | Extra vs tied | Mean final loss | Mean final PPL | Tokens/s | Wall s | Peak MB | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied | 3 | 30,023,808 | 0 | 4.977866 | 145.1663 | 10740.34 | 477.0 | 1716.8 | 9.223e+14 |
| untied | 3 | 49,322,496 | 19,298,688 | 5.134988 | 169.9198 | 9622.51 | 532.4 | 2012.1 | 9.223e+14 |
| partial | 3 | 30,834,064 | 810,256 | 5.041516 | 154.7271 | 9448.85 | 542.2 | 1873.2 | 9.349e+14 |
| capacity_control | 3 | 30,834,816 | 811,008 | 5.533017 | 253.0435 | 10348.40 | 495.1 | 1739.0 | 9.472e+14 |

## Paired differences versus tied

| Condition | Metric | Mean delta | 95% interval |
|---|---|---:|---:|
| untied | final_val_loss | 0.157123 | [0.114829, 0.179687] |
| untied | best_val_loss | 0.031010 | [-0.001678, 0.095979] |
| untied | final_val_perplexity | 24.753505 | [17.762420, 28.369370] |
| untied | best_val_perplexity | 2.494198 | [-0.128880, 7.715589] |
| partial | final_val_loss | 0.063650 | [0.041720, 0.094485] |
| partial | best_val_loss | 0.013566 | [0.002921, 0.026597] |
| partial | final_val_perplexity | 9.560802 | [6.218824, 14.281452] |
| partial | best_val_perplexity | 1.070476 | [0.224817, 2.064469] |
| capacity_control | final_val_loss | 0.555151 | [0.511973, 0.601625] |
| capacity_control | best_val_loss | 0.118635 | [0.100586, 0.143602] |
| capacity_control | final_val_perplexity | 107.877214 | [97.213137, 118.912644] |
| capacity_control | best_val_perplexity | 9.864305 | [8.679998, 11.827712] |

## Research question

```json
{
  "best_val_loss": {
    "partial": 4.376100401083629,
    "partial_improvement_over_tied": -0.01356566349665389,
    "partial_to_untied_gap": -0.017444666226705152,
    "recovered_fraction": null,
    "recovery_status": "statistically_uncertain_or_zero_denominator",
    "tied": 4.362534737586975,
    "untied": 4.393545067310334,
    "untied_improvement_over_tied": -0.031010329723359042
  },
  "final_val_loss": {
    "partial": 5.041515910625458,
    "partial_improvement_over_tied": -0.06365024646123274,
    "partial_to_untied_gap": -0.09347240130106549,
    "recovered_fraction": null,
    "recovery_status": "untied_not_better_than_tied",
    "tied": 4.977865664164225,
    "untied": 5.134988311926524,
    "untied_improvement_over_tied": -0.15712264776229823
  }
}
```
