# Primary study aggregate

Generated directly from validated per-seed JSONL artifacts. Lower loss is better.

| Condition | Seeds | Total params | Extra vs tied | Mean final loss | Mean final PPL | Tokens/s | Wall s | Peak MB | FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tied | 3 | 16,081,664 | 0 | 5.501100 | 244.9830 | 18553.34 | 276.1 | 1384.7 | 4.940e+14 |
| untied | 3 | 28,947,456 | 12,865,792 | 5.521981 | 250.2210 | 16172.35 | 316.7 | 1584.5 | 4.940e+14 |
| partial | 3 | 16,889,872 | 808,208 | 5.501633 | 245.1282 | 14941.34 | 343.0 | 1496.1 | 5.065e+14 |
| capacity_control | 3 | 16,890,112 | 808,448 | 5.494046 | 243.2558 | 17384.75 | 294.9 | 1415.0 | 5.189e+14 |

## Paired differences versus tied

| Condition | Metric | Mean delta | 95% interval |
|---|---|---:|---:|
| untied | final_val_loss | 0.020881 | [-0.009483, 0.059558] |
| untied | best_val_loss | 0.021611 | [-0.009483, 0.059558] |
| untied | final_val_perplexity | 5.237923 | [-2.296614, 14.854422] |
| untied | best_val_perplexity | 5.419869 | [-2.296614, 14.854422] |
| partial | final_val_loss | 0.000533 | [-0.007100, 0.006101] |
| partial | best_val_loss | 0.000272 | [-0.007100, 0.005317] |
| partial | final_val_perplexity | 0.145166 | [-1.721648, 1.527244] |
| partial | best_val_perplexity | 0.078548 | [-1.721648, 1.327391] |
| capacity_control | final_val_loss | -0.007054 | [-0.014612, 0.004849] |
| capacity_control | best_val_loss | -0.006324 | [-0.014612, 0.004849] |
| capacity_control | final_val_perplexity | -1.727190 | [-3.529761, 1.176661] |
| capacity_control | best_val_perplexity | -1.545244 | [-3.529761, 1.176661] |

## Research question

```json
{
  "best_val_loss": {
    "partial": 5.500641910235087,
    "partial_improvement_over_tied": -0.0002717018127444959,
    "partial_to_untied_gap": -0.021338955561319928,
    "recovered_fraction": null,
    "recovery_status": "statistically_uncertain_or_zero_denominator",
    "tied": 5.500370208422343,
    "untied": 5.521980865796407,
    "untied_improvement_over_tied": -0.021610657374064424
  },
  "final_val_loss": {
    "partial": 5.501633350054424,
    "partial_improvement_over_tied": -0.0005332628885907909,
    "partial_to_untied_gap": -0.020347515741983457,
    "recovered_fraction": null,
    "recovery_status": "statistically_uncertain_or_zero_denominator",
    "tied": 5.501100087165833,
    "untied": 5.521980865796407,
    "untied_improvement_over_tied": -0.020880778630574248
  }
}
```
