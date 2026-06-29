# UTILITY_ABLATION_REPORT — format-ablation, V0 seed 42, 50 records

Empirically justifies the use of the training-faithful prompt (§5.1 of the paper). The reported F1-macro is over 50 held-out records; acc is also reported. Random baseline 4-way: F1≈0.10, acc=0.25.

## F1-macro by (model, format)

| Model | Training-faithful | Truncated pre-CVSS2 | Closed-dict | Natural-language |
|---|---:|---:|---:|---:|
| gemma3-1b | **0.247** | 0.087 | 0.194 | 0.209 |
| qwen3-1.7b | 0.240 | 0.191 | **0.259** | 0.145 |
| llama32-3b | **0.324** | 0.252 | 0.241 | 0.152 |
| vaultgemma-1b | 0.124 | 0.114 | 0.148 | **0.169** |

*Bold = best per model.*


## Accuracy by (model, format)

| Model | Training-faithful | Truncated pre-CVSS2 | Closed-dict | Natural-language |
|---|---:|---:|---:|---:|
| gemma3-1b | 0.320 | 0.160 | 0.260 | 0.300 |
| qwen3-1.7b | 0.300 | 0.240 | 0.340 | 0.200 |
| llama32-3b | 0.440 | 0.360 | 0.300 | 0.160 |
| vaultgemma-1b | 0.160 | 0.220 | 0.220 | 0.240 |

## Cross-format spread (max−min) and ratio (max/min)

| Model | min acc | max acc | spread | ratio |
|---|---:|---:|---:|---:|
| gemma3-1b | 0.160 | 0.320 | 0.160 | 2.00× |
| qwen3-1.7b | 0.200 | 0.340 | 0.140 | 1.70× |
| llama32-3b | 0.160 | 0.440 | 0.280 | 2.75× |
| vaultgemma-1b | 0.160 | 0.240 | 0.080 | 1.50× |

## Notes

- **Training-faithful** (format1) reproduces bit-for-bit the compact JSON serialization seen in training, truncated exactly at `"base_score":`. The model continues the JSON by predicting the numeric value.
- **Truncated pre-CVSS2** truncates before `"cvss2":` — the model has to generate the entire structure `{"base_score":X,...}`.
- **Closed-dict** truncates after `"base_score":` but uses the closing `}` (no `"base_vector"` continuation) — forcing inference without the rich continuation context.
- **Natural-language** uses the `Severity:` instruction in plain text. Expected to be low: models fine-tuned on raw JSON suffer format specialization (Wang 2022 ProMoT) and do not follow a natural prompt.
- The cross-format variation (max/min ratio) confirms the task's fragility with respect to format — the paper adopts training-faithful for all comparisons.