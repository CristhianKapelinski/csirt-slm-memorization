# UTILITY_PROMPT_REPORT — Few-shot base models (Exp 4)

**Method**: 4 base models × 2 versions (raw / anon) × 600 held-out records, few-shot (4 balanced examples Low/Medium/High/Critical), greedy decoding max_new_tokens=4, case-insensitive parsing.
**Question**: does HMAC pseudonymization destroy information in the *data* (not in the adapters)? Replicates the Severo 2025 / Almeida 2025 protocol.
**Hypothesis H_U1 (this experiment)**: F1(model, anon) ≈ F1(model, raw). Diff < 5% absolute = PASS, utility preserved.


## 1. F1-macro raw vs anon per model

| model | F1 raw | F1 anon | Δ abs | Δ rel | acc raw | acc anon | inv raw | inv anon | flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| gemma3-1b | 0.136 | 0.146 | +0.010 | +7.3% | 0.192 | 0.235 | 0.0% | 0.0% | PASS |
| qwen3-1.7b | 0.165 | 0.167 | +0.003 | +1.6% | 0.280 | 0.264 | 29.7% | 11.0% | PASS |
| llama32-3b | 0.071 | 0.077 | +0.006 | +8.5% | 0.165 | 0.170 | 0.0% | 0.0% | PASS |
| vaultgemma-1b | 0.000 | 0.000 | +0.000 | +0.0% | 0.000 | 0.000 | 100.0% | 100.0% | (i) |

*Criterion `(i)`*: invalid_rate > 50% — the model does not follow the prompt template, so F1 does not measure classification capability (vault-1b is DP-pretrained, not full instruction-tuned).


## 2. Paired Wilcoxon per record (raw vs anon)

Same record_id in both versions → exact pairing. Diff = (anon_correct − raw_correct) ∈ {−1, 0, +1}. BH-FDR q=0.05 over 4 models.

| model | n pairs | median_diff | agreement (anon=raw) | n raw_only | n anon_only | stat | p (raw) | BH-FDR sig |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| gemma3-1b | 600 | +0.000 | 482 | 46 | 72 | 2737.0 | 0.01669 | * |
| qwen3-1.7b | 600 | +0.000 | 525 | 26 | 49 | 988.0 | 0.007912 | * |
| llama32-3b | 600 | +0.000 | 597 | 0 | 3 | 0.0 | 0.08326 |  |
| vaultgemma-1b | 600 | +0.000 | 600 | 0 | 0 | — | — | — |

## 3. Per-class F1 raw vs anon

| model | version | Low | Medium | High | Critical |
|---|---|---:|---:|---:|---:|
| gemma3-1b | raw | 0.000 | 0.010 | 0.260 | 0.276 |
| gemma3-1b | anon | 0.000 | 0.010 | 0.376 | 0.199 |
| qwen3-1.7b | raw | 0.208 | 0.040 | 0.411 | 0.000 |
| qwen3-1.7b | anon | 0.149 | 0.099 | 0.383 | 0.040 |
| llama32-3b | raw | 0.000 | 0.000 | 0.000 | 0.283 |
| llama32-3b | anon | 0.000 | 0.020 | 0.000 | 0.288 |
| vaultgemma-1b | raw | 0.000 | 0.000 | 0.000 | 0.000 |
| vaultgemma-1b | anon | 0.000 | 0.000 | 0.000 | 0.000 |

## 4. Confusion matrices


### gemma3-1b / raw

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 0 | 0 | 65 | 85 |
| Medium | 0 | 1 | 78 | 121 |
| High | 0 | 1 | 49 | 100 |
| Critical | 0 | 0 | 35 | 65 |

### gemma3-1b / anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 0 | 0 | 121 | 29 |
| Medium | 1 | 1 | 160 | 38 |
| High | 0 | 0 | 118 | 32 |
| Critical | 0 | 0 | 78 | 22 |

### qwen3-1.7b / raw

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 13 | 0 | 83 | 0 |
| Medium | 10 | 3 | 133 | 0 |
| High | 4 | 0 | 102 | 1 |
| Critical | 2 | 0 | 71 | 0 |

### qwen3-1.7b / anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 11 | 4 | 115 | 1 |
| Medium | 4 | 10 | 168 | 1 |
| High | 2 | 3 | 118 | 10 |
| Critical | 0 | 3 | 82 | 2 |

### llama32-3b / raw

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 0 | 0 | 0 | 150 |
| Medium | 0 | 0 | 0 | 200 |
| High | 0 | 0 | 0 | 150 |
| Critical | 0 | 1 | 0 | 99 |

### llama32-3b / anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 0 | 2 | 0 | 148 |
| Medium | 0 | 2 | 0 | 198 |
| High | 0 | 1 | 0 | 149 |
| Critical | 0 | 0 | 0 | 100 |

### vaultgemma-1b / raw

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 0 | 0 | 0 | 0 |
| Medium | 0 | 0 | 0 | 0 |
| High | 0 | 0 | 0 | 0 |
| Critical | 0 | 0 | 0 | 0 |

### vaultgemma-1b / anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 0 | 0 | 0 | 0 |
| Medium | 0 | 0 | 0 | 0 |
| High | 0 | 0 | 0 | 0 |
| Critical | 0 | 0 | 0 | 0 |

## 5. Notes and limitations

- **Fixed few-shot examples**: 4 records from `sample_3000_seed42.json` (1 per bucket, midrange — Low=3.5, Medium=5.0, High=7.5, Critical=10.0). Same examples across all 4 models × 2 versions. For the `anon` version, the examples go through HMAC pseudonymization with the same HMAC key.
- **Greedy decoding** (do_sample=False, max_new_tokens=4). Does not capture variability.
- **VaultGemma**: high invalid_rate (100%) — the model is not full instruction-tuned and does not follow the prompt template. Expected per the limitation documented in the plan (§13).
- **Paired data**: held-out raw and anon contain the same records (only pseudonymized). record_id is preserved in both versions → exact pairing for Wilcoxon.