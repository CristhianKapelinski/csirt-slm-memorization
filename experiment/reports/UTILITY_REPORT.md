# UTILITY_REPORT — Severity Classification (logit-based, Plan A1)

**N adapters evaluated**: 96

**Method**: format-faithful prompt (training-faithful), Σ log P over the canonical continuation (`{value},"base_vector"`), argmax over the 4 buckets.
**Summary criterion**: random baseline 4-way = acc 0.25; F1-macro random ≈ 0.10. Small models fine-tuned on raw JSON suffer **format specialization** (Wang 2022 ProMoT) — classification capability stays marginally above random. Methodological finding, see §6.


## 1. F1-macro cross-seed (mean ± std over 3 seeds)

| model | variant | use_anon | F1_raw | F1_norm | acc_raw | acc_norm |
|---|---|:---:|---:|---:|---:|---:|
| gemma3-1b | v0 | — | 0.250±0.004 | 0.250 | 0.273 | 0.273 |
| gemma3-1b | v0 | Anon | 0.251±0.006 | 0.251 | 0.267 | 0.267 |
| gemma3-1b | v1 | — | 0.224±0.018 | 0.224 | 0.251 | 0.251 |
| gemma3-1b | v1 | Anon | 0.227±0.005 | 0.227 | 0.234 | 0.234 |
| gemma3-1b | v2 | — | 0.228±0.007 | 0.228 | 0.257 | 0.257 |
| gemma3-1b | v2 | Anon | 0.242±0.013 | 0.242 | 0.247 | 0.247 |
| gemma3-1b | v3 | — | 0.232±0.008 | 0.232 | 0.260 | 0.260 |
| gemma3-1b | v3 | Anon | 0.241±0.016 | 0.241 | 0.246 | 0.246 |
| qwen3-1.7b | v0 | — | 0.257±0.017 | 0.257 | 0.279 | 0.279 |
| qwen3-1.7b | v0 | Anon | 0.246±0.017 | 0.246 | 0.269 | 0.269 |
| qwen3-1.7b | v1 | — | 0.213±0.008 | 0.213 | 0.264 | 0.264 |
| qwen3-1.7b | v1 | Anon | 0.212±0.007 | 0.212 | 0.254 | 0.254 |
| qwen3-1.7b | v2 | — | 0.206±0.012 | 0.206 | 0.249 | 0.249 |
| qwen3-1.7b | v2 | Anon | 0.219±0.006 | 0.219 | 0.257 | 0.257 |
| qwen3-1.7b | v3 | — | 0.202±0.013 | 0.202 | 0.246 | 0.246 |
| qwen3-1.7b | v3 | Anon | 0.217±0.008 | 0.217 | 0.254 | 0.254 |
| llama32-3b | v0 | — | 0.278±0.011 | 0.278 | 0.303 | 0.303 |
| llama32-3b | v0 | Anon | 0.260±0.014 | 0.260 | 0.281 | 0.281 |
| llama32-3b | v1 | — | 0.210±0.008 | 0.210 | 0.241 | 0.241 |
| llama32-3b | v1 | Anon | 0.246±0.037 | 0.246 | 0.273 | 0.273 |
| llama32-3b | v2 | — | 0.246±0.031 | 0.246 | 0.258 | 0.258 |
| llama32-3b | v2 | Anon | 0.231±0.021 | 0.231 | 0.249 | 0.249 |
| llama32-3b | v3 | — | 0.245±0.016 | 0.245 | 0.254 | 0.254 |
| llama32-3b | v3 | Anon | 0.241±0.019 | 0.241 | 0.254 | 0.254 |
| vaultgemma-1b | v0 | — | 0.212±0.007 | 0.212 | 0.249 | 0.249 |
| vaultgemma-1b | v0 | Anon | 0.234±0.011 | 0.234 | 0.259 | 0.259 |
| vaultgemma-1b | v1 | — | 0.189±0.006 | 0.189 | 0.224 | 0.224 |
| vaultgemma-1b | v1 | Anon | 0.253±0.009 | 0.253 | 0.262 | 0.262 |
| vaultgemma-1b | v2 | — | 0.196±0.013 | 0.196 | 0.242 | 0.242 |
| vaultgemma-1b | v2 | Anon | 0.250±0.015 | 0.250 | 0.267 | 0.267 |
| vaultgemma-1b | v3 | — | 0.198±0.016 | 0.198 | 0.242 | 0.242 |
| vaultgemma-1b | v3 | Anon | 0.258±0.014 | 0.258 | 0.274 | 0.274 |

## 2. Vk vs Anon-Vk (effect of HMAC pseudonymization on utility)

**H_U1**: F1(Anon-Vk) ≈ F1(Vk). Diff < 5% absolute = PASS, utility preserved.

| model | variant | F1 Vk | F1 Anon | Δ abs | flag |
|---|---|---:|---:|---:|:---:|
| gemma3-1b | v0 | 0.250 | 0.251 | +0.001 | PASS |
| gemma3-1b | v1 | 0.224 | 0.227 | +0.004 | PASS |
| gemma3-1b | v2 | 0.228 | 0.242 | +0.014 | PASS |
| gemma3-1b | v3 | 0.232 | 0.241 | +0.010 | PASS |
| qwen3-1.7b | v0 | 0.257 | 0.246 | -0.011 | PASS |
| qwen3-1.7b | v1 | 0.213 | 0.212 | -0.002 | PASS |
| qwen3-1.7b | v2 | 0.206 | 0.219 | +0.013 | PASS |
| qwen3-1.7b | v3 | 0.202 | 0.217 | +0.015 | PASS |
| llama32-3b | v0 | 0.278 | 0.260 | -0.018 | PASS |
| llama32-3b | v1 | 0.210 | 0.246 | +0.036 | PASS |
| llama32-3b | v2 | 0.246 | 0.231 | -0.014 | PASS |
| llama32-3b | v3 | 0.245 | 0.241 | -0.004 | PASS |
| vaultgemma-1b | v0 | 0.212 | 0.234 | +0.022 | PASS |
| vaultgemma-1b | v1 | 0.189 | 0.253 | +0.065 | [!] |
| vaultgemma-1b | v2 | 0.196 | 0.250 | +0.054 | [!] |
| vaultgemma-1b | v3 | 0.198 | 0.258 | +0.059 | [!] |

## 3. Paired Wilcoxon per record (Vk vs Anon-Vk, cross-seed pooled)

Pairs predictions by record index (held-out raw and anon keep the same order). Wilcoxon signed-rank over the diff (Anon_correct − Vk_correct). BH-FDR q=0.05 over 16 comparisons (4 models × 4 variants).

| model | variant | n pairs | median_diff | stat | p (raw) | BH-FDR sig |
|---|---|---:|---:|---:|---:|:---:|
| gemma3-1b | v0 | 1800 | +0.000 | 104806.5 | 0.6953 |  |
| gemma3-1b | v1 | 1800 | +0.000 | 89790.0 | 0.226 |  |
| gemma3-1b | v2 | 1800 | +0.000 | 103342.0 | 0.5052 |  |
| gemma3-1b | v3 | 1800 | +0.000 | 102835.0 | 0.3093 |  |
| qwen3-1.7b | v0 | 1800 | +0.000 | 78660.0 | 0.476 |  |
| qwen3-1.7b | v1 | 1800 | +0.000 | 69541.5 | 0.4369 |  |
| qwen3-1.7b | v2 | 1800 | +0.000 | 78246.0 | 0.5562 |  |
| qwen3-1.7b | v3 | 1800 | +0.000 | 69680.0 | 0.5167 |  |
| llama32-3b | v0 | 1800 | +0.000 | 73884.0 | 0.1002 |  |
| llama32-3b | v1 | 1800 | +0.000 | 60060.0 | 0.01235 |  |
| llama32-3b | v2 | 1800 | +0.000 | 83790.0 | 0.4829 |  |
| llama32-3b | v3 | 1800 | +0.000 | 90150.0 | 1 |  |
| vaultgemma-1b | v0 | 1800 | +0.000 | 85095.5 | 0.4594 |  |
| vaultgemma-1b | v1 | 1800 | +0.000 | 76570.0 | 0.005043 |  |
| vaultgemma-1b | v2 | 1800 | +0.000 | 99256.0 | 0.08486 |  |
| vaultgemma-1b | v3 | 1800 | +0.000 | 101232.0 | 0.02708 |  |

## 4. Aggregated confusion matrices by (model, variant, anon)


### gemma3-1b / v0 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 239 | 95 | 60 | 56 |
| Medium | 305 | 121 | 108 | 66 |
| High | 241 | 72 | 87 | 50 |
| Critical | 157 | 37 | 62 | 44 |

### gemma3-1b / v0 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 182 | 108 | 55 | 105 |
| Medium | 231 | 166 | 73 | 130 |
| High | 171 | 115 | 54 | 110 |
| Critical | 101 | 81 | 39 | 79 |

### gemma3-1b / v1 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 230 | 103 | 82 | 35 |
| Medium | 330 | 100 | 111 | 59 |
| High | 238 | 74 | 94 | 44 |
| Critical | 185 | 47 | 41 | 27 |

### gemma3-1b / v1 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 166 | 46 | 95 | 143 |
| Medium | 205 | 77 | 140 | 178 |
| High | 199 | 32 | 92 | 127 |
| Critical | 100 | 47 | 67 | 86 |

### gemma3-1b / v2 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 238 | 88 | 86 | 38 |
| Medium | 331 | 92 | 119 | 58 |
| High | 229 | 68 | 105 | 48 |
| Critical | 175 | 49 | 49 | 27 |

### gemma3-1b / v2 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 143 | 42 | 117 | 148 |
| Medium | 172 | 69 | 168 | 191 |
| High | 150 | 35 | 142 | 123 |
| Critical | 95 | 34 | 80 | 91 |

### gemma3-1b / v3 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 239 | 93 | 81 | 37 |
| Medium | 332 | 96 | 113 | 59 |
| High | 231 | 68 | 106 | 45 |
| Critical | 178 | 51 | 44 | 27 |

### gemma3-1b / v3 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 146 | 43 | 107 | 154 |
| Medium | 177 | 70 | 146 | 207 |
| High | 158 | 28 | 125 | 139 |
| Critical | 87 | 35 | 77 | 101 |

### qwen3-1.7b / v0 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 212 | 100 | 89 | 49 |
| Medium | 262 | 163 | 126 | 49 |
| High | 218 | 99 | 88 | 45 |
| Critical | 131 | 53 | 77 | 39 |

### qwen3-1.7b / v0 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 210 | 92 | 105 | 43 |
| Medium | 283 | 134 | 132 | 51 |
| High | 236 | 66 | 109 | 39 |
| Critical | 123 | 46 | 99 | 32 |

### qwen3-1.7b / v1 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 213 | 65 | 148 | 24 |
| Medium | 295 | 65 | 223 | 17 |
| High | 185 | 58 | 193 | 14 |
| Critical | 152 | 43 | 101 | 4 |

### qwen3-1.7b / v1 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 219 | 101 | 116 | 14 |
| Medium | 289 | 101 | 182 | 28 |
| High | 223 | 73 | 131 | 23 |
| Critical | 134 | 69 | 91 | 6 |

### qwen3-1.7b / v2 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 204 | 57 | 163 | 26 |
| Medium | 285 | 49 | 247 | 19 |
| High | 188 | 52 | 184 | 26 |
| Critical | 140 | 34 | 114 | 12 |

### qwen3-1.7b / v2 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 215 | 82 | 137 | 16 |
| Medium | 273 | 106 | 189 | 32 |
| High | 219 | 70 | 132 | 29 |
| Critical | 119 | 52 | 119 | 10 |

### qwen3-1.7b / v3 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 205 | 63 | 163 | 19 |
| Medium | 283 | 44 | 255 | 18 |
| High | 191 | 49 | 183 | 27 |
| Critical | 142 | 37 | 110 | 11 |

### qwen3-1.7b / v3 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 220 | 72 | 141 | 17 |
| Medium | 288 | 95 | 188 | 29 |
| High | 217 | 72 | 132 | 29 |
| Critical | 114 | 54 | 121 | 11 |

### llama32-3b / v0 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 190 | 50 | 145 | 65 |
| Medium | 222 | 72 | 236 | 70 |
| High | 114 | 41 | 232 | 63 |
| Critical | 88 | 29 | 132 | 51 |

### llama32-3b / v0 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 189 | 43 | 165 | 53 |
| Medium | 235 | 87 | 230 | 48 |
| High | 160 | 48 | 189 | 53 |
| Critical | 75 | 43 | 141 | 41 |

### llama32-3b / v1 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 119 | 53 | 208 | 70 |
| Medium | 164 | 31 | 310 | 95 |
| High | 123 | 22 | 241 | 64 |
| Critical | 92 | 4 | 161 | 43 |

### llama32-3b / v1 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 131 | 40 | 221 | 58 |
| Medium | 169 | 70 | 280 | 81 |
| High | 107 | 36 | 246 | 61 |
| Critical | 65 | 21 | 170 | 44 |

### llama32-3b / v2 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 134 | 108 | 117 | 91 |
| Medium | 160 | 159 | 148 | 133 |
| High | 162 | 96 | 116 | 76 |
| Critical | 112 | 61 | 71 | 56 |

### llama32-3b / v2 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 112 | 102 | 183 | 53 |
| Medium | 162 | 111 | 253 | 74 |
| High | 136 | 63 | 185 | 66 |
| Critical | 70 | 58 | 132 | 40 |

### llama32-3b / v3 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 124 | 109 | 112 | 105 |
| Medium | 155 | 156 | 145 | 144 |
| High | 156 | 105 | 115 | 74 |
| Critical | 110 | 68 | 59 | 63 |

### llama32-3b / v3 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 117 | 101 | 186 | 46 |
| Medium | 166 | 122 | 241 | 71 |
| High | 138 | 73 | 173 | 66 |
| Critical | 70 | 67 | 117 | 46 |

### vaultgemma-1b / v0 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 247 | 69 | 59 | 75 |
| Medium | 305 | 127 | 62 | 106 |
| High | 232 | 89 | 54 | 75 |
| Critical | 186 | 71 | 23 | 20 |

### vaultgemma-1b / v0 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 206 | 45 | 142 | 57 |
| Medium | 256 | 87 | 186 | 71 |
| High | 198 | 49 | 144 | 59 |
| Critical | 128 | 46 | 97 | 29 |

### vaultgemma-1b / v1 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 256 | 71 | 64 | 59 |
| Medium | 385 | 59 | 78 | 78 |
| High | 286 | 31 | 60 | 73 |
| Critical | 182 | 50 | 39 | 29 |

### vaultgemma-1b / v1 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 176 | 101 | 82 | 91 |
| Medium | 208 | 135 | 131 | 126 |
| High | 149 | 102 | 94 | 105 |
| Critical | 77 | 85 | 71 | 67 |

### vaultgemma-1b / v2 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 289 | 47 | 60 | 54 |
| Medium | 393 | 58 | 78 | 71 |
| High | 291 | 26 | 67 | 66 |
| Critical | 194 | 49 | 35 | 22 |

### vaultgemma-1b / v2 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 140 | 154 | 76 | 80 |
| Medium | 183 | 201 | 105 | 111 |
| High | 128 | 143 | 84 | 95 |
| Critical | 68 | 106 | 71 | 55 |

### vaultgemma-1b / v3 / Vk

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 283 | 52 | 62 | 53 |
| Medium | 397 | 60 | 73 | 70 |
| High | 284 | 28 | 70 | 68 |
| Critical | 196 | 43 | 38 | 23 |

### vaultgemma-1b / v3 / Anon

| GT \ pred | Low | Medium | High | Critical |
|---|---:|---:|---:|---:|
| Low | 138 | 163 | 76 | 73 |
| Medium | 181 | 208 | 105 | 106 |
| High | 122 | 145 | 91 | 92 |
| Critical | 66 | 109 | 69 | 56 |

## 5. Methodological finding — format specialization

The adapters were fine-tuned on **serialized raw JSON** (next-token prediction). They were not trained on (prompt, label) pairs for severity classification. When given a natural prompt (`Severity:`), the models respond by continuing the JSON (95-100% invalid output) — a phenomenon known as **format specialization** (Wang et al. 2022, ProMoT, ICLR 2023). This reflects that fine-tuning destroys the ability to follow instructions outside the learned domain; it is not a failure of HMAC pseudonymization.

This experiment worked around the problem using **logit-based scoring in the training-faithful format**: the prompt reproduces exactly the serialization seen in training (compact JSON truncated at `"base_score":`), and the prediction is argmax over Σ log P of 4 canonical continuations (one per bucket). Acc raw vs acc_norm (byte-length normalized, lm-eval-harness convention) are reported in §1.

A cross-format ablation on 1 adapter per model (50 records each) confirmed:
- format-faithful (canonical) is the best for all of them;
- cross-format acc variation of 1.5×-2.75× — **prompt fragility**;
- the natural prompt (`Severity:`) drops to acc ≈ 0.16-0.24 (near random).