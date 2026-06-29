# Experiment 1 - Final Report (Multi-Seed)

**Seeds:** [42, 1337, 2024]  -  **Adapters:** 48/48 expected

> Multi-seed: paired tests by (canary, seed) -> n_pooled = 20 x 3 = **60**.

---

## 1. Aggregate metrics (mean +/- std cross-seed)

| model | variant | exp_all (mean+/-std) | AUC_loss_canary (mean+/-std) |
|--------|----------|--------------------:|--------------------------:|
| gemma3-1b         |       v0 |      3.49 +/- 0.49 |            0.592 +/- 0.003 |
| gemma3-1b         |       v1 |      1.38 +/- 0.14 |            0.510 +/- 0.009 |
| gemma3-1b         |       v2 |      1.54 +/- 0.32 |            0.522 +/- 0.004 |
| gemma3-1b         |       v3 |      1.50 +/- 0.35 |            0.525 +/- 0.009 |
| qwen3-1.7b        |       v0 |      4.48 +/- 0.31 |            0.770 +/- 0.020 |
| qwen3-1.7b        |       v1 |      2.83 +/- 0.27 |            0.570 +/- 0.011 |
| qwen3-1.7b        |       v2 |      2.60 +/- 0.39 |            0.564 +/- 0.004 |
| qwen3-1.7b        |       v3 |      2.51 +/- 0.40 |            0.558 +/- 0.005 |
| llama32-3b        |       v0 |      4.98 +/- 0.18 |            0.848 +/- 0.009 |
| llama32-3b        |       v1 |      3.20 +/- 0.20 |            0.594 +/- 0.004 |
| llama32-3b        |       v2 |      2.68 +/- 0.32 |            0.578 +/- 0.008 |
| llama32-3b        |       v3 |      2.92 +/- 0.58 |            0.576 +/- 0.012 |
| vaultgemma-1b     |       v0 |      3.60 +/- 0.15 |            0.596 +/- 0.005 |
| vaultgemma-1b     |       v1 |      1.24 +/- 0.26 |            0.486 +/- 0.004 |
| vaultgemma-1b     |       v2 |      1.26 +/- 0.08 |            0.487 +/- 0.004 |
| vaultgemma-1b     |       v3 |      1.45 +/- 0.42 |            0.494 +/- 0.005 |

## 2. Paired Wilcoxon by (canary x seed) - intra-model

| Hypothesis | Model | n | mean_diff | median_diff | stat | p (raw) | Cliff delta |
|----------|--------|--:|----------:|------------:|-----:|--------:|--------:|
| H6'  V1-V0                     | gemma3-1b                           |  60 | -2.103 | -1.866 |   84.0 | 2.33e-09 | -0.604 * |
| H6'  V1-V0                     | qwen3-1.7b                          |  60 | -1.652 | -1.292 |  166.0 | 1.91e-06 | -0.383 * |
| H6'  V1-V0                     | llama32-3b                          |  60 | -1.784 | -1.661 |  215.5 | 9.55e-06 | -0.448 * |
| H6'  V1-V0                     | vaultgemma-1b                       |  60 | -2.366 | -2.146 |  121.0 | 8.08e-09 | -0.561 * |
| H6   V2-V1                     | gemma3-1b                           |  60 | +0.155 | +0.000 |  771.5 |   0.662 | +0.048 |
| H6   V2-V1                     | qwen3-1.7b                          |  60 | -0.229 | -0.099 |  494.0 |   0.050 | -0.076 |
| H6   V2-V1                     | llama32-3b                          |  60 | -0.513 | -0.200 |  476.0 |   0.052 | -0.126 |
| H6   V2-V1                     | vaultgemma-1b                       |  60 | +0.025 | +0.000 |  740.5 |   0.805 | +0.003 |
| H6e2 V3-V1                     | gemma3-1b                           |  60 | +0.113 | -0.016 |  791.5 |   0.781 | +0.023 |
| H6e2 V3-V1                     | qwen3-1.7b                          |  60 | -0.321 | -0.029 |  519.0 |   0.082 | -0.108 |
| H6e2 V3-V1                     | llama32-3b                          |  60 | -0.278 | +0.000 |  501.0 |   0.188 | -0.072 |
| H6e2 V3-V1                     | vaultgemma-1b                       |  60 | +0.207 | +0.039 |  616.5 |   0.198 | +0.043 |
| H9   V3-V2                     | gemma3-1b                           |  60 | -0.042 | +0.009 |  842.5 |   0.748 | -0.027 |
| H9   V3-V2                     | qwen3-1.7b                          |  60 | -0.092 | +0.000 |  646.0 |   0.873 | -0.027 |
| H9   V3-V2                     | llama32-3b                          |  60 | +0.235 | +0.000 |  472.0 |   0.073 | +0.049 |
| H9   V3-V2                     | vaultgemma-1b                       |  60 | +0.182 | +0.000 |  680.5 |   0.593 | +0.039 |

## 3. Paired Wilcoxon by (canary x seed) - H8 (defense in depth)

| Hypothesis | Models | n | mean_diff | median_diff | stat | p (raw) | Cliff delta |
|----------|---------|--:|----------:|------------:|-----:|--------:|--------:|
| H8a  vault-V0 vs gemma3-V0     | vaultgemma-1b / gemma3-1b           |  60 | +0.118 | +0.000 |  586.5 |   0.473 | +0.007 |
| H8a' vault-V1 vs gemma3-V1     | vaultgemma-1b / gemma3-1b           |  60 | -0.145 | -0.117 |  772.0 |   0.394 | -0.118 |
| H8b  vault V2 vs V0            | vaultgemma-1b                       |  60 | -2.341 | -2.391 |  147.0 | 1.57e-08 | -0.557 * |
| H8b' vault V2 vs V1            | vaultgemma-1b                       |  60 | +0.025 | +0.000 |  740.5 |   0.805 | +0.003 |
| H8c  vault V3 vs V2            | vaultgemma-1b                       |  60 | +0.182 | +0.000 |  680.5 |   0.593 | +0.039 |

## 4. Friedman 4-way (V0/V1/V2/V3) per model (pooled multi-seed)

| Model | n | chi2 | p (raw) |
|--------|--:|----:|--------:|
| gemma3-1b         |  60 | 57.68 | 1.84e-12 |
| qwen3-1.7b        |  60 | 41.13 | 6.15e-09 |
| llama32-3b        |  60 | 55.85 | 4.53e-12 |
| vaultgemma-1b     |  60 | 42.96 | 2.51e-09 |

## 5. Hypothesis summary

BH-FDR q=0.05 over all paired p-values above.

| Hypothesis | Sig. + median_diff < 0 in models |
|----------|-------------------------------------|
| H6'      | gemma3-1b, llama32-3b, vaultgemma-1b, qwen3-1.7b |
| H6       | gemma3-1b, llama32-3b, vaultgemma-1b, qwen3-1.7b |
| H6e2     | - |
| H9       | - |
| H8a      | - |
| H8a'     | - |
| H8b      | vaultgemma-1b |
| H8b'     | - |
| H8c      | - |

## 6. Notes

- **Multi-seed pooling**: each pair has n = 20 canaries x 3 seeds = 60 points. Inference gains power vs single-seed (W1).
- BH-FDR controls family-wise error via FDR over **all** reported p-values.
- Negative Cliff delta = first variant < second (i.e., exposure reduction).
- Expected direction: H6'/H6/H6e2/H9/H8a/b/c all predict `mean_diff < 0`.