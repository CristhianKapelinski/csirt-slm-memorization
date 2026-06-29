# ABLATION_STACK_REPORT - stack decomposition (Sec 4.4)

Delta = Exp(cell) - Exp(V0), cross-seed (n=3). A = large-lot batching (lot 32, bf16), B = NF4 quantization (lot 4, NF4), V1 = full stack. 95% CIs from a paired bootstrap over 60 (canary x seed) observations (B=10000).

| Model | Delta_A [95% CI] | Delta_B [95% CI] | Delta_V1 [95% CI] |
|---|---:|---:|---:|
| gemma3-1b | -2.00 [-2.41, -1.54] | -0.67 [-1.11, -0.25] | -2.10 [-2.59, -1.62] |
| qwen3-1.7b | -1.53 [-2.07, -0.99] | -0.63 [-0.99, -0.29] | -1.65 [-2.21, -1.10] |
| llama32-3b | -1.79 [-2.28, -1.30] | -0.19 [-0.54, +0.15] | -1.78 [-2.43, -1.12] |
| vaultgemma-1b | -0.96 [-1.50, -0.44] | -1.41 [-1.96, -0.86] | -2.37 [-2.96, -1.78] |

Delta_V1 range across models: [-2.37, -1.65]
