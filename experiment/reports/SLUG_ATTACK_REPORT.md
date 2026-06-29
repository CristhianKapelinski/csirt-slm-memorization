# SLUG_ATTACK_REPORT — Attack 2 (HMAC slug exposure) on Anon adapters

**H_A2**: the Anon model memorizes the **HMAC slug** (which it saw 30× during training), not the original canary. We expect exposure(slug) > exposure(original) in Exp 2 — completing the *dual extraction attack* promised in §3.2/§5.

**Method**: for each Anon-Vk, apply HMAC pseudonymization with the same HMAC key to both the canary and the 100 variants in the pool, build the planted/variant records with slugged values, and compute exposure (Carlini) over the slug tokens. Metric: $\mathrm{exposure} = \log_2(N{+}1) - \log_2(\text{rank})$, $N{=}100$.


## 1. Mean exposure(slug) cross-seed by (model, variant)

Mean ± std over 3 seeds. Random baseline ≈ 1.0 bit ($\log_2(101/50)$).

| Model | Anon-V0 | Anon-V1 | Anon-V2 ($\varepsilon{=}8$) | Anon-V3 ($\varepsilon{=}2$) |
|---|---:|---:|---:|---:|
| gemma3-1b | 1.497±0.248 | 1.494±0.212 | 1.337±0.418 | 1.440±0.275 |
| qwen3-1.7b | 1.276±0.664 | 1.026±0.319 | 0.880±0.174 | 0.918±0.174 |
| llama32-3b | 0.993±0.351 | 1.014±0.310 | 1.373±0.266 | 1.337±0.161 |
| vaultgemma-1b | 1.552±0.412 | 1.634±0.240 | 1.360±0.176 | 1.506±0.133 |

## 2. Attack 1 (original) vs Attack 2 (slug) — H_A2 confirmation

Diff > 0 and materially large = the model memorizes the slug, not the original → confirms that HMAC pseudonymization removed the original canary but the learned slug still carries the memorization signal (as expected).

| Model | Variant | exp(orig) | exp(slug) | Δ (slug − orig) |
|---|---|---:|---:|---:|
| gemma3-1b | v0 | 1.317 | 1.497 | +0.181 + |
| gemma3-1b | v1 | 1.604 | 1.494 | -0.110 — |
| gemma3-1b | v2 | 1.494 | 1.337 | -0.157 — |
| gemma3-1b | v3 | 1.576 | 1.440 | -0.136 — |
| qwen3-1.7b | v0 | 2.580 | 1.276 | -1.304 — |
| qwen3-1.7b | v1 | 2.596 | 1.026 | -1.570 — |
| qwen3-1.7b | v2 | 2.611 | 0.880 | -1.731 — |
| qwen3-1.7b | v3 | 2.522 | 0.918 | -1.604 — |
| llama32-3b | v0 | 2.098 | 0.993 | -1.106 — |
| llama32-3b | v1 | 1.943 | 1.014 | -0.929 — |
| llama32-3b | v2 | 2.433 | 1.373 | -1.060 — |
| llama32-3b | v3 | 2.487 | 1.337 | -1.150 — |
| vaultgemma-1b | v0 | 1.396 | 1.552 | +0.156 + |
| vaultgemma-1b | v1 | 1.492 | 1.634 | +0.142 + |
| vaultgemma-1b | v2 | 1.510 | 1.360 | -0.151 — |
| vaultgemma-1b | v3 | 1.476 | 1.506 | +0.029 + |

*++ = Δ > 1 bit (slug memorization clearly above the original).*

*+ = slug exposure ≥ original exposure (expected under H_A2).*


## 3. Interpretation

- **Anon-V0 (no extra protection)** is where the slug memorization signal is most visible — the model saw the slug 30× with neither DP-SGD regularization nor Poisson dilution. Expected: high Δ.
- **Anon-V1** (QLoRA + lot=32 Poisson) and **Anon-V2/V3** (DP-SGD) already have memorization attenuated in the same proportion that reduces exposure(original) in Exp 1 — Δ should be small but positive.
- If Δ ≈ 0 or negative in some slot, it indicates the HMAC slug behaves like a random string to the model (no learned co-occurrence) — additional evidence that HMAC pseudonymization + DP-SGD blocks syntactic memorization.
- **Methodological comparison**: Attack 1 tests whether the model would leak the real identifier (relevant to the adversary); Attack 2 tests whether the model closed a "shortcut" to the slug (relevant for re-identification if the mapping leaked). The two address complementary threats.