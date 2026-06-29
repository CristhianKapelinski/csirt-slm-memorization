# Decomposing Memorization Reduction in Privacy-Preserving Fine-Tuning of SLMs for CSIRTs

This is the companion artifact for the BRACIS 2026 paper of the same name. It packages the full empirical pipeline for studying canary memorization when Small Language Models are fine-tuned on CSIRT vulnerability-scan data. The study covers four 1–3B SLMs (Gemma 3 1B, Qwen3 1.7B, Llama 3.2 3B, VaultGemma 1B) across four protection regimes — raw bf16 (V0), QLoRA with large-lot Poisson sampling (V1), DP-SGD at ε=8 (V2), and DP-SGD at ε=2 (V3) — measures memorization with four scoring metrics (Carlini exposure, Yeom AUC, Lukas loss-canary AUC, Min-K%++), evaluates an HMAC-SHA256 pseudonymization layer (AnonShield) with a dual-target extraction protocol over the original identifier and its slug, and probes downstream CVSS-severity classification utility. The artifact ships the pipeline, the synthetic dataset generator, the canaries, the training configs, the pre-computed evaluation results, the analyses, and the verification harness that grounds every numerical claim in the paper. The no-GPU path regenerates all reports and verifies all paper numbers in about a minute; the GPU path retrains everything from scratch.

## Requirements

- Python 3.12.
- **Path A (no GPU):** only `numpy` + `scipy`. Install with `pip install .[reproduce]`.
- **Path B (full training):** a CUDA GPU plus the ML stack. Install with `pip install .[train]`.

`uv` works as a drop-in too: `uv pip install .[reproduce]` (or `.[train]`).

## Quick reproduce (no GPU)

```bash
pip install .[reproduce]
./reproduce.sh
```

`reproduce.sh` regenerates every report under `experiment/reports/` from the shipped pre-computed evaluation results and then runs `scripts/verify_claims.py`, which asserts all paper numbers within tolerance: **19 checks** covering Tables 1–6, the matched-update finding, and the cross-seed stack ablation. It takes about one minute and exits 0 if and only if every check passes.

## Full reproduce (from scratch, GPU)

```bash
./reproduce_full.sh
```

This installs the training stack, prepares the datasets, pseudonymizes the Anon splits, builds the configs, trains the LoRA adapters, runs the canary, slug, and utility evaluations, and finally re-runs all analyses via `reproduce.sh`. End-to-end cost is roughly **222 GPU-hours**.

## AnonShield (external dependency)

The HMAC pseudonymization tool is **not** vendored in this repository. It is maintained separately at `github.com/AnonShield/anonshield` and published on Docker Hub as `anonshield/anon`. Fetch it with:

```bash
./fetch_anonshield.sh
```

The script pulls the Docker image when Docker is available, otherwise it clones the source checkout. Path B uses AnonShield to produce the Anon-V0 cells (pseudonymizing `train_v0.jsonl` and the utility held-out split with your own `ANON_SECRET_KEY`).

## What reproduces what

| Paper claim | Table | Script | `verify_claims.py` asserts |
| --- | --- | --- | --- |
| Cross-seed exposure V0–V3 | Table 1 | `analyze_final.py` | 8 per-cell V0/V1 exposure means within ±0.05 bits |
| QLoRA stack cuts V0→V1 exposure 36–66% | Table 1 | `analyze_final.py` | V0→V1 reduction range falls in [36%, 66%] |
| AnonShield cuts V0 exposure 40–61% | Table 2 | `analyze_anonshield.py` | Anon-V0 reductions in [40%, 61%] for ≥4 models |
| Matched-update control: recovered fraction 66–132%, mean 100% | Finding (i) | `analyze_matched_update.py` | per-model fractions match paper (±2), mean ≈100%, mean Δ ≈ −1.98 bits |
| Stack decomposition Δ_A (lot size) / Δ_B (NF4) + CIs | §4.4 | `analyze_ablation_stack.py` | Δ_A/Δ_B point estimates within ±0.05; Δ_V1 95% bootstrap CI excludes 0 for all four models; range [−2.37, −1.65] |
| Slug exposure stays at floor 0.88–1.63 bits | Table 3 | `analyze_slug.py` | slug exposure range in [0.88, 1.63] |
| Format ablation (training-faithful prompt wins) | Table 4 | `analyze_ablation.py` | report regenerated for the format-spread claim |
| Cross-seed adapter utility F1-macro 0.19–0.28 | Table 5 | `analyze_utility_logits.py` | F1-macro range in [0.19, 0.28] |
| Prompt-eng utility preserved (\|ΔF1\| < 0.05) | Table 6 | `analyze_prompt.py` | 3/3 instruction-tuned base models keep \|ΔF1\| < 0.05 |

## Repository structure

```
.
├── reproduce.sh              Path A: regenerate all reports + verify (no GPU, ~1 min)
├── reproduce_full.sh         Path B: train + eval + analyze from scratch (GPU, ~222 GPU-h)
├── fetch_anonshield.sh       pull the external AnonShield image / clone its repo
├── Dockerfile                CPU image that runs Path A by default
├── pyproject.toml            Python 3.12; [reproduce] / [train] / [figures] extras
├── CITATION.cff
├── LICENSE                   MIT
├── scripts/
│   └── verify_claims.py      asserts all 19 paper numbers; exits 0 iff all pass
├── data/
│   └── mock/cais_mock.json.xz   full synthetic corpus (70,951 records; xz, ~37 MB)
├── cais_mock/
│   └── generator.py          synthetic Tenable-schema dataset generator (provenance)
└── experiment/
    ├── canaries/             Class A + Class B canaries and their variant pools
    ├── configs/              156 training YAMLs (models × protection variants × seeds × forms)
    ├── data/                 train and held-out splits (V0 and Anon)
    ├── scripts/              train.py, eval.py, eval_slug_attack.py, analyze_*.py, ...
    ├── results/              shipped pre-computed eval results
    │                         (eval_checkpoints, eval_slug, utility_*, train_telemetry)
    ├── reports/              Markdown reports regenerated by reproduce.sh
    └── figures/              dose-response, AUC-dilution, and Pareto figures
```

## Security and data

The training data shipped under `experiment/data/` is **fully synthetic** and has **zero overlap** with any real CAIS data on every identifier dimension (IPv4, IPv6, MAC, hostname, asset name, CVE-ID). The injected canaries are synthetic as well. Pseudonymization uses keyed **HMAC-SHA256** over a fixed schema; you supply your own `ANON_SECRET_KEY` when running Path B. No real CAIS data, no real PII, and no secret keys ship in this repository. Slug values produced under a different key will not match bit-for-bit but carry identical memorization semantics, since exposure of cryptographically random hex strings is structurally independent of the key.

## How to cite

If you use this artifact or build on its findings, please cite the paper:

> Cristhian Kapelinski and Diego Kreutz. **Decomposing Memorization Reduction in
> Privacy-Preserving Fine-Tuning of SLMs for CSIRTs.** In *Proceedings of the Brazilian
> Conference on Intelligent Systems (BRACIS 2026)*, Lecture Notes in Computer Science,
> Springer, 2026.

```bibtex
@inproceedings{kapelinski2026decomposing,
  author    = {Kapelinski, Cristhian and Kreutz, Diego},
  title     = {Decomposing Memorization Reduction in Privacy-Preserving Fine-Tuning of {SLMs} for {CSIRTs}},
  booktitle = {Proceedings of the Brazilian Conference on Intelligent Systems (BRACIS 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026}
}
```

A machine-readable `CITATION.cff` is included; GitHub renders a "Cite this repository" button from it.

## License

MIT — see `LICENSE`. Third-party components retain their original licenses (HuggingFace model weights under their respective licenses; the external AnonShield tool under its own license).
