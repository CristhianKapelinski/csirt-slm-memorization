"""Stack ablation (Sec 4.4): batching (A) vs NF4 (B) → reports/ABLATION_STACK_REPORT.md."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "experiment" / "results" / "eval_checkpoints"
OUT = ROOT / "experiment" / "reports" / "ABLATION_STACK_REPORT.md"

MODELS = ["gemma3-1b", "qwen3-1.7b", "llama32-3b", "vaultgemma-1b"]
SEEDS = [42, 1337, 2024]
N_BOOT = 10000
CELLS = [("A", "va"), ("B", "vb"), ("V1", "v1")]


def _last_rows(model: str, variant: str, seed: int) -> list[dict] | None:
    path = EVAL / f"{model}_{variant}_seed{seed}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    steps = [r["step"] for r in rows if "step" in r]
    if not steps:
        return None
    last = max(steps)
    return [r for r in rows if r.get("step") == last]


def adapter_exposure(model: str, variant: str, seed: int) -> float | None:
    rows = _last_rows(model, variant, seed)
    if rows is None:
        return None
    return next((r["mean"] for r in rows if r.get("kind") == "exposure_agg" and r.get("class") == "all"), None)


def per_canary(model: str, variant: str, seed: int) -> dict[str, float]:
    rows = _last_rows(model, variant, seed) or []
    return {r["canary_id"]: r["exposure_bits"] for r in rows if r.get("kind") == "canary"}


def cross_seed(model: str, variant: str) -> float | None:
    vals = [adapter_exposure(model, variant, s) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def paired_diffs(model: str, variant: str) -> np.ndarray:
    diffs = []
    for seed in SEEDS:
        treat, base = per_canary(model, variant, seed), per_canary(model, "v0", seed)
        diffs += [treat[c] - base[c] for c in treat.keys() & base.keys()]
    return np.array(diffs)


def bootstrap_ci(diffs: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    means = diffs[rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    lines = [
        "# ABLATION_STACK_REPORT - stack decomposition (Sec 4.4)\n",
        "Delta = Exp(cell) - Exp(V0), cross-seed (n=3). A = large-lot batching "
        "(lot 32, bf16), B = NF4 quantization (lot 4, NF4), V1 = full stack. "
        f"95% CIs from a paired bootstrap over 60 (canary x seed) observations (B={N_BOOT}).\n",
        "| Model | Delta_A [95% CI] | Delta_B [95% CI] | Delta_V1 [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    dv1 = []
    for model in MODELS:
        e0 = cross_seed(model, "v0")
        cells = []
        for _name, variant in CELLS:
            point = cross_seed(model, variant) - e0
            lo, hi = bootstrap_ci(paired_diffs(model, variant))
            cells.append(f"{point:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        dv1.append(cross_seed(model, "v1") - e0)
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Delta_V1 range across models: [{min(dv1):.2f}, {max(dv1):.2f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"[ablation] Delta_V1 range [{min(dv1):.2f}, {max(dv1):.2f}]")


if __name__ == "__main__":
    main()
