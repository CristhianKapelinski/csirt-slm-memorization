"""Finding (i) / C1: matched-update control → reports/MATCHED_UPDATE_REPORT.md."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "experiment" / "results" / "eval_checkpoints"
OUT = ROOT / "experiment" / "reports" / "MATCHED_UPDATE_REPORT.md"

MODELS = ["gemma3-1b", "qwen3-1.7b", "llama32-3b", "vaultgemma-1b"]
SEEDS = [42, 1337, 2024]


def adapter_exposure(model: str, variant: str, seed: int) -> float | None:
    path = EVAL / f"{model}_{variant}_seed{seed}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    steps = [r["step"] for r in rows if "step" in r]
    if not steps:
        return None
    last = max(steps)
    for r in rows:
        if r.get("step") == last and r.get("kind") == "exposure_agg" and r.get("class") == "all":
            return r["mean"]
    return None


def cross_seed(model: str, variant: str) -> float | None:
    vals = [adapter_exposure(model, variant, s) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def main() -> None:
    by_model: dict[str, float] = {}
    deltas: list[float] = []
    for model in MODELS:
        e0, e1, efs = cross_seed(model, "v0"), cross_seed(model, "v1"), cross_seed(model, "v0fs")
        if None in (e0, e1, efs) or e0 == e1:
            continue
        by_model[model] = (e0 - efs) / (e0 - e1)
        deltas.append(e1 - e0)

    mean_frac = statistics.fmean(by_model.values())
    mean_delta = statistics.fmean(deltas)
    lo, hi = min(by_model.values()), max(by_model.values())

    lines = [
        "# MATCHED_UPDATE_REPORT - finding (i) / C1\n",
        "Recovered fraction = (Exp(V0) - Exp(V0fs)) / (Exp(V0) - Exp(V1)), cross-seed "
        "(seeds 42/1337/2024, n=3). V0fs = V0 truncated to ~300 optimizer updates. A "
        "fraction near 1.0 means the V0->V1 memorization reduction is driven by fewer "
        "optimizer updates, not by DP-SGD.\n",
        "| Model | recovered fraction |",
        "|---|---:|",
    ]
    for model in MODELS:
        frac = by_model.get(model)
        lines.append(f"| {model} | {frac * 100:.0f}% |" if frac is not None else f"| {model} | - |")
    lines.append(f"| **mean** | **{mean_frac * 100:.0f}%** |")
    lines.append("")
    lines.append(f"Range: {lo * 100:.0f}%-{hi * 100:.0f}%")
    lines.append(f"Mean Delta = Exp(V1) - Exp(V0): {mean_delta:.2f} bits")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"[matched-update] mean {mean_frac * 100:.0f}%, range {lo * 100:.0f}-{hi * 100:.0f}%, "
          f"mean Delta {mean_delta:.2f} bits")


if __name__ == "__main__":
    main()
