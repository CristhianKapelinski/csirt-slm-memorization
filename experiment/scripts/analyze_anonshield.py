"""Exp 2 / Table 2: HMAC pseudonymization (AnonShield) reduces V0 canary exposure → ANONSHIELD_REPORT.md."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "experiment" / "results" / "eval_checkpoints"
OUT = ROOT / "experiment" / "reports" / "ANONSHIELD_REPORT.md"

MODELS = ["gemma3-1b", "qwen3-1.7b", "llama32-3b", "vaultgemma-1b"]
SEEDS = [42, 1337, 2024]


def pooled_exposure(prefix: str, model: str) -> float | None:
    vals = []
    for seed in SEEDS:
        path = EVAL / f"{prefix}{model}_v0_seed{seed}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "canary":
                vals.append(rec["exposure_bits"])
    return statistics.fmean(vals) if vals else None


def main() -> None:
    lines = [
        "# ANONSHIELD_REPORT - Table 2 (Exp 2: HMAC pseudonymization)\n",
        "Reduction = (Exp(V0) - Exp(Anon-V0)) / Exp(V0), pooled per-canary mean "
        "exposure across seeds 42/1337/2024.\n",
        "| Model | Exp(V0) | Exp(Anon-V0) | reduction |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        raw = pooled_exposure("", model)
        anon = pooled_exposure("anon_", model)
        red = (raw - anon) / raw * 100
        lines.append(f"| {model} | {raw:.3f} | {anon:.3f} | +{red:.1f}% |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("[anonshield] Table 2 AnonShield-V0 reductions written")


if __name__ == "__main__":
    main()
