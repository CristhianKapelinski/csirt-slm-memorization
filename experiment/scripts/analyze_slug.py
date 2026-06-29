"""Attack 2 (slug exposure) analysis → reports/SLUG_ATTACK_REPORT.md."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
SLUG_DIR = PILOT / "results" / "eval_slug"
ORIG_DIR = PILOT / "results" / "eval_checkpoints"
OUT = PILOT / "reports" / "SLUG_ATTACK_REPORT.md"

MODELS = ["gemma3-1b", "qwen3-1.7b", "llama32-3b", "vaultgemma-1b"]
VARIANTS = ["v0", "v1", "v2", "v3"]
SEEDS = [42, 1337, 2024]
BUCKETS = ["A", "B"]


def model_short(name: str) -> str | None:
    n = name.lower()
    if "gemma-3-1b" in n: return "gemma3-1b"
    if "qwen3-1.7" in n: return "qwen3-1.7b"
    if "llama-3.2-3b" in n: return "llama32-3b"
    if "vaultgemma" in n: return "vaultgemma-1b"
    return None


def load_records(p: Path) -> list[dict]:
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def aggregate(dir_: Path, kind_filter: str) -> dict[tuple[str, str], list[float]]:
    """Returns {(model, variant): [mean_exposure_per_seed]} for kind in jsonl."""
    out = defaultdict(list)
    if not dir_.exists(): return out
    for p in sorted(dir_.glob("*.jsonl")):
        rid = p.stem
        if not rid.startswith("anon_"):
            # also accept non-anon for comparison (Attack 1)
            pass
        recs = load_records(p)
        meta = next((r for r in recs if "model" in r and "variant" in r), None)
        if not meta: continue
        m = model_short(meta["model"])
        v = meta.get("variant")
        if not m or not v: continue
        agg = next((r for r in recs if r.get("kind") == kind_filter and r.get("class") == "all"), None)
        if agg and agg.get("mean") is not None:
            out[(m, v)].append(float(agg["mean"]))
    return out


def main():
    slug_data = aggregate(SLUG_DIR, "slug_exposure_agg")
    orig_anon_data = aggregate(ORIG_DIR, "exposure_agg")

    # re-read filtered to anon_ runs (Exp 2 only)
    orig_anon_filtered = defaultdict(list)
    for p in sorted(ORIG_DIR.glob("anon_*.jsonl")):
        recs = load_records(p)
        meta = next((r for r in recs if "model" in r and "variant" in r), None)
        if not meta: continue
        m, v = model_short(meta["model"]), meta.get("variant")
        if not m or not v: continue
        agg = next((r for r in recs if r.get("kind") == "exposure_agg" and r.get("class") == "all"), None)
        if agg and agg.get("mean") is not None:
            orig_anon_filtered[(m, v)].append(float(agg["mean"]))

    out = ["# SLUG_ATTACK_REPORT — Attack 2 (HMAC slug exposure) on Anon adapters\n"]
    out.append("**H_A2**: the Anon model memorizes the **HMAC slug** (which it saw 30× during "
               "training), not the original canary. We expect exposure(slug) > exposure(original) "
               "in Exp 2 — completing the *dual extraction attack* promised in §3.2/§5.\n")
    out.append("**Method**: for each Anon-Vk, apply HMAC pseudonymization with the same HMAC key to "
               "both the canary and the 100 variants in the pool, build the planted/variant records "
               "with slugged values, and compute exposure (Carlini) over the slug tokens. Metric: "
               "$\\mathrm{exposure} = \\log_2(N{+}1) - \\log_2(\\text{rank})$, $N{=}100$.\n")

    out.append("\n## 1. Mean exposure(slug) cross-seed by (model, variant)\n")
    out.append("Mean ± std over 3 seeds. Random baseline ≈ 1.0 bit ($\\log_2(101/50)$).\n")
    out.append("| Model | Anon-V0 | Anon-V1 | Anon-V2 ($\\varepsilon{=}8$) | Anon-V3 ($\\varepsilon{=}2$) |")
    out.append("|---|---:|---:|---:|---:|")
    for m in MODELS:
        cells = []
        for v in VARIANTS:
            vals = slug_data.get((m, v), [])
            if not vals:
                cells.append("—")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3f}")
            else:
                mn = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                cells.append(f"{mn:.3f}±{sd:.3f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")

    out.append("\n## 2. Attack 1 (original) vs Attack 2 (slug) — H_A2 confirmation\n")
    out.append("Diff > 0 and materially large = the model memorizes the slug, not the original "
               "→ confirms that HMAC pseudonymization removed the original canary but the learned slug "
               "still carries the memorization signal (as expected).\n")
    out.append("| Model | Variant | exp(orig) | exp(slug) | Δ (slug − orig) |")
    out.append("|---|---|---:|---:|---:|")
    for m in MODELS:
        for v in VARIANTS:
            o = orig_anon_filtered.get((m, v), [])
            s = slug_data.get((m, v), [])
            if not o or not s:
                continue
            mo = statistics.mean(o)
            ms = statistics.mean(s)
            d = ms - mo
            flag = "++" if d > 1.0 else "+" if d > 0 else "—"
            out.append(f"| {m} | {v} | {mo:.3f} | {ms:.3f} | {d:+.3f} {flag} |")

    out.append("\n*++ = Δ > 1 bit (slug memorization clearly above the original).*\n")
    out.append("*+ = slug exposure ≥ original exposure (expected under H_A2).*\n")

    out.append("\n## 3. Interpretation\n")
    out.append("- **Anon-V0 (no extra protection)** is where the slug memorization signal is "
               "most visible — the model saw the slug 30× with neither DP-SGD regularization nor "
               "Poisson dilution. Expected: high Δ.\n"
               "- **Anon-V1** (QLoRA + lot=32 Poisson) and **Anon-V2/V3** (DP-SGD) already have "
               "memorization attenuated in the same proportion that reduces exposure(original) in "
               "Exp 1 — Δ should be small but positive.\n"
               "- If Δ ≈ 0 or negative in some slot, it indicates the HMAC slug behaves like a "
               "random string to the model (no learned co-occurrence) — additional evidence that "
               "HMAC pseudonymization + DP-SGD blocks syntactic memorization.\n"
               "- **Methodological comparison**: Attack 1 tests whether the model would leak the "
               "real identifier (relevant to the adversary); Attack 2 tests whether the model "
               "closed a \"shortcut\" to the slug (relevant for re-identification if the mapping "
               "leaked). The two address complementary threats.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
