"""Multi-seed analysis (Waves 1+2+3): 48 adapters x 3 seeds, paired n=60 -> experiment/FINAL_REPORT.md."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "experiment" / "results" / "eval_checkpoints"
TELEM_DIR = ROOT / "experiment" / "results" / "train_telemetry"
OUT_MD = ROOT / "experiment" / "FINAL_REPORT.md"

MODELS = ("gemma3-1b", "qwen3-1.7b", "llama32-3b", "vaultgemma-1b")
VARIANTS = ("v0", "v1", "v2", "v3")
SEEDS = (42, 1337, 2024)


def load_canary_exposures() -> dict:
    """Returns {(model, variant, seed): {canary_id: exposure}}."""
    out = {}
    for m in MODELS:
        for v in VARIANTS:
            for s in SEEDS:
                fp = EVAL_DIR / f"{m}_{v}_seed{s}.jsonl"
                if not fp.exists():
                    continue
                rows = [json.loads(l) for l in fp.open()]
                steps = sorted({r.get("step") for r in rows if "step" in r and r.get("kind") == "canary"}, reverse=True)
                if not steps:
                    continue
                last = max(steps)
                canaries = {r["canary_id"]: r["exposure_bits"]
                            for r in rows if r.get("kind") == "canary" and r.get("step") == last}
                if canaries:
                    out[(m, v, s)] = canaries
    return out


def load_aggregates() -> dict:
    """{(model, variant, seed): {exp_*, auc_*}}."""
    out = {}
    for m in MODELS:
        for v in VARIANTS:
            for s in SEEDS:
                fp = EVAL_DIR / f"{m}_{v}_seed{s}.jsonl"
                if not fp.exists():
                    continue
                rows = [json.loads(l) for l in fp.open()]
                steps = sorted({r.get("step") for r in rows if "step" in r}, reverse=True)
                if not steps:
                    continue
                last = max(steps)
                last_rows = [r for r in rows if r.get("step") == last]
                out[(m, v, s)] = {
                    "exp_a": next((r["mean"] for r in last_rows if r.get("kind") == "exposure_agg" and r.get("class") == "A"), float("nan")),
                    "exp_b": next((r["mean"] for r in last_rows if r.get("kind") == "exposure_agg" and r.get("class") == "B"), float("nan")),
                    "exp_all": next((r["mean"] for r in last_rows if r.get("kind") == "exposure_agg" and r.get("class") == "all"), float("nan")),
                    "auc_lc": next((r["value"] for r in last_rows if r.get("kind") == "auc" and r.get("class") == "all" and r.get("metric") == "loss_canary"), float("nan")),
                    "auc_l": next((r["value"] for r in last_rows if r.get("kind") == "auc" and r.get("class") == "all" and r.get("metric") == "loss"), float("nan")),
                }
    return out


def cliff_delta(x, y):
    n_g = sum(1 for a in x for b in y if a > b)
    n_l = sum(1 for a in x for b in y if a < b)
    n = len(x) * len(y) if x and y else 1
    return (n_g - n_l) / n


def wilcoxon_paired_multi_seed(canary, model, var_a, var_b, seeds=SEEDS):
    """Pairs by (canary, seed); returns pooled multi-seed stats."""
    a_vals, b_vals = [], []
    for s in seeds:
        ka = (model, var_a, s)
        kb = (model, var_b, s)
        if ka not in canary or kb not in canary:
            continue
        common = sorted(set(canary[ka]) & set(canary[kb]))
        for c in common:
            a_vals.append(canary[ka][c])
            b_vals.append(canary[kb][c])
    if len(a_vals) < 6:
        return {"n": len(a_vals), "p": float("nan"), "stat": float("nan"), "delta": float("nan"), "median_diff": float("nan")}
    a, b = np.array(a_vals), np.array(b_vals)
    if np.allclose(a, b):
        return {"n": len(a), "p": 1.0, "stat": 0.0, "delta": 0.0, "median_diff": 0.0}
    try:
        res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="approx")
        return {
            "n": len(a),
            "stat": float(res.statistic),
            "p": float(res.pvalue),
            "delta": cliff_delta(list(a), list(b)),
            "median_diff": float(np.median(a - b)),
            "mean_diff": float(np.mean(a - b)),
        }
    except ValueError:
        return {"n": len(a), "p": float("nan"), "stat": float("nan"), "delta": float("nan"), "median_diff": float("nan")}


def wilcoxon_cross_model(canary, m_a, var_a, m_b, var_b, seeds=SEEDS):
    """H8 cross-model: pairs by (canary, seed) across two models."""
    a_vals, b_vals = [], []
    for s in seeds:
        ka = (m_a, var_a, s)
        kb = (m_b, var_b, s)
        if ka not in canary or kb not in canary:
            continue
        common = sorted(set(canary[ka]) & set(canary[kb]))
        for c in common:
            a_vals.append(canary[ka][c])
            b_vals.append(canary[kb][c])
    if len(a_vals) < 6:
        return {"n": len(a_vals), "p": float("nan"), "stat": float("nan"), "delta": float("nan"), "median_diff": float("nan")}
    a, b = np.array(a_vals), np.array(b_vals)
    if np.allclose(a, b):
        return {"n": len(a), "p": 1.0, "stat": 0.0, "delta": 0.0, "median_diff": 0.0}
    try:
        res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="approx")
        return {
            "n": len(a),
            "stat": float(res.statistic),
            "p": float(res.pvalue),
            "delta": cliff_delta(list(a), list(b)),
            "median_diff": float(np.median(a - b)),
            "mean_diff": float(np.mean(a - b)),
        }
    except ValueError:
        return {"n": len(a), "p": float("nan"), "stat": float("nan"), "delta": float("nan"), "median_diff": float("nan")}


def friedman_4way(canary, model, seeds=SEEDS):
    """Friedman pooled multi-seed."""
    arrays = {v: [] for v in VARIANTS}
    common_per_seed = {}
    for s in seeds:
        keys = {v: (model, v, s) for v in VARIANTS}
        if not all(k in canary for k in keys.values()):
            continue
        common = sorted(set.intersection(*(set(canary[k]) for k in keys.values())))
        common_per_seed[s] = common
        for v in VARIANTS:
            arrays[v].extend(canary[keys[v]][c] for c in common)
    n = len(arrays[VARIANTS[0]])
    if n < 6:
        return {"n": n, "p": float("nan"), "stat": float("nan")}
    try:
        res = stats.friedmanchisquare(*[np.array(arrays[v]) for v in VARIANTS])
        return {"n": n, "stat": float(res.statistic), "p": float(res.pvalue)}
    except ValueError:
        return {"n": n, "p": float("nan"), "stat": float("nan")}


def bh_fdr(pvals, q=0.05):
    p = np.array(pvals)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, m + 1) / m) * q
    below = ranked <= thresh
    if not below.any():
        return [False] * m
    k = np.where(below)[0].max()
    cutoff = ranked[k]
    return [bool(pv <= cutoff) for pv in p]


def fmt_p(p):
    if math.isnan(p): return "-"
    if p < 0.001: return f"{p:.2e}"
    return f"{p:.3f}"


def main():
    canary = load_canary_exposures()
    agg = load_aggregates()
    if not canary:
        print("error: no evals", file=sys.stderr)
        return 1

    seeds_avail = sorted({s for (_, _, s) in canary})
    print(f"seeds available: {seeds_avail}")

    L = ["# Experiment 1 - Final Report (Multi-Seed)\n"]
    L.append(f"**Seeds:** {seeds_avail}  -  **Adapters:** {len([k for k in canary if canary[k]])}/{len(MODELS)*len(VARIANTS)*len(SEEDS)} expected\n")
    n_pooled = 20 * len(seeds_avail)
    L.append(f"> Multi-seed: paired tests by (canary, seed) -> n_pooled = 20 x {len(seeds_avail)} = **{n_pooled}**.\n")
    L.append("---\n")

    L.append("## 1. Aggregate metrics (mean +/- std cross-seed)\n")
    L.append("| model | variant | exp_all (mean+/-std) | AUC_loss_canary (mean+/-std) |")
    L.append("|--------|----------|--------------------:|--------------------------:|")
    for m in MODELS:
        for v in VARIANTS:
            vals_exp = [agg[(m, v, s)]["exp_all"] for s in seeds_avail if (m, v, s) in agg]
            vals_auc = [agg[(m, v, s)]["auc_lc"] for s in seeds_avail if (m, v, s) in agg]
            if not vals_exp:
                continue
            exp_str = f"{np.mean(vals_exp):.2f} +/- {np.std(vals_exp, ddof=1) if len(vals_exp)>1 else 0:.2f}"
            auc_str = f"{np.mean(vals_auc):.3f} +/- {np.std(vals_auc, ddof=1) if len(vals_auc)>1 else 0:.3f}"
            L.append(f"| {m:17s} | {v:>8s} | {exp_str:>18s} | {auc_str:>26s} |")
    L.append("")

    L.append("## 2. Paired Wilcoxon by (canary x seed) - intra-model\n")
    L.append("| Hypothesis | Model | n | mean_diff | median_diff | stat | p (raw) | Cliff delta |")
    L.append("|----------|--------|--:|----------:|------------:|-----:|--------:|--------:|")
    pairs_intra = [
        ("H6'  V1-V0", "v1", "v0"),
        ("H6   V2-V1", "v2", "v1"),
        ("H6e2 V3-V1", "v3", "v1"),
        ("H9   V3-V2", "v3", "v2"),
    ]
    pair_results = []
    for h, va, vb in pairs_intra:
        for m in MODELS:
            r = wilcoxon_paired_multi_seed(canary, m, va, vb, seeds_avail)
            pair_results.append((h, m, va, vb, r))

    h8_pairs = [
        ("H8a  vault-V0 vs gemma3-V0", "vaultgemma-1b", "v0", "gemma3-1b", "v0", True),
        ("H8a' vault-V1 vs gemma3-V1", "vaultgemma-1b", "v1", "gemma3-1b", "v1", True),
        ("H8b  vault V2 vs V0", "vaultgemma-1b", "v2", "vaultgemma-1b", "v0", False),
        ("H8b' vault V2 vs V1", "vaultgemma-1b", "v2", "vaultgemma-1b", "v1", False),
        ("H8c  vault V3 vs V2", "vaultgemma-1b", "v3", "vaultgemma-1b", "v2", False),
    ]
    for h, ma, va, mb, vb, cross in h8_pairs:
        if cross:
            r = wilcoxon_cross_model(canary, ma, va, mb, vb, seeds_avail)
        else:
            r = wilcoxon_paired_multi_seed(canary, ma, va, vb, seeds_avail)
        pair_results.append((h, f"{ma} / {mb}" if cross else ma, va, vb, r))

    pvals = [r["p"] if not math.isnan(r["p"]) else 1.0 for *_, r in pair_results]
    rejected = bh_fdr(pvals, q=0.05)

    intra_n = len(pairs_intra) * len(MODELS)
    for i, ((h, m_or_pair, va, vb, r), rej) in enumerate(zip(pair_results, rejected)):
        if i == intra_n:
            L.append("")
            L.append("## 3. Paired Wilcoxon by (canary x seed) - H8 (defense in depth)\n")
            L.append("| Hypothesis | Models | n | mean_diff | median_diff | stat | p (raw) | Cliff delta |")
            L.append("|----------|---------|--:|----------:|------------:|-----:|--------:|--------:|")
        sig = " *" if rej else ""
        md = r.get("mean_diff", float("nan"))
        L.append(f"| {h:30s} | {m_or_pair:35s} | {r['n']:>3d} | {md:+.3f} | {r.get('median_diff', float('nan')):+.3f} | {r['stat']:>6.1f} | {fmt_p(r['p']):>7s} | {r['delta']:+.3f}{sig} |")
    L.append("")

    L.append("## 4. Friedman 4-way (V0/V1/V2/V3) per model (pooled multi-seed)\n")
    L.append("| Model | n | chi2 | p (raw) |")
    L.append("|--------|--:|----:|--------:|")
    for m in MODELS:
        r = friedman_4way(canary, m, seeds_avail)
        L.append(f"| {m:17s} | {r['n']:>3d} | {r['stat']:.2f} | {fmt_p(r['p']):>7s} |")
    L.append("")

    L.append("## 5. Hypothesis summary\n")
    L.append("BH-FDR q=0.05 over all paired p-values above.\n")
    L.append("| Hypothesis | Sig. + median_diff < 0 in models |")
    L.append("|----------|-------------------------------------|")
    for h_label in ["H6'", "H6", "H6e2", "H9", "H8a", "H8a'", "H8b", "H8b'", "H8c"]:
        passed = []
        for (h, m, _, _, r), rej in zip(pair_results, rejected):
            if h.startswith(h_label) and rej and r.get("mean_diff", 0) < 0:
                passed.append(m.split('/')[0].strip())
        L.append(f"| {h_label:8s} | {', '.join(set(passed)) if passed else '-'} |")
    L.append("")

    L.append("## 6. Notes\n")
    L.append(f"- **Multi-seed pooling**: each pair has n = 20 canaries x {len(seeds_avail)} seeds = {n_pooled} points. Inference gains power vs single-seed (W1).")
    L.append("- BH-FDR controls family-wise error via FDR over **all** reported p-values.")
    L.append("- Negative Cliff delta = first variant < second (i.e., exposure reduction).")
    L.append("- Expected direction: H6'/H6/H6e2/H9/H8a/b/c all predict `mean_diff < 0`.")

    OUT_MD.write_text("\n".join(L))
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
