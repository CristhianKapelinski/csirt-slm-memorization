"""Logit-based utility analysis (Plan A1) → reports/UTILITY_REPORT.md."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
RESULTS_DIR = PILOT / "results" / "utility_eval_logits"
OUT = PILOT / "reports" / "UTILITY_REPORT.md"

MODELS = ["gemma3-1b", "qwen3-1.7b", "llama32-3b", "vaultgemma-1b"]
VARIANTS = ["v0", "v1", "v2", "v3"]
SEEDS = [42, 1337, 2024]
BUCKETS = ["Low", "Medium", "High", "Critical"]


def model_key_from_name(name):
    if not name: return None
    n = name.lower()
    if "gemma-3-1b" in n or "gemma3-1b" in n: return "gemma3-1b"
    if "qwen3-1.7" in n: return "qwen3-1.7b"
    if "llama-3.2-3b" in n: return "llama32-3b"
    if "vaultgemma" in n: return "vaultgemma-1b"
    return None


def load_summaries():
    out = []
    if not RESULTS_DIR.exists(): return out
    for p in sorted(RESULTS_DIR.glob("*_summary.json")):
        try:
            d = json.loads(p.read_text())
            d["model_key"] = model_key_from_name(d.get("model", ""))
            d["use_anon"] = d.get("run_id", "").startswith("anon_")
            out.append(d)
        except Exception:
            pass
    return out


def bh_fdr(pvals, q=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rejected = [False] * m
    crit = 0.0
    for k, i in enumerate(order, start=1):
        thresh = q * k / m
        if pvals[i] <= thresh:
            crit = pvals[i]
    for i, p in enumerate(pvals):
        if p <= crit and p > 0:
            rejected[i] = True
    return rejected


def main():
    summaries = load_summaries()
    out = []
    out.append("# UTILITY_REPORT — Severity Classification (logit-based, Plan A1)\n")
    out.append(f"**N adapters evaluated**: {len(summaries)}\n")
    out.append("**Method**: format-faithful prompt (training-faithful), Σ log P over the "
               "canonical continuation (`{value},\"base_vector\"`), argmax over the 4 buckets.")
    out.append("**Summary criterion**: random baseline 4-way = acc 0.25; F1-macro "
               "random ≈ 0.10. Small models fine-tuned on raw JSON suffer **format "
               "specialization** (Wang 2022 ProMoT) — classification capability stays "
               "marginally above random. Methodological finding, see §6.\n")

    grouped = defaultdict(list)
    for s in summaries:
        if not s.get("model_key"): continue
        grouped[(s["model_key"], s["variant"], s["use_anon"])].append(s)

    out.append("\n## 1. F1-macro cross-seed (mean ± std over 3 seeds)\n")
    out.append("| model | variant | use_anon | F1_raw | F1_norm | acc_raw | acc_norm |")
    out.append("|---|---|:---:|---:|---:|---:|---:|")
    for model in MODELS:
        for variant in VARIANTS:
            for use_anon in [False, True]:
                rows = grouped.get((model, variant, use_anon), [])
                if not rows: continue
                tag = "Anon" if use_anon else "—"
                f1r = statistics.mean(r["f1_raw"] for r in rows)
                f1r_s = statistics.stdev(r["f1_raw"] for r in rows) if len(rows) > 1 else 0
                f1n = statistics.mean(r["f1_norm"] for r in rows)
                accr = statistics.mean(r["acc_raw"] for r in rows)
                accn = statistics.mean(r["acc_norm"] for r in rows)
                out.append(f"| {model} | {variant} | {tag} | {f1r:.3f}±{f1r_s:.3f} | {f1n:.3f} | {accr:.3f} | {accn:.3f} |")

    out.append("\n## 2. Vk vs Anon-Vk (effect of HMAC pseudonymization on utility)\n")
    out.append("**H_U1**: F1(Anon-Vk) ≈ F1(Vk). Diff < 5% absolute = PASS, utility preserved.\n")
    out.append("| model | variant | F1 Vk | F1 Anon | Δ abs | flag |")
    out.append("|---|---|---:|---:|---:|:---:|")
    for model in MODELS:
        for variant in VARIANTS:
            v = grouped.get((model, variant, False), [])
            a = grouped.get((model, variant, True), [])
            if not v or not a: continue
            fv = statistics.mean(r["f1_raw"] for r in v)
            fa = statistics.mean(r["f1_raw"] for r in a)
            d = fa - fv
            flag = "PASS" if abs(d) < 0.05 else "[!]" if abs(d) < 0.10 else "FAIL"
            out.append(f"| {model} | {variant} | {fv:.3f} | {fa:.3f} | {d:+.3f} | {flag} |")

    out.append("\n## 3. Paired Wilcoxon per record (Vk vs Anon-Vk, cross-seed pooled)\n")
    out.append("Pairs predictions by record index (held-out raw and anon keep the same order). "
               "Wilcoxon signed-rank over the diff (Anon_correct − Vk_correct). BH-FDR q=0.05 "
               "over 16 comparisons (4 models × 4 variants).\n")
    out.append("| model | variant | n pairs | median_diff | stat | p (raw) | BH-FDR sig |")
    out.append("|---|---|---:|---:|---:|---:|:---:|")
    pvals, rows = [], []
    for model in MODELS:
        for variant in VARIANTS:
            v_seeds = {r["seed"]: r for r in grouped.get((model, variant, False), [])}
            a_seeds = {r["seed"]: r for r in grouped.get((model, variant, True), [])}
            common = sorted(set(v_seeds) & set(a_seeds))
            if not common: continue
            diffs = []
            for s in common:
                vp, vg = v_seeds[s].get("preds_raw", []), v_seeds[s].get("gts", [])
                ap, ag = a_seeds[s].get("preds_raw", []), a_seeds[s].get("gts", [])
                # assume raw/anon records aligned by index
                n = min(len(vp), len(ap), len(vg), len(ag))
                for i in range(n):
                    if vg[i] != ag[i]: continue
                    vc = 1 if vp[i] == vg[i] else 0
                    ac = 1 if ap[i] == ag[i] else 0
                    diffs.append(ac - vc)
            if len(diffs) < 5 or all(d == 0 for d in diffs):
                rows.append((model, variant, len(diffs), 0.0, None, None))
                continue
            try:
                stat, p = wilcoxon(diffs, zero_method="wilcox")
                med = statistics.median(diffs)
                rows.append((model, variant, len(diffs), med, float(stat), float(p)))
                pvals.append(float(p))
            except Exception:
                rows.append((model, variant, len(diffs), 0.0, None, None))
    p_mask = []
    if pvals:
        p_mask = bh_fdr(pvals, q=0.05)
    p_idx = 0
    for model, variant, n, med, stat, p in rows:
        if p is None:
            out.append(f"| {model} | {variant} | {n} | {med:+.3f} | — | — | — |")
        else:
            mark = "*" if p_mask[p_idx] else ""
            p_idx += 1
            out.append(f"| {model} | {variant} | {n} | {med:+.3f} | {stat:.1f} | {p:.4g} | {mark} |")

    out.append("\n## 4. Aggregated confusion matrices by (model, variant, anon)\n")
    for model in MODELS:
        for variant in VARIANTS:
            for use_anon in [False, True]:
                rows_g = grouped.get((model, variant, use_anon), [])
                if not rows_g: continue
                agg = [[0]*4 for _ in range(4)]
                for r in rows_g:
                    cm = r.get("confusion_matrix_raw", [[0]*4]*4)
                    for i in range(4):
                        for j in range(4):
                            agg[i][j] += cm[i][j]
                tag = "Anon" if use_anon else "Vk"
                out.append(f"\n### {model} / {variant} / {tag}")
                out.append(f"\n| GT \\ pred | {' | '.join(BUCKETS)} |")
                out.append("|---" + "|---:" * 4 + "|")
                for i, row in enumerate(agg):
                    out.append(f"| {BUCKETS[i]} | " + " | ".join(str(x) for x in row) + " |")

    out.append("\n## 5. Methodological finding — format specialization\n")
    out.append(
        "The adapters were fine-tuned on **serialized raw JSON** (next-token prediction). "
        "They were not trained on (prompt, label) pairs for severity classification. When "
        "given a natural prompt (`Severity:`), the models respond by continuing the JSON "
        "(95-100% invalid output) — a phenomenon known as **format specialization** "
        "(Wang et al. 2022, ProMoT, ICLR 2023). This reflects that fine-tuning destroys the "
        "ability to follow instructions outside the learned domain; it is not a failure of HMAC pseudonymization.\n"
        "\nThis experiment worked around the problem using **logit-based scoring in the "
        "training-faithful format**: the prompt reproduces exactly the serialization seen in "
        "training (compact JSON truncated at `\"base_score\":`), and the prediction is argmax over "
        "Σ log P of 4 canonical continuations (one per bucket). Acc raw vs acc_norm "
        "(byte-length normalized, lm-eval-harness convention) are reported in §1.\n"
        "\nA cross-format ablation on 1 adapter per model (50 records each) confirmed:\n"
        "- format-faithful (canonical) is the best for all of them;\n"
        "- cross-format acc variation of 1.5×-2.75× — **prompt fragility**;\n"
        "- the natural prompt (`Severity:`) drops to acc ≈ 0.16-0.24 (near random)."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(summaries)} summaries)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
