"""Exp 4 utility-prompt analysis → reports/UTILITY_PROMPT_REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
RESULTS = PILOT / "results" / "utility_prompt"
OUT = PILOT / "reports" / "UTILITY_PROMPT_REPORT.md"

MODELS = ["gemma3-1b", "qwen3-1.7b", "llama32-3b", "vaultgemma-1b"]
BUCKETS = ["Low", "Medium", "High", "Critical"]


def load(model, version):
    p = RESULTS / f"{model}_{version}_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_records(model, version):
    p = RESULTS / f"{model}_{version}.jsonl"
    if not p.exists(): return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def bh_fdr(pvals, q=0.05):
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    rejected = [False] * m
    crit = 0.0
    for k, i in enumerate(order, start=1):
        if pvals[i] <= q * k / m:
            crit = pvals[i]
    return [(p <= crit and p > 0) for p in pvals]


def main():
    out = []
    out.append("# UTILITY_PROMPT_REPORT — Few-shot base models (Exp 4)\n")
    out.append("**Method**: 4 base models × 2 versions (raw / anon) × 600 held-out records, "
               "few-shot (4 balanced examples Low/Medium/High/Critical), greedy decoding "
               "max_new_tokens=4, case-insensitive parsing.")
    out.append("**Question**: does HMAC pseudonymization destroy information in the *data* (not in the adapters)? "
               "Replicates the Severo 2025 / Almeida 2025 protocol.")
    out.append("**Hypothesis H_U1 (this experiment)**: F1(model, anon) ≈ F1(model, raw). "
               "Diff < 5% absolute = PASS, utility preserved.\n")

    out.append("\n## 1. F1-macro raw vs anon per model\n")
    out.append("| model | F1 raw | F1 anon | Δ abs | Δ rel | acc raw | acc anon | inv raw | inv anon | flag |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    rows = []
    for m in MODELS:
        r = load(m, "raw"); a = load(m, "anon")
        if not r or not a: continue
        f1r, f1a = r["f1_macro"], a["f1_macro"]
        d = f1a - f1r
        rel = (d / f1r * 100) if f1r else 0
        flag = "PASS" if abs(d) < 0.05 else "[!]" if abs(d) < 0.10 else "FAIL"
        if r.get("invalid_rate", 0) > 0.5 or a.get("invalid_rate", 0) > 0.5:
            flag = "(i)"
        out.append(f"| {m} | {f1r:.3f} | {f1a:.3f} | {d:+.3f} | {rel:+.1f}% | "
                   f"{r['accuracy']:.3f} | {a['accuracy']:.3f} | "
                   f"{r['invalid_rate']*100:.1f}% | {a['invalid_rate']*100:.1f}% | {flag} |")
        rows.append((m, r, a))

    out.append("\n*Criterion `(i)`*: invalid_rate > 50% — the model does not follow the prompt "
               "template, so F1 does not measure classification capability (vault-1b is "
               "DP-pretrained, not full instruction-tuned).\n")

    out.append("\n## 2. Paired Wilcoxon per record (raw vs anon)\n")
    out.append("Same record_id in both versions → exact pairing. Diff = "
               "(anon_correct − raw_correct) ∈ {−1, 0, +1}. BH-FDR q=0.05 over 4 models.\n")
    out.append("| model | n pairs | median_diff | agreement (anon=raw) | n raw_only | n anon_only | stat | p (raw) | BH-FDR sig |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|:---:|")
    pvals_w, w_rows = [], []
    for m, r, a in rows:
        recs_r = {x["record_id"]: x for x in load_records(m, "raw")}
        recs_a = {x["record_id"]: x for x in load_records(m, "anon")}
        common = set(recs_r) & set(recs_a)
        diffs = []
        n_concord = n_raw_only = n_anon_only = 0
        for rid in common:
            r_correct = 1 if recs_r[rid]["prediction"] == recs_r[rid]["ground_truth"] else 0
            a_correct = 1 if recs_a[rid]["prediction"] == recs_a[rid]["ground_truth"] else 0
            d = a_correct - r_correct
            diffs.append(d)
            if d == 0: n_concord += 1
            elif d == -1: n_raw_only += 1
            elif d == 1: n_anon_only += 1
        if len(diffs) < 5 or all(d == 0 for d in diffs):
            w_rows.append((m, len(diffs), 0.0, n_concord, n_raw_only, n_anon_only, None, None))
            continue
        try:
            stat, p = wilcoxon(diffs, zero_method="wilcox")
            import statistics
            w_rows.append((m, len(diffs), statistics.median(diffs), n_concord, n_raw_only, n_anon_only, float(stat), float(p)))
            pvals_w.append(float(p))
        except Exception:
            w_rows.append((m, len(diffs), 0.0, n_concord, n_raw_only, n_anon_only, None, None))
    p_mask = bh_fdr(pvals_w, 0.05) if pvals_w else []
    pi = 0
    for m, n, med, nc, nr, na, st, p in w_rows:
        if p is None:
            out.append(f"| {m} | {n} | {med:+.3f} | {nc} | {nr} | {na} | — | — | — |")
        else:
            mark = "*" if p_mask[pi] else ""
            pi += 1
            out.append(f"| {m} | {n} | {med:+.3f} | {nc} | {nr} | {na} | {st:.1f} | {p:.4g} | {mark} |")

    out.append("\n## 3. Per-class F1 raw vs anon\n")
    out.append("| model | version | Low | Medium | High | Critical |")
    out.append("|---|---|---:|---:|---:|---:|")
    for m, r, a in rows:
        for label, summ in [("raw", r), ("anon", a)]:
            pc = summ.get("per_class_f1", {})
            out.append(f"| {m} | {label} | {pc.get('Low',0):.3f} | {pc.get('Medium',0):.3f} | "
                       f"{pc.get('High',0):.3f} | {pc.get('Critical',0):.3f} |")

    out.append("\n## 4. Confusion matrices\n")
    for m, r, a in rows:
        for label, summ in [("raw", r), ("anon", a)]:
            cm = summ.get("confusion_matrix", [[0]*4]*4)
            out.append(f"\n### {m} / {label}")
            out.append(f"\n| GT \\ pred | {' | '.join(BUCKETS)} |")
            out.append("|---" + "|---:" * 4 + "|")
            for i, row in enumerate(cm):
                out.append(f"| {BUCKETS[i]} | " + " | ".join(str(x) for x in row) + " |")

    out.append("\n## 5. Notes and limitations\n")
    out.append(
        "- **Fixed few-shot examples**: 4 records from `sample_3000_seed42.json` (1 per bucket, "
        "midrange — Low=3.5, Medium=5.0, High=7.5, Critical=10.0). Same examples across all "
        "4 models × 2 versions. For the `anon` version, the examples go through HMAC pseudonymization with "
        "the same HMAC key.\n"
        "- **Greedy decoding** (do_sample=False, max_new_tokens=4). Does not capture variability.\n"
        "- **VaultGemma**: high invalid_rate (100%) — the model is not full instruction-tuned and "
        "does not follow the prompt template. Expected per the limitation documented in the plan (§13).\n"
        "- **Paired data**: held-out raw and anon contain the same records (only pseudonymized). "
        "record_id is preserved in both versions → exact pairing for Wilcoxon."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(rows)} models)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
