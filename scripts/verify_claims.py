#!/usr/bin/env python3
"""Assert the paper's numerical claims hold within tolerance; exit 0 iff all pass."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "experiment" / "reports"

passed, failed = 0, 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if ok: passed += 1
    else:  failed += 1


def parse_md_table(text: str, header_match: str) -> list[list[str]]:
    """Parse a Markdown pipe-table whose header line contains `header_match`."""
    lines = text.splitlines()
    rows: list[list[str]] = []
    in_tbl = False
    for line in lines:
        if not in_tbl:
            if header_match in line and "|" in line:
                in_tbl = True
            continue
        if not line.strip().startswith("|"):
            break
        cells = [c.strip() for c in line.strip("| \t").split("|")]
        if cells and not cells[0].startswith(("---", "===")):
            rows.append(cells)
    return rows


def parse_mean(s: str) -> float | None:
    """Parses a 'mean ± std' or 'mean' cell into the mean float."""
    m = re.search(r"([-+]?\d+\.\d+)", s)
    return float(m.group(1)) if m else None


def main() -> int:
    print("===== Verifying paper claims =====\n")

    print("Table 1 (Exp1 main exposure cross-seed) vs paper §4.3:")
    fr = (REPORTS / "FINAL_REPORT.md")
    if not fr.exists():
        check("FINAL_REPORT.md generated", False)
    else:
        rows = parse_md_table(fr.read_text(), "exp_all")
        expected = {
            ("gemma3-1b", "v0"): 3.49, ("gemma3-1b", "v1"): 1.38,
            ("qwen3-1.7b", "v0"): 4.48, ("qwen3-1.7b", "v1"): 2.83,
            ("llama32-3b", "v0"): 4.98, ("llama32-3b", "v1"): 3.20,
            ("vaultgemma-1b", "v0"): 3.60, ("vaultgemma-1b", "v1"): 1.24,
        }
        got = {}
        for r in rows:
            if len(r) >= 3:
                got[(r[0], r[1])] = parse_mean(r[2])
        for k, ev in expected.items():
            av = got.get(k)
            ok = av is not None and abs(av - ev) <= 0.05
            check(f"  {k[0]:14s} {k[1]}: paper {ev:.2f}, got {av:.2f}" if av else f"  {k[0]} {k[1]}", ok)

    print("\nTable 1 — V0→V1 reduction in [36%, 66%]:")
    if fr.exists():
        rows = parse_md_table(fr.read_text(), "exp_all")
        per_model = {}
        for r in rows:
            if len(r) >= 3 and r[1] in ("v0", "v1"):
                per_model.setdefault(r[0], {})[r[1]] = parse_mean(r[2])
        reductions = []
        for m, d in per_model.items():
            if "v0" in d and "v1" in d and d["v0"] and d["v1"]:
                reductions.append((d["v0"] - d["v1"]) / d["v0"] * 100)
        if reductions:
            rmin, rmax = min(reductions), max(reductions)
            check(f"  range = {rmin:.1f}%–{rmax:.1f}% (paper: 36%–66%)",
                  35 <= rmin <= 40 and 60 <= rmax <= 70)

    print("\nTable 2 (AnonShield V0): reduction in [40%, 61%]:")
    ar = (REPORTS / "ANONSHIELD_REPORT.md")
    if ar.exists():
        pcts = [float(p) for p in re.findall(r"\|\s*\+([\d.]+)%\s*\|", ar.read_text())]
        check(f"  reductions: {[f'{p:.1f}%' for p in pcts]} (paper: 40–61%)",
              len(pcts) >= 4 and all(40 <= p <= 62 for p in pcts))
    else:
        check("  ANONSHIELD_REPORT.md generated", False)

    print("\nTable 3 (Attack 2: slug exposure) — range claim [0.88, 1.63]:")
    sr = (REPORTS / "SLUG_ATTACK_REPORT.md")
    if sr.exists():
        rows = parse_md_table(sr.read_text(), "Anon-V0")
        vals = []
        for r in rows:
            for c in r[1:5]:
                v = parse_mean(c)
                if v is not None:
                    vals.append(v)
        if vals:
            smin, smax = min(vals), max(vals)
            check(f"  range = {smin:.2f}–{smax:.2f} (paper: 0.88–1.63)",
                  0.7 <= smin <= 1.0 and 1.4 <= smax <= 1.8)
        else:
            check("  Could not parse SLUG_ATTACK_REPORT.md", False)

    print("\nTable 5 (utility cross-seed F1-macro) — range [0.19, 0.28]:")
    ur = (REPORTS / "UTILITY_REPORT.md")
    if ur.exists():
        rows = parse_md_table(ur.read_text(), "F1_raw")
        f1s = []
        for r in rows:
            if len(r) >= 4:
                v = parse_mean(r[3])
                if v is not None:
                    f1s.append(v)
        if f1s:
            fmin, fmax = min(f1s), max(f1s)
            check(f"  range = {fmin:.3f}–{fmax:.3f} (paper: 0.19–0.28)",
                  0.17 <= fmin <= 0.22 and 0.25 <= fmax <= 0.30)

    print("\nTable 6 (prompt-eng utility) — |ΔF1| < 0.05 for instruction-tuned:")
    pr = (REPORTS / "UTILITY_PROMPT_REPORT.md")
    if pr.exists():
        rows = parse_md_table(pr.read_text(), "F1 raw")
        gemma_qwen_llama_ok = 0
        for r in rows:
            if len(r) >= 4 and r[0].startswith(("gemma", "qwen", "llama")):
                d = parse_mean(r[3])
                if d is not None and abs(d) < 0.05:
                    gemma_qwen_llama_ok += 1
        check(f"  instruction-tuned models with |ΔF1|<0.05: {gemma_qwen_llama_ok}/3 (paper: 3/3)",
              gemma_qwen_llama_ok == 3)

    print("\nFinding (i) / matched-update (recovered fraction in [66,132]%, mean ~100%, Delta ~ -1.98):")
    mu = (REPORTS / "MATCHED_UPDATE_REPORT.md")
    if mu.exists():
        text = mu.read_text()
        fracs = {m: int(p) for m, p in re.findall(
            r"\|\s*(gemma3-1b|qwen3-1\.7b|llama32-3b|vaultgemma-1b)\s*\|\s*(\d+)%", text)}
        paper_frac = {"gemma3-1b": 89, "qwen3-1.7b": 114, "llama32-3b": 132, "vaultgemma-1b": 66}
        check(f"  per-model fractions {list(fracs.values())} match paper {list(paper_frac.values())} (+/-2)",
              bool(fracs) and all(abs(fracs.get(m, -999) - p) <= 2 for m, p in paper_frac.items()))
        m_mean = re.search(r"\*\*mean\*\*\s*\|\s*\*\*(\d+)%", text)
        if m_mean:
            mv = int(m_mean.group(1))
            check(f"  mean recovered fraction = {mv}% (paper 100%)", abs(mv - 100) <= 5)
        m_delta = re.search(r"Mean Delta[^:]*:\s*([-+]?\d+\.\d+)", text)
        if m_delta:
            dv = float(m_delta.group(1))
            check(f"  mean Delta = {dv:.2f} bits (paper -1.98)", abs(dv + 1.98) <= 0.1)
    else:
        check("  MATCHED_UPDATE_REPORT.md generated", False)

    print("\nStack ablation (Delta_A/Delta_B vs paper, Delta_V1 CIs exclude 0):")
    ab = (REPORTS / "ABLATION_STACK_REPORT.md")
    if ab.exists():
        text = ab.read_text()
        paper = {"gemma3-1b": (-2.00, -0.67), "qwen3-1.7b": (-1.53, -0.63),
                 "llama32-3b": (-1.79, -0.19), "vaultgemma-1b": (-0.96, -1.41)}
        got = {}
        for r in parse_md_table(text, "Delta_A"):
            if r and r[0] in paper and len(r) >= 4:
                nums = [re.findall(r"([-+]\d+\.\d+)", cell) for cell in r[1:4]]
                got[r[0]] = nums  # [[A_point,A_lo,A_hi],[B...],[V1...]]
        check("  Delta_A/Delta_B per model within +/-0.05 of paper",
              all(m in got and abs(float(got[m][0][0]) - paper[m][0]) <= 0.05
                  and abs(float(got[m][1][0]) - paper[m][1]) <= 0.05 for m in paper))
        check("  Delta_V1 95% CI excludes 0 for all four models (robustly negative)",
              all(len(got.get(m, [[], [], []])[2]) >= 3 and float(got[m][2][2]) < 0 for m in paper))
        rng = re.search(r"range across models:\s*\[([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\]", text)
        if rng:
            lo, hi = float(rng.group(1)), float(rng.group(2))
            check(f"  Delta_V1 range [{lo:.2f}, {hi:.2f}] (paper [-2.37,-1.65])",
                  abs(lo + 2.37) <= 0.1 and abs(hi + 1.65) <= 0.1)
    else:
        check("  ABLATION_STACK_REPORT.md generated", False)

    print("\n===== Verification summary =====")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
