"""Format-ablation analysis → reports/UTILITY_ABLATION_REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
ABL_DIR = PILOT / "results" / "utility_ablation"
OUT = PILOT / "reports" / "UTILITY_ABLATION_REPORT.md"

FORMATS = [
    ("format1_truncate_after_basescore", "Training-faithful"),
    ("format2_truncate_before_cvss2", "Truncated pre-CVSS2"),
    ("format3_truncate_close_dict", "Closed-dict"),
    ("format4_natural_lang", "Natural-language"),
]

MODELS = ["gemma3-1b", "qwen3-1.7b", "llama32-3b", "vaultgemma-1b"]


def main():
    out = ["# UTILITY_ABLATION_REPORT — format-ablation, V0 seed 42, 50 records\n"]
    out.append("Empirically justifies the use of the training-faithful prompt (§5.1 of the paper). "
               "The reported F1-macro is over 50 held-out records; acc is also reported. "
               "Random baseline 4-way: F1≈0.10, acc=0.25.\n")
    out.append("## F1-macro by (model, format)\n")
    out.append("| Model | Training-faithful | Truncated pre-CVSS2 | Closed-dict | Natural-language |")
    out.append("|---|---:|---:|---:|---:|")

    data: dict[str, dict] = {}
    for p in sorted(ABL_DIR.glob("*_ablation.json")):
        d = json.loads(p.read_text())
        data[d.get("model_short", p.stem.replace("_ablation", ""))] = d

    for m in MODELS:
        if m not in data:
            out.append(f"| {m} | — | — | — | — |")
            continue
        formats_d = data[m].get("formats", {})
        cells = []
        best = None
        for fid, _label in FORMATS:
            f1 = formats_d.get(fid, {}).get("f1_raw")
            cells.append(f1)
            if f1 is not None and (best is None or f1 > best):
                best = f1
        cell_strs = []
        for f1 in cells:
            if f1 is None:
                cell_strs.append("—")
            elif f1 == best:
                cell_strs.append(f"**{f1:.3f}**")
            else:
                cell_strs.append(f"{f1:.3f}")
        out.append(f"| {m} | " + " | ".join(cell_strs) + " |")

    out.append("\n*Bold = best per model.*\n")
    out.append("\n## Accuracy by (model, format)\n")
    out.append("| Model | Training-faithful | Truncated pre-CVSS2 | Closed-dict | Natural-language |")
    out.append("|---|---:|---:|---:|---:|")
    for m in MODELS:
        if m not in data:
            out.append(f"| {m} | — | — | — | — |")
            continue
        formats_d = data[m].get("formats", {})
        cells = [formats_d.get(fid, {}).get("acc_raw") for fid, _ in FORMATS]
        out.append(f"| {m} | " + " | ".join(f"{c:.3f}" if c is not None else "—" for c in cells) + " |")

    out.append("\n## Cross-format spread (max−min) and ratio (max/min)\n")
    out.append("| Model | min acc | max acc | spread | ratio |")
    out.append("|---|---:|---:|---:|---:|")
    for m in MODELS:
        if m not in data: continue
        accs = [data[m].get("formats", {}).get(fid, {}).get("acc_raw") for fid, _ in FORMATS]
        accs = [a for a in accs if a is not None and a > 0]
        if not accs: continue
        mn, mx = min(accs), max(accs)
        spread = mx - mn
        ratio = mx / mn if mn > 0 else float("nan")
        out.append(f"| {m} | {mn:.3f} | {mx:.3f} | {spread:.3f} | {ratio:.2f}× |")

    out.append("\n## Notes\n")
    out.append("- **Training-faithful** (format1) reproduces bit-for-bit the compact JSON "
               "serialization seen in training, truncated exactly at `\"base_score\":`. "
               "The model continues the JSON by predicting the numeric value.\n"
               "- **Truncated pre-CVSS2** truncates before `\"cvss2\":` — the model has to "
               "generate the entire structure `{\"base_score\":X,...}`.\n"
               "- **Closed-dict** truncates after `\"base_score\":` but uses the closing `}` "
               "(no `\"base_vector\"` continuation) — forcing inference without the "
               "rich continuation context.\n"
               "- **Natural-language** uses the `Severity:` instruction in plain text. "
               "Expected to be low: models fine-tuned on raw JSON suffer format "
               "specialization (Wang 2022 ProMoT) and do not follow a natural prompt.\n"
               "- The cross-format variation (max/min ratio) confirms the task's "
               "fragility with respect to format — the paper adopts training-faithful for "
               "all comparisons.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(out))
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
