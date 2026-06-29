"""Utility eval - Plan A (logit-based, format-faithful)."""
from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
ADAPTERS_DIR = PILOT / "adapters"
CONFIGS_DIR = PILOT / "configs"
RESULTS_DIR = PILOT / "results" / "utility_eval_logits"
HELDOUT_RAW = PILOT / "data" / "utility_held_out.jsonl"
HELDOUT_ANON = PILOT / "data" / "utility_held_out_anon.jsonl"

BUCKETS = ["Low", "Medium", "High", "Critical"]
BUCKET_TO_IDX = {b: i for i, b in enumerate(BUCKETS)}
# mid-range numeric prototype per bucket
BUCKET_PROTO = {"Low": "2.0", "Medium": "5.5", "High": "8.0", "Critical": "9.5"}

# JSON closure after the score (next field is base_vector)
DEFAULT_SUFFIX = ',"base_vector"'


def serialize(record: dict) -> str:
    """Match train.py exactly."""
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def cvss_bucket(score) -> str | None:
    if score is None: return None
    s = float(score)
    if 0.0 <= s < 4.0: return "Low"
    if 4.0 <= s < 7.0: return "Medium"
    if 7.0 <= s < 9.0: return "High"
    if 9.0 <= s <= 10.0: return "Critical"
    return None


def build_prompt_format1(record: dict) -> tuple[str, str] | None:
    """Format 1 (canonical, training-faithful): truncate right after `"base_score":`."""
    raw = serialize(record)
    m = re.search(r'"cvss2":\{[^}]*?"base_score":', raw)
    if not m:
        return None
    cut = m.end()
    return raw[:cut], DEFAULT_SUFFIX


def build_prompt_format2(record: dict) -> tuple[str, str] | None:
    """Format 2: truncate BEFORE `"cvss2":`. Continuation includes the full cvss2 dict opening."""
    raw = serialize(record)
    m = re.search(r'"cvss2":', raw)
    if not m:
        return None
    cut = m.end()
    return raw[:cut], '{"base_score":VALUE,"base_vector"'


def build_prompt_format3(record: dict) -> tuple[str, str] | None:
    """Format 3: same prefix as format 1, but suffix is just `}` (close cvss2 dict)."""
    raw = serialize(record)
    m = re.search(r'"cvss2":\{[^}]*?"base_score":', raw)
    if not m:
        return None
    cut = m.end()
    return raw[:cut], "}"


def build_prompt_format4(record: dict) -> tuple[str, str] | None:
    """Format 4 (natural language, the failed v1) - for ablation comparison."""
    rec_clean = json.loads(json.dumps(record))
    rec_clean.get("definition", {}).get("cvss2", {}).pop("base_score", None)
    rec_json = json.dumps(rec_clean, ensure_ascii=False, separators=(",", ":"))
    prompt = (
        "Given the following vulnerability record, classify the severity as one of: "
        "Low, Medium, High, Critical based on CVSS score range.\n\n"
        f"Record:\n{rec_json}\n\nSeverity:"
    )
    return prompt, " VALUE"


def strip_cvss(record: dict) -> dict:
    rec = json.loads(json.dumps(record))
    rec.get("definition", {}).get("cvss2", {}).pop("base_score", None)
    return rec


def score_continuation(model, tok, prefix_ids: torch.Tensor,
                       continuation_ids: torch.Tensor, device) -> tuple[float, int]:
    """Teacher-forced sum log P(continuation | prefix), with token count."""
    full = torch.cat([prefix_ids, continuation_ids], dim=-1).to(device)
    with torch.no_grad():
        out = model(full.unsqueeze(0))
    logits = out.logits[0]
    pref_len = prefix_ids.shape[-1]
    cont_len = continuation_ids.shape[-1]
    log_probs = torch.log_softmax(logits[pref_len - 1: pref_len + cont_len - 1], dim=-1)
    target = full[pref_len: pref_len + cont_len]
    sum_lp = float(log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item())
    return sum_lp, cont_len


def evaluate_one(model, tok, records: list[dict], format_fn, device,
                 bucket_proto=None, suffix_template_replace_VALUE=None) -> dict:
    """Returns dict with predictions, gts, F1, acc, acc_norm, time, raw scores."""
    if bucket_proto is None:
        bucket_proto = BUCKET_PROTO

    t0 = time.monotonic()
    preds_raw, preds_norm, gts = [], [], []
    sample_scores = []
    n_eval = 0

    for rec in records:
        gt = cvss_bucket(rec.get("definition", {}).get("cvss2", {}).get("base_score"))
        if gt is None:
            continue
        pp = format_fn(rec)
        if pp is None:
            continue
        prefix_str, suffix_pattern = pp

        prefix_ids = tok(prefix_str, return_tensors="pt", add_special_tokens=True,
                         truncation=True, max_length=900).input_ids[0]

        scores_lp = []
        scores_norm = []
        bytes_per = []
        for b in BUCKETS:
            value = bucket_proto[b]
            if "VALUE" in suffix_pattern:
                cont_str = suffix_pattern.replace("VALUE", value)
            else:
                cont_str = value + suffix_pattern
            cont_ids = tok(cont_str, return_tensors="pt", add_special_tokens=False).input_ids[0]
            sum_lp, n_tok = score_continuation(model, tok, prefix_ids, cont_ids, device)
            scores_lp.append(sum_lp)
            scores_norm.append(sum_lp / max(n_tok, 1))
            bytes_per.append(len(cont_str.encode("utf-8")))

        # acc_norm convention: divide by char-bytes (lm-eval-harness)
        scores_byte_norm = [scores_lp[i] / max(bytes_per[i], 1) for i in range(4)]

        pred_raw = max(range(4), key=lambda i: scores_lp[i])
        pred_norm = max(range(4), key=lambda i: scores_byte_norm[i])

        preds_raw.append(pred_raw)
        preds_norm.append(pred_norm)
        gts.append(BUCKET_TO_IDX[gt])
        if len(sample_scores) < 5:
            sample_scores.append({
                "gt": gt, "pred_raw": BUCKETS[pred_raw], "pred_norm": BUCKETS[pred_norm],
                "scores_lp": [round(s, 2) for s in scores_lp],
                "scores_byte_norm": [round(s, 4) for s in scores_byte_norm],
            })
        n_eval += 1

    elapsed = time.monotonic() - t0
    if not gts:
        return {"n_eval": 0, "elapsed": elapsed}

    f1m_raw = f1_score(gts, preds_raw, average="macro", labels=list(range(4)), zero_division=0)
    acc_raw = accuracy_score(gts, preds_raw)
    f1m_norm = f1_score(gts, preds_norm, average="macro", labels=list(range(4)), zero_division=0)
    acc_norm = accuracy_score(gts, preds_norm)
    cm_raw = confusion_matrix(gts, preds_raw, labels=list(range(4))).tolist()

    return {
        "n_eval": n_eval, "elapsed": elapsed,
        "f1_raw": float(f1m_raw), "acc_raw": float(acc_raw),
        "f1_norm": float(f1m_norm), "acc_norm": float(acc_norm),
        "confusion_matrix_raw": cm_raw,
        "preds_raw": preds_raw, "preds_norm": preds_norm, "gts": gts,
        "samples": sample_scores,
    }


def find_adapter(run_id):
    base = ADAPTERS_DIR / run_id
    for cand in ["checkpoint-epoch3", "marker_final"]:
        p = base / cand
        if (p / "adapter_config.json").exists():
            return p
    ckpts = sorted(base.glob("checkpoint-*"))
    for c in reversed(ckpts):
        if (c / "adapter_config.json").exists():
            return c
    return None


def build_model(cfg, adapter_path):
    variant = cfg["variant"]
    model_name = cfg["model_name"]
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if variant == "v0":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        )
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tok


def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_label_tokenization(tok, model_key: str):
    print(f"\n[tokenization] base = {model_key}")
    for b in BUCKETS:
        proto = BUCKET_PROTO[b]
        cont_canonical = proto + DEFAULT_SUFFIX
        ids = tok(cont_canonical, add_special_tokens=False).input_ids
        print(f"  {b:<10} proto={proto!r}  ids={ids}  decoded={tok.decode(ids)!r}")


def evaluate_run(run_id: str, format_name: str = "format1") -> int:
    cfg_path = CONFIGS_DIR / f"{run_id}.yaml"
    if not cfg_path.exists():
        print(f"[skip] no config for {run_id}", file=sys.stderr); return 1
    cfg = yaml.safe_load(cfg_path.read_text())
    adapter = find_adapter(run_id)
    if adapter is None:
        print(f"[skip] no adapter for {run_id}", file=sys.stderr); return 1

    use_anon = run_id.startswith("anon_") or run_id.endswith("_w0") and run_id.startswith("anon_")
    use_anon = run_id.startswith("anon_")
    held = HELDOUT_ANON if use_anon else HELDOUT_RAW
    records = [json.loads(l) for l in held.read_text().splitlines() if l.strip()]
    print(f"[eval] {run_id}  use_anon={use_anon}  n={len(records)}  format={format_name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_summary = RESULTS_DIR / f"{run_id}_summary.json"
    if out_summary.exists():
        print(f"[skip] {run_id} already evaluated"); return 0

    model, tok = build_model(cfg, adapter)
    device = next(model.parameters()).device

    fmt_map = {"format1": build_prompt_format1, "format2": build_prompt_format2,
               "format3": build_prompt_format3, "format4": build_prompt_format4}
    fmt_fn = fmt_map[format_name]
    proto = BUCKET_PROTO if format_name != "format4" else {b: b for b in BUCKETS}

    res = evaluate_one(model, tok, records, fmt_fn, device, bucket_proto=proto)
    res["run_id"] = run_id
    res["model"] = cfg.get("model_name")
    res["variant"] = cfg["variant"]
    res["seed"] = cfg["seed"]
    res["use_anon"] = use_anon
    res["format"] = format_name
    out_summary.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[done] {run_id}  F1_raw={res['f1_raw']:.3f}  acc_raw={res['acc_raw']:.3f}  "
          f"F1_norm={res['f1_norm']:.3f}  acc_norm={res['acc_norm']:.3f}  "
          f"n={res['n_eval']}  t={res['elapsed']:.0f}s")
    free_model(model)
    return 0


def ablation(run_id: str, n_records: int = 50) -> int:
    """Compare 4 prompt formats on a single adapter (50 records)."""
    cfg_path = CONFIGS_DIR / f"{run_id}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    adapter = find_adapter(run_id)
    if adapter is None:
        print(f"[ABLATION] no adapter for {run_id}"); return 1

    use_anon = run_id.startswith("anon_")
    held = HELDOUT_ANON if use_anon else HELDOUT_RAW
    records = [json.loads(l) for l in held.read_text().splitlines() if l.strip()][:n_records]

    print(f"[ABLATION] {run_id}  n={len(records)}  4 prompt formats")
    model, tok = build_model(cfg, adapter)
    device = next(model.parameters()).device

    log_label_tokenization(tok, cfg["model_name"])

    formats = [
        ("format1_truncate_after_basescore", build_prompt_format1, BUCKET_PROTO),
        ("format2_truncate_before_cvss2", build_prompt_format2, BUCKET_PROTO),
        ("format3_truncate_close_dict", build_prompt_format3, BUCKET_PROTO),
        ("format4_natural_lang", build_prompt_format4, {b: b for b in BUCKETS}),
    ]
    results = []
    for name, fn, proto in formats:
        print(f"\n--- {name} ---")
        r = evaluate_one(model, tok, records, fn, device, bucket_proto=proto)
        r["format"] = name
        results.append(r)
        if r.get("n_eval"):
            print(f"  F1_raw={r['f1_raw']:.3f}  acc_raw={r['acc_raw']:.3f}  "
                  f"F1_norm={r['f1_norm']:.3f}  acc_norm={r['acc_norm']:.3f}  "
                  f"t={r['elapsed']:.0f}s")
            print(f"  sample (first 3):")
            for s in r["samples"][:3]:
                print(f"    GT={s['gt']:<10} pred_raw={s['pred_raw']:<10} pred_norm={s['pred_norm']:<10} "
                      f"lp={s['scores_lp']}  byte_norm={s['scores_byte_norm']}")
        else:
            print(f"  n_eval=0 (format failed to apply)")

    print("\n" + "=" * 60)
    print(f"ABLATION SUMMARY — {run_id}")
    print("=" * 60)
    print(f"{'format':<40} {'F1_raw':>8} {'acc_raw':>8} {'F1_norm':>9} {'acc_norm':>9}")
    for r in results:
        if r.get("n_eval"):
            print(f"{r['format']:<40} {r['f1_raw']:>8.3f} {r['acc_raw']:>8.3f} "
                  f"{r['f1_norm']:>9.3f} {r['acc_norm']:>9.3f}")
        else:
            print(f"{r['format']:<40}    failed")

    out_dir = PILOT / "results" / "utility_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_short = run_id.split("_v0_")[0] if "_v0_" in run_id else run_id
    persisted = {
        "run_id": run_id,
        "model": cfg.get("model_name"),
        "model_short": model_short,
        "n": n_records,
        "seed": cfg.get("seed"),
        "variant": cfg.get("variant"),
        "formats": {
            r["format"]: {
                "f1_raw": r.get("f1_raw"),
                "f1_norm": r.get("f1_norm"),
                "acc_raw": r.get("acc_raw"),
                "acc_norm": r.get("acc_norm"),
                "n_eval": r.get("n_eval"),
                "elapsed": r.get("elapsed"),
            } for r in results
        },
    }
    out_path = out_dir / f"{model_short}_ablation.json"
    out_path.write_text(json.dumps(persisted, indent=2, ensure_ascii=False))
    print(f"\n[ABLATION] persisted → {out_path.relative_to(ROOT)}")

    free_model(model)
    return 0


def list_targets() -> list[str]:
    if not ADAPTERS_DIR.exists(): return []
    targets = []
    for d in sorted(ADAPTERS_DIR.iterdir()):
        if not d.is_dir(): continue
        rid = d.name
        if rid.endswith("_w0"): continue
        if not (CONFIGS_DIR / f"{rid}.yaml").exists(): continue
        if find_adapter(rid) is None: continue
        if (RESULTS_DIR / f"{rid}_summary.json").exists(): continue
        targets.append(rid)
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--ablation", help="run_id to do 4-format ablation on")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--format", default="format1")
    args = ap.parse_args()

    if args.ablation:
        return ablation(args.ablation)
    if args.scan:
        targets = list_targets()
        print(f"[scan] {len(targets)} targets")
        rcs = []
        for rid in targets:
            try:
                rcs.append(evaluate_run(rid, args.format))
            except Exception as e:
                print(f"[FAIL] {rid}: {e}", file=sys.stderr)
                rcs.append(2)
        return max(rcs) if rcs else 0
    elif args.run_id:
        return evaluate_run(args.run_id, args.format)
    else:
        ap.error("either --run-id, --scan, or --ablation required")


if __name__ == "__main__":
    sys.exit(main() or 0)
