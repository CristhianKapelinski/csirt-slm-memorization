"""Utility eval via prompt engineering on base models (no fine-tune)."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
RESULTS_DIR = PILOT / "results" / "utility_prompt"
HELDOUT_RAW = PILOT / "data" / "utility_held_out.jsonl"
HELDOUT_ANON = PILOT / "data" / "utility_held_out_anon.jsonl"
SAMPLE_TRAIN = PILOT / "data" / "sample_3000_seed42.json"

ANON_CONFIG = PILOT / "configs" / "anonymization_config_tenable.json"

# Same default key as train_anon
ANON_KEY = "artifact-anon-key-v1"

MODELS = {
    "gemma3-1b":     "unsloth/gemma-3-1b-it",
    "qwen3-1.7b":    "Qwen/Qwen3-1.7B",
    "llama32-3b":    "unsloth/Llama-3.2-3B-Instruct",
    "vaultgemma-1b": "google/vaultgemma-1b",
}

# Selected deterministically from sample_3000 (one per bucket, midrange)
FEWSHOT_IDS = {
    "Low":      "b0c5cd5e-7322-41aa-5bfe-71e09c8c40de",  # cvss2=3.5
    "Medium":   "aab47bbe-f0ed-6c0e-8b87-af8d57fe6356",  # cvss2=5.0
    "High":     "077f59e0-1515-8a5e-c5e4-0bf4b9c5d271",  # cvss2=7.5
    "Critical": "9bc656c2-8b95-45a9-aad7-43584c2f1812",  # cvss2=10.0
}

BUCKETS = ["Low", "Medium", "High", "Critical"]
BUCKET_TO_IDX = {b: i for i, b in enumerate(BUCKETS)}


def cvss_bucket(score) -> str | None:
    if score is None: return None
    s = float(score)
    if 0.0 <= s < 4.0: return "Low"
    if 4.0 <= s < 7.0: return "Medium"
    if 7.0 <= s < 9.0: return "High"
    if 9.0 <= s <= 10.0: return "Critical"
    return None


def strip_cvss(record: dict) -> dict:
    rec = json.loads(json.dumps(record))
    rec.get("definition", {}).get("cvss2", {}).pop("base_score", None)
    return rec


def serialize(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


# HMAC pseudonymization (mirrors the data-side step)
import hashlib, hmac, re

TAG_SUFFIX_RE = re.compile(r"^(.+)\[([A-Z_]+)\]$")


def slug(value: str, etype: str, key: bytes, length: int = 8) -> str:
    digest = hmac.new(key, f"{etype}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"[{etype}_{digest[:length]}]"


def _apply_path(obj, path, etype, key, length, tag_filter):
    if not path: return 0
    head, rest = path[0], path[1:]
    if head == "*":
        if not isinstance(obj, list): return 0
        n = 0
        for item in obj:
            n += _apply_path(item, rest, etype, key, length, tag_filter)
        return n
    if not isinstance(obj, dict) or head not in obj: return 0
    if not rest:
        val = obj[head]
        if isinstance(val, str):
            obj[head] = slug(val, etype, key, length); return 1
        if isinstance(val, list):
            obj[head] = [slug(v, etype, key, length) if isinstance(v, str) else v for v in val]
            return sum(1 for v in val if isinstance(v, str))
        return 0
    return _apply_path(obj[head], rest, etype, key, length, tag_filter)


def apply_rule(record, rule, etype, key, length):
    m = TAG_SUFFIX_RE.match(rule)
    tag_filter = None
    if m: rule, tag_filter = m.group(1), m.group(2)
    return _apply_path(record, rule.split("."), etype, key, length, tag_filter)


def pseudonymize(record, config, key, length=8):
    n = 0
    for rule, etype in config["force_anonymize"].items():
        n += apply_rule(record, rule, etype, key, length)
    return n


def load_fewshot(version: str) -> list[dict]:
    """Returns 4 fewshot records (one per bucket) - matched to version (raw/anon)."""
    sample = json.loads(SAMPLE_TRAIN.read_text())
    by_id = {r["id"]: r for r in sample}
    examples = []
    for bucket, rid in FEWSHOT_IDS.items():
        rec = by_id.get(rid)
        if rec is None:
            raise RuntimeError(f"few-shot {bucket} id={rid} not in sample_3000")
        rec_copy = json.loads(json.dumps(rec))
        if version == "anon":
            cfg = json.loads(ANON_CONFIG.read_text())
            pseudonymize(rec_copy, cfg, ANON_KEY.encode())
        examples.append({"record": rec_copy, "bucket": bucket, "id": rid})
    return examples


PROMPT_HEADER = (
    "You are a cybersecurity analyst. Classify the severity of vulnerability records "
    "as Low, Medium, High, or Critical based on CVSS score range:\n"
    "- Low: 0.0 - 3.9\n"
    "- Medium: 4.0 - 6.9\n"
    "- High: 7.0 - 8.9\n"
    "- Critical: 9.0 - 10.0\n\n"
    "Examples:\n\n"
)


def build_prompt(target_record: dict, fewshot_records: list[dict]) -> str:
    parts = [PROMPT_HEADER]
    for fs in fewshot_records:
        rec_stripped = strip_cvss(fs["record"])
        parts.append(f"Record: {serialize(rec_stripped)}\nSeverity: {fs['bucket']}\n\n")
    parts.append("Now classify:\n\n")
    target_stripped = strip_cvss(target_record)
    parts.append(f"Record: {serialize(target_stripped)}\nSeverity:")
    return "".join(parts)


def parse_severity(raw: str) -> tuple[str | None, bool]:
    if not raw: return None, False
    low = raw.lower()
    for key, val in [("low", "Low"), ("medium", "Medium"), ("med", "Medium"),
                     ("moderate", "Medium"), ("high", "High"),
                     ("critical", "Critical"), ("crit", "Critical")]:
        if key in low:
            return val, True
    return None, False


def eval_one(model_key: str, version: str, batch_size: int = 8) -> int:
    model_name = MODELS[model_key]
    held = HELDOUT_ANON if version == "anon" else HELDOUT_RAW
    records = [json.loads(l) for l in held.read_text().splitlines() if l.strip()]
    print(f"[eval] {model_key} version={version}  n={len(records)}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_jsonl = RESULTS_DIR / f"{model_key}_{version}.jsonl"
    out_summary = RESULTS_DIR / f"{model_key}_{version}_summary.json"
    if out_summary.exists():
        print(f"[skip] {model_key}_{version} already done"); return 0

    fewshot = load_fewshot(version)
    fewshot_label = ", ".join(f"{e['bucket']}={e['id'][:8]}" for e in fewshot)
    print(f"  fewshot: {fewshot_label}", flush=True)

    print(f"  loading {model_name} (bf16, no adapter)", flush=True)
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"  # preserve "Severity:" at end
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.monotonic()-t0:.1f}s", flush=True)

    prompts, gts, rids = [], [], []
    for r in records:
        gt = cvss_bucket(r.get("definition", {}).get("cvss2", {}).get("base_score"))
        if gt is None: continue
        prompts.append(build_prompt(r, fewshot))
        gts.append(gt)
        rids.append(r.get("id", "?"))

    print(f"  built {len(prompts)} prompts (sample len: {len(prompts[0])} chars)", flush=True)

    preds, raws = [], []
    t0 = time.monotonic()
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=8192)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            gen = model.generate(
                **enc, max_new_tokens=4, do_sample=False, num_beams=1,
                pad_token_id=tok.pad_token_id,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for raw in decoded:
                pred, _ = parse_severity(raw)
                preds.append(pred)
                raws.append(raw)
            if (i // batch_size) % 5 == 0:
                print(f"  [{i+len(batch)}/{len(prompts)}]", flush=True)
    elapsed = time.monotonic() - t0

    with out_jsonl.open("w") as fout:
        for j, (rid, gt, pred, raw) in enumerate(zip(rids, gts, preds, raws)):
            fout.write(json.dumps({
                "model": model_key, "version": version,
                "record_id": rid, "ground_truth": gt, "prediction": pred,
                "raw_output": raw, "valid": pred is not None,
                "fewshot_used": [e["id"] for e in fewshot],
            }) + "\n")

    valid_mask = [p in BUCKET_TO_IDX for p in preds]
    n_valid = sum(valid_mask)
    n_invalid = len(prompts) - n_valid
    y_true = [BUCKET_TO_IDX[g] for g, m in zip(gts, valid_mask) if m]
    y_pred = [BUCKET_TO_IDX[p] for p, m in zip(preds, valid_mask) if m]
    if n_valid:
        f1m = f1_score(y_true, y_pred, average="macro", labels=list(range(4)), zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist()
        per_class = f1_score(y_true, y_pred, average=None, labels=list(range(4)), zero_division=0).tolist()
    else:
        f1m, acc, cm, per_class = 0.0, 0.0, [[0]*4]*4, [0.0]*4

    summary = {
        "model": model_key, "model_name": model_name, "version": version,
        "n_total": len(prompts), "n_valid": n_valid, "n_invalid": n_invalid,
        "invalid_rate": n_invalid / max(len(prompts), 1),
        "f1_macro": float(f1m), "accuracy": float(acc),
        "confusion_matrix": cm,
        "per_class_f1": dict(zip(BUCKETS, per_class)),
        "fewshot_ids": list(FEWSHOT_IDS.values()),
        "elapsed_s": elapsed,
        "raw_output_sample": raws[:5],
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {model_key}_{version}  F1={f1m:.3f}  acc={acc:.3f}  invalid={n_invalid}/{len(prompts)}  t={elapsed:.0f}s", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()))
    ap.add_argument("--version", choices=["raw", "anon"])
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--models", nargs="+", choices=list(MODELS.keys()), default=list(MODELS.keys()))
    ap.add_argument("--versions", nargs="+", choices=["raw", "anon"], default=["raw", "anon"])
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.scan:
        rcs = []
        for m in args.models:
            for v in args.versions:
                try:
                    rcs.append(eval_one(m, v, args.batch_size))
                except Exception as e:
                    print(f"[FAIL] {m}_{v}: {e}", file=sys.stderr)
                    rcs.append(2)
        return max(rcs) if rcs else 0
    elif args.model and args.version:
        return eval_one(args.model, args.version, args.batch_size)
    else:
        ap.error("--model+--version OR --scan required")


if __name__ == "__main__":
    sys.exit(main() or 0)
