"""Eval driver: memorization attacks + utility (severity F1)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache" / "hf" / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

PILOT = ROOT / "experiment"
RESULTS_DIR = PILOT / "results" / "eval_checkpoints"
CANARY_A = PILOT / "canaries" / "class_a.json"
CANARY_B = PILOT / "canaries" / "class_b.json"
VARIANTS_A = PILOT / "canaries" / "variants_pool_a.json"
VARIANTS_B = PILOT / "canaries" / "variants_pool_b.json"
HELD_OUT = PILOT / "data" / "eval_held_out.jsonl"

CVSS_BUCKETS = (("Low", 0.0, 4.0), ("Medium", 4.0, 7.0), ("High", 7.0, 9.0), ("Critical", 9.0, 10.01))


def serialize(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def set_path(obj: dict, path: str, value) -> dict:
    """Apply canary injection path-style; supports tags list insertion (mirror de 00_prepare_data)."""
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        if p == "tags" and isinstance(cur.get(p), list):
            last = parts[-1]
            new_tag = {"id": f"canary-tag-{last}", "category": last.upper(), "value": value, "type": "static"}
            cur[p] = [t for t in cur[p] if t.get("id") != new_tag["id"]] + [new_tag]
            return obj
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return obj


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def nll_per_token(model, tokenizer, texts: list[str], batch_size: int = 1) -> list[torch.Tensor]:
    """Per-token negative log-likelihoods (batch_size=1 keeps Gemma-4 logits under 16GB)."""
    out: list[torch.Tensor] = []
    device = next(model.parameters()).device
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = enc.input_ids[:, 1:]
            mask = enc.attention_mask[:, 1:].bool()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_nll = -log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1).float().cpu()
            mask_cpu = mask.cpu()
            del logits, shift_logits, log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for j in range(token_nll.size(0)):
            out.append(token_nll[j][mask_cpu[j]].clone())
    return out


def loss_and_mink(nlls: list[torch.Tensor], k: float = 0.2) -> tuple[list[float], list[float]]:
    losses, minks = [], []
    for x in nlls:
        if x.numel() == 0:
            losses.append(float("nan"))
            minks.append(float("nan"))
            continue
        losses.append(float(x.mean()))
        n = max(1, int(x.numel() * k))
        mu, sigma = float(x.mean()), float(x.std() + 1e-8)
        z = ((x - mu) / sigma).numpy()
        minks.append(float(np.sort(z)[-n:].mean()))
    return losses, minks


def zlib_ratio(text: str, loss: float) -> float:
    z = len(zlib.compress(text.encode("utf-8")))
    return loss / max(z, 1)


def exposure(loss_planted: float, variant_losses: list[float]) -> tuple[float, int]:
    """Carlini exposure: lower perplexity than more variants -> higher exposure."""
    all_losses = [loss_planted] + variant_losses
    rank = 1 + sum(1 for l in variant_losses if l < loss_planted)
    return math.log2(len(all_losses)) - math.log2(rank), rank


def canary_record_for(canary: dict, base_eval: list[dict], rng: np.random.Generator) -> dict:
    """Build a record that structurally mirrors a training canary, using a held-out base."""
    base = base_eval[int(rng.integers(0, len(base_eval)))]
    rec = json.loads(json.dumps(base))
    rec["id"] = f"canary-{canary['id']}-eval"
    set_path(rec, canary["field"], canary["value"])
    return rec


def _canary_token_slice(tokenizer, text: str, canary_value: str, max_length: int = 1024) -> tuple[int, int] | None:
    """Token (start, end) of canary_value in text, clipped to max_length; None if truncated out."""
    char_pos = text.find(canary_value)
    if char_pos < 0:
        return None
    pre_ids = tokenizer(text[:char_pos], add_special_tokens=False).input_ids
    start = len(pre_ids)
    cand_ids = tokenizer(text[:char_pos + len(canary_value)], add_special_tokens=False).input_ids
    end = len(cand_ids)
    if start >= max_length:
        return None
    return (start, min(end, max_length))


def attack_canaries(model, tokenizer, canaries: list[dict], variants_pool: dict[str, list[str]], base_eval: list[dict], rng: np.random.Generator) -> list[dict]:
    rows = []
    for c in canaries:
        planted_rec = canary_record_for(c, base_eval, rng)
        planted_text = serialize(planted_rec)

        variant_texts = []
        variant_vals = variants_pool.get(c["id"], [])
        for v in variant_vals:
            var_rec = json.loads(json.dumps(planted_rec))
            set_path(var_rec, c["field"], v)
            variant_texts.append(serialize(var_rec))

        texts = [planted_text] + variant_texts
        nlls = nll_per_token(model, tokenizer, texts)
        losses, minks = loss_and_mink(nlls)
        zratios = [zlib_ratio(t, l) for t, l in zip(texts, losses)]

        # canary-only loss: mean NLL over the canary token slice (avoids AUC dilution)
        planted_slice = _canary_token_slice(tokenizer, planted_text, c["value"])
        canary_losses: list[float] = []
        if planted_slice is not None:
            s, e = planted_slice
            # shifted NLL spans [1, T-1] in input positions; slice needs shift of 1
            s_sh, e_sh = max(0, s - 1), max(0, e - 1)
            if e_sh > s_sh and nlls[0].numel() > s_sh:
                canary_losses.append(float(nlls[0][s_sh:e_sh].mean()))
            else:
                canary_losses.append(float("nan"))
            for var_val, t, n in zip(variant_vals, variant_texts[:len(variant_vals)], nlls[1:]):
                sl = _canary_token_slice(tokenizer, t, var_val)
                if sl is None:
                    canary_losses.append(float("nan"))
                    continue
                vs, ve = max(0, sl[0] - 1), max(0, sl[1] - 1)
                if ve > vs and n.numel() > vs:
                    canary_losses.append(float(n[vs:ve].mean()))
                else:
                    canary_losses.append(float("nan"))
        else:
            canary_losses = [float("nan")] * (1 + len(variant_texts))

        exp_val, rank = exposure(losses[0], losses[1:])
        loss_c_planted = canary_losses[0] if canary_losses else float("nan")
        loss_c_variants = [x for x in canary_losses[1:] if not math.isnan(x)]
        loss_c_variants_mean = float(np.mean(loss_c_variants)) if loss_c_variants else float("nan")
        rows.append({
            "canary_id": c["id"],
            "class": c["id"][0],
            "loss_planted": losses[0],
            "loss_variants_mean": float(np.mean(losses[1:])),
            "loss_canary_planted": loss_c_planted,
            "loss_canary_variants_mean": loss_c_variants_mean,
            "mink_planted": minks[0],
            "mink_variants_mean": float(np.mean(minks[1:])),
            "zlib_planted": zratios[0],
            "zlib_variants_mean": float(np.mean(zratios[1:])),
            "exposure_bits": exp_val,
            "rank": rank,
        })
    return rows


def auc_ROC_planted_vs_variants(attack_rows: list[dict], metric: str) -> float:
    """Binary classif: planted (positive, label=1) vs variants via *_variants_mean."""
    planted_key = f"{metric}_planted" if metric != "loss_canary" else "loss_canary_planted"
    variants_key = f"{metric}_variants_mean" if metric != "loss_canary" else "loss_canary_variants_mean"
    y, s = [], []
    for row in attack_rows:
        p = row.get(planted_key)
        v = row.get(variants_key)
        if isinstance(p, float) and not math.isnan(p):
            y.append(1); s.append(-p)
        if isinstance(v, float) and not math.isnan(v):
            y.append(0); s.append(-v)
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def cvss_bucket(rec: dict) -> str | None:
    v = (rec.get("definition") or {}).get("cvss2") or {}
    try:
        s = float(v.get("base_score"))
    except (TypeError, ValueError):
        return None
    for name, lo, hi in CVSS_BUCKETS:
        if lo <= s < hi:
            return name
    return None


def severity_eval(model, tokenizer, held: list[dict]) -> dict:
    """Perplexity-based zero-shot severity: argmin NLL over the 4 labels as continuation."""
    labels = ["Low", "Medium", "High", "Critical"]
    prompt_tpl = "{body}\n\nThe severity level is"
    y_true, y_pred = [], []
    for rec in held:
        gold = cvss_bucket(rec)
        if not gold:
            continue
        stripped = json.loads(json.dumps(rec))
        stripped.pop("severity", None)
        (stripped.get("definition") or {}).pop("severity", None)
        prompt = prompt_tpl.format(body=serialize(stripped)[:1400])
        candidates = [prompt + " " + lab for lab in labels]
        nlls = nll_per_token(model, tokenizer, candidates)
        # label is a single trailing token, so its NLL is the last shifted element
        label_nlls = [float(n[-1]) if n.numel() >= 1 else float("nan") for n in nlls]
        if any(math.isnan(x) for x in label_nlls):
            continue
        y_pred.append(labels[int(np.argmin(label_nlls))])
        y_true.append(gold)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else float("nan")
    f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if y_true else float("nan")
    return {"n": len(y_true), "accuracy": acc, "f1_macro": f1}


def _unwrap_custom_linears(model) -> None:
    """Mirror of train._unwrap_custom_linears; eval must unwrap before PeftModel loads."""
    for _, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            inner = getattr(child, "linear", None)
            if isinstance(inner, torch.nn.Linear) and not isinstance(child, torch.nn.Linear):
                setattr(parent, child_name, inner)


def load_model(base_name: str, adapter_path: Path | None):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        base_name, dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    _unwrap_custom_linears(model)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def write_results(run_id: str, step: int, meta: dict, rows: Iterable[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{run_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"run_id": run_id, "step": step, **meta, **row}) + "\n")
    return path


def eval_checkpoint(run_id: str, base_model: str, variant: str, seed: int, ckpt: Path | None, step: int) -> None:
    start = time.monotonic()
    model, tokenizer = load_model(base_model, ckpt)
    rng = np.random.default_rng(int(seed))

    class_a = load_json(CANARY_A)
    class_b = load_json(CANARY_B)
    variants_a = load_json(VARIANTS_A)
    variants_b = load_json(VARIANTS_B)
    held = load_jsonl(HELD_OUT)

    attack_rows = []
    attack_rows.extend(attack_canaries(model, tokenizer, class_a, variants_a, held, rng))
    attack_rows.extend(attack_canaries(model, tokenizer, class_b, variants_b, held, rng))

    agg_rows = []
    for row in attack_rows:
        agg_rows.append({"kind": "canary", **row})

    for klass, subset_filter in (("A", lambda r: r["class"] == "A"), ("B", lambda r: r["class"] == "B"), ("all", lambda r: True)):
        subset = [r for r in attack_rows if subset_filter(r)]
        if not subset:
            continue
        exps = [r["exposure_bits"] for r in subset]
        agg_rows.append({
            "kind": "exposure_agg", "class": klass,
            "mean": float(np.mean(exps)), "median": float(np.median(exps)),
            "max": float(np.max(exps)), "n": len(exps),
        })
        for metric in ("loss", "mink", "zlib", "loss_canary"):
            auc = auc_ROC_planted_vs_variants(subset, metric)
            agg_rows.append({"kind": "auc", "class": klass, "metric": metric, "value": auc})

    util = severity_eval(model, tokenizer, held)
    agg_rows.append({"kind": "utility", "task": "severity", **util})

    agg_rows.append({"kind": "eval_wallclock", "seconds": time.monotonic() - start})

    write_results(run_id, step, {"variant": variant, "model": base_model, "seed": seed}, agg_rows)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pick_checkpoints(run_dir: Path) -> list[tuple[Path, int]]:
    """Collect checkpoints; epoch<N> dirs map to step=N*10000 to order without colliding."""
    items = []
    for ckpt in run_dir.glob("checkpoint-*"):
        name = ckpt.name
        tail = name.rsplit("-", 1)[-1]
        if tail.isdigit():
            items.append((ckpt, int(tail)))
            continue
        if tail.startswith("epoch"):
            try:
                epoch_num = int(tail[len("epoch"):])
            except ValueError:
                continue
            items.append((ckpt, epoch_num * 10000))
    for marker in run_dir.glob("marker_*"):
        items.append((marker, -1))
    return sorted(items, key=lambda p: p[1])


def already_evaluated(run_id: str, step: int) -> bool:
    path = RESULTS_DIR / f"{run_id}.jsonl"
    if not path.exists():
        return False
    needle = f'"step": {step},'
    for line in path.open(encoding="utf-8"):
        if needle in line:
            return True
    return False


def scan(adapters_root: Path, config_dir: Path, force: bool = False) -> int:
    if not adapters_root.exists():
        print(f"no adapters at {adapters_root}", file=sys.stderr)
        return 0
    n = 0
    for run_dir in sorted(adapters_root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = config_dir / f"{run_dir.name}.yaml"
        if not cfg_path.exists():
            continue
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
        for ckpt, step in pick_checkpoints(run_dir):
            if not force and already_evaluated(run_dir.name, step):
                continue
            eval_checkpoint(run_dir.name, cfg["model_name"], cfg["variant"], int(cfg["seed"]), ckpt, step)
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--config-dir", type=Path, default=PILOT / "configs")
    ap.add_argument("--adapters-dir", type=Path, default=PILOT / "adapters")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if args.scan:
        n = scan(args.adapters_dir, args.config_dir, force=args.force)
        print(f"evaluated {n} new checkpoint(s)", file=sys.stderr)
        return 0

    if args.run_id and args.checkpoint:
        import yaml
        cfg = yaml.safe_load((args.config_dir / f"{args.run_id}.yaml").read_text())
        eval_checkpoint(args.run_id, cfg["model_name"], cfg["variant"], int(cfg["seed"]), args.checkpoint, args.step)
        return 0

    ap.error("either --scan or --run-id + --checkpoint required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
