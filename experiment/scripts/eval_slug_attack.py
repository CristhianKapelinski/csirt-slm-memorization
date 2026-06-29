"""Attack 2 - exposure of HMAC slug instead of original canary (Exp 2 dual attack)."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache" / "hf" / "hub"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as ev  # type: ignore

import numpy as np
import math

PILOT = ROOT / "experiment"
SLUG_RESULTS = PILOT / "results" / "eval_slug"
DEFAULT_KEY = "artifact-anon-key-v1"

# canary type -> HMAC etype
TYPE_TO_ETYPE = {
    "ipv4": "IP_ADDRESS",
    "ipv6": "IP_ADDRESS",
    "fqdn": "HOSTNAME",
    "mac": "MAC_ADDRESS",
    "asset_name": "HOSTNAME",
}


def slug_value(value: str, etype: str, key: bytes, length: int = 8) -> str:
    digest = hmac.new(key, f"{etype}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"[{etype}_{digest[:length]}]"


def attack_canaries_slug(model, tokenizer, canaries, variants_pool, base_eval, rng, key: bytes):
    """Same as eval.attack_canaries but slugs every value before serialization."""
    rows = []
    for c in canaries:
        etype = TYPE_TO_ETYPE.get(c["type"])
        if etype is None:
            print(f"[skip] no etype mapping for canary type {c['type']}", file=sys.stderr)
            continue
        slugged_value = slug_value(c["value"], etype, key)

        base = base_eval[int(rng.integers(0, len(base_eval)))]
        planted_rec = json.loads(json.dumps(base))
        planted_rec["id"] = f"canary-{c['id']}-eval"
        ev.set_path(planted_rec, c["field"], slugged_value)
        planted_text = ev.serialize(planted_rec)

        variant_vals_orig = variants_pool.get(c["id"], [])
        variant_slugs = [slug_value(v, etype, key) for v in variant_vals_orig]
        variant_texts = []
        for vs in variant_slugs:
            var_rec = json.loads(json.dumps(planted_rec))
            ev.set_path(var_rec, c["field"], vs)
            variant_texts.append(ev.serialize(var_rec))

        texts = [planted_text] + variant_texts
        nlls = ev.nll_per_token(model, tokenizer, texts)
        losses, minks = ev.loss_and_mink(nlls)
        zratios = [ev.zlib_ratio(t, l) for t, l in zip(texts, losses)]

        # canary-token-only loss on the slug
        planted_slice = ev._canary_token_slice(tokenizer, planted_text, slugged_value)
        canary_losses: list[float] = []
        if planted_slice is not None:
            s, e = planted_slice
            s_sh, e_sh = max(0, s - 1), max(0, e - 1)
            if e_sh > s_sh and nlls[0].numel() > s_sh:
                canary_losses.append(float(nlls[0][s_sh:e_sh].mean()))
            else:
                canary_losses.append(float("nan"))
            for var_slug, t, n in zip(variant_slugs, variant_texts, nlls[1:]):
                sl = ev._canary_token_slice(tokenizer, t, var_slug)
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

        exp_val, rank = ev.exposure(losses[0], losses[1:])
        loss_c_planted = canary_losses[0] if canary_losses else float("nan")
        loss_c_variants = [x for x in canary_losses[1:] if not math.isnan(x)]
        loss_c_variants_mean = float(np.mean(loss_c_variants)) if loss_c_variants else float("nan")
        rows.append({
            "canary_id": c["id"],
            "class": c["id"][0],
            "canary_original": c["value"],
            "canary_slug": slugged_value,
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


def write_slug_results(run_id: str, step: int, meta: dict, rows):
    SLUG_RESULTS.mkdir(parents=True, exist_ok=True)
    path = SLUG_RESULTS / f"{run_id}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"run_id": run_id, "step": step, **meta, **row}) + "\n")
    return path


def eval_one(run_id: str, base_model: str, variant: str, seed: int, ckpt: Path | None, step: int, key: bytes):
    out = SLUG_RESULTS / f"{run_id}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {run_id} already done")
        return
    start = time.monotonic()
    model, tokenizer = ev.load_model(base_model, ckpt)
    rng = np.random.default_rng(int(seed))

    class_a = ev.load_json(ev.CANARY_A)
    class_b = ev.load_json(ev.CANARY_B)
    variants_a = ev.load_json(ev.VARIANTS_A)
    variants_b = ev.load_json(ev.VARIANTS_B)
    held = ev.load_jsonl(ev.HELD_OUT)

    rows = []
    rows.extend(attack_canaries_slug(model, tokenizer, class_a, variants_a, held, rng, key))
    rows.extend(attack_canaries_slug(model, tokenizer, class_b, variants_b, held, rng, key))

    agg = []
    for r in rows:
        agg.append({"kind": "slug_canary", **r})

    for klass, filt in (("A", lambda r: r["class"] == "A"), ("B", lambda r: r["class"] == "B"), ("all", lambda r: True)):
        sub = [r for r in rows if filt(r)]
        if not sub: continue
        exps = [r["exposure_bits"] for r in sub]
        agg.append({"kind": "slug_exposure_agg", "class": klass,
                    "mean": float(np.mean(exps)), "median": float(np.median(exps)),
                    "max": float(np.max(exps)), "n": len(exps)})
        for metric in ("loss", "mink", "zlib", "loss_canary"):
            auc = ev.auc_ROC_planted_vs_variants(sub, metric)
            agg.append({"kind": "slug_auc", "class": klass, "metric": metric, "value": auc})

    agg.append({"kind": "slug_eval_wallclock", "seconds": time.monotonic() - start})

    write_slug_results(run_id, step, {"variant": variant, "model": base_model, "seed": seed,
                                      "attack": "slug"}, agg)
    print(f"[done] {run_id}  exposures: A={agg[20]['mean']:.3f}  B={agg[26]['mean']:.3f}  "
          f"all={agg[32]['mean']:.3f}  t={time.monotonic()-start:.0f}s",
          flush=True)
    del model
    import gc; gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_anon_adapters(adapters_root: Path) -> list[tuple[str, Path]]:
    out = []
    for d in sorted(adapters_root.iterdir()):
        if not d.is_dir(): continue
        if not d.name.startswith("anon_"): continue
        for cand in ["checkpoint-epoch3", "marker_final"]:
            p = d / cand
            if (p / "adapter_config.json").exists():
                out.append((d.name, p)); break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--config-dir", type=Path, default=PILOT / "configs")
    ap.add_argument("--adapters-dir", type=Path, default=PILOT / "adapters")
    ap.add_argument("--key", default=os.environ.get("ANON_SECRET_KEY") or DEFAULT_KEY)
    args = ap.parse_args()

    key = args.key.encode()
    fp = hashlib.sha256(key).hexdigest()[:12]
    print(f"[slug-attack] key fingerprint: {fp}", flush=True)

    import yaml
    if args.scan:
        targets = find_anon_adapters(args.adapters_dir)
        print(f"[scan] {len(targets)} Anon adapters", flush=True)
        for run_id, ckpt in targets:
            cfg_path = args.config_dir / f"{run_id}.yaml"
            if not cfg_path.exists():
                print(f"[skip] no config for {run_id}", file=sys.stderr); continue
            cfg = yaml.safe_load(cfg_path.read_text())
            try:
                eval_one(run_id, cfg["model_name"], cfg["variant"], int(cfg["seed"]), ckpt, 0, key)
            except Exception as e:
                print(f"[FAIL] {run_id}: {e}", file=sys.stderr)
        return 0
    elif args.run_id:
        cfg = yaml.safe_load((args.config_dir / f"{args.run_id}.yaml").read_text())
        ckpt = args.adapters_dir / args.run_id / "checkpoint-epoch3"
        if not (ckpt / "adapter_config.json").exists():
            ckpt = args.adapters_dir / args.run_id / "marker_final"
        eval_one(args.run_id, cfg["model_name"], cfg["variant"], int(cfg["seed"]), ckpt, 0, key)
        return 0
    else:
        ap.error("--scan or --run-id required")


if __name__ == "__main__":
    sys.exit(main() or 0)
