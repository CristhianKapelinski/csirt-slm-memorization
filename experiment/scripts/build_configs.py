"""Generate YAMLs per (model x variant x seed x wave) per the V3 plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
CONFIG_DIR = PILOT / "configs"
DATA_DIR = PILOT / "data"

MODELS = {
    "gemma3-1b":     "unsloth/gemma-3-1b-it",
    "qwen3-1.7b":    "Qwen/Qwen3-1.7B",
    "llama32-3b":    "unsloth/Llama-3.2-3B-Instruct",
    "vaultgemma-1b": "google/vaultgemma-1b",
}

# Order matters for round-robin (variant first, then model).
VARIANTS = ("v0", "v1", "v2", "v3")

BASE = {
    "num_epochs": 3,
    "batch_size": 1,
    "grad_accum": 4,
    "learning_rate": 1.0e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "max_seq_length": 1024,
    "save_steps": 500,
    "save_total_limit": 3,
    "logging_steps": 10,
    "lossguard_min": 0.3,
    "bf16": True,  # V0; V1/V2/V3 use 4-bit NF4 internally
    # DP defaults (ignored if variant in {v0, v1})
    "dp_target_epsilon": 8.0,
    "dp_target_delta": 1.0e-5,
    "dp_max_grad_norm": 1.0,
    "dp_lot_size": 32,
    "dp_grad_sample_mode": "ghost",
}

VARIANT_OVERRIDES = {
    "v0": {},  # bf16 + SFTTrainer + adamw_8bit
    "v1": {},  # same stack as V2 but without PrivacyEngine - train.py honors variant
    "v2": {"dp_target_epsilon": 8.0},
    "v3": {"dp_target_epsilon": 2.0},
}

# Waves: smoke (1 ep, 90 records) + 3 seeds
WAVES = {
    0: {"seeds": [42],   "num_epochs": 1, "train_file_suffix": "_smoke",
        "dp_lot_size": 16, "save_steps": 50, "logging_steps": 5},
    1: {"seeds": [42],   "num_epochs": 3, "train_file_suffix": ""},
    2: {"seeds": [1337], "num_epochs": 3, "train_file_suffix": ""},
    3: {"seeds": [2024], "num_epochs": 3, "train_file_suffix": ""},
}


def run_id_for(model_key: str, variant: str, seed: int, wave: int) -> str:
    return f"{model_key}_{variant}_seed{seed}" + ("_w0" if wave == 0 else "")


def train_file_for(suffix: str) -> str:
    return str(DATA_DIR / f"train_v0{suffix}.jsonl")


def build_config(model_key: str, variant: str, seed: int, wave: int) -> dict:
    spec = WAVES[wave]
    cfg = {
        "run_id": run_id_for(model_key, variant, seed, wave),
        "model_name": MODELS[model_key],
        "variant": variant,
        "seed": seed,
        "train_file": train_file_for(spec["train_file_suffix"]),
        **BASE,
        **VARIANT_OVERRIDES[variant],
    }
    for k, v in spec.items():
        if k in ("seeds", "train_file_suffix"):
            continue
        cfg[k] = v
    # Llama 3.2 3B in V2/V3 needs seq=768 to fit 16GB (covers all canaries)
    if model_key == "llama32-3b" and variant in ("v2", "v3"):
        cfg["max_seq_length"] = 768
    return cfg


def write_config(cfg: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{cfg['run_id']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def build_queue() -> list[dict]:
    """Round-robin: variant -> model -> wave seed."""
    queue: list[dict] = []
    slot = 0
    for wave in sorted(WAVES):
        spec = WAVES[wave]
        for variant in VARIANTS:
            for model_key in MODELS:
                for seed in spec["seeds"]:
                    cfg = build_config(model_key, variant, seed, wave)
                    write_config(cfg)
                    slot += 1
                    queue.append({
                        "run_id": cfg["run_id"],
                        "wave": wave,
                        "slot_order": slot,
                        "model": model_key,
                        "variant": variant,
                        "seed": seed,
                    })
    return queue


# ----------------------------------------------------------------------------
# Smoke trainfile builder
# ----------------------------------------------------------------------------

SMOKE_N_ORGANIC = 50
SMOKE_K_DUP = 2
SMOKE_SEED = 42


def build_smoke_trainfile() -> Path:
    """50 organic + 20 canaries x K=2 = 90 records (deterministic, seed 42)."""
    src = DATA_DIR / "train_v0.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing - run 00_prepare_data.py first")

    organic: list[dict] = []
    canary_reps: dict[str, list[dict]] = {}
    with src.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id", "")
            if isinstance(rid, str) and rid.startswith("canary-"):
                # id = "canary-<CANARY_ID>-rep-NN" where CANARY_ID = "A-IP4-01" (3 segments)
                key = "-".join(rid.split("-")[:4])
                canary_reps.setdefault(key, []).append(rec)
            else:
                organic.append(rec)

    rng = random.Random(SMOKE_SEED)
    rng.shuffle(organic)
    chosen_organic = organic[:SMOKE_N_ORGANIC]

    chosen_canaries: list[dict] = []
    for key in sorted(canary_reps):
        reps = sorted(canary_reps[key], key=lambda r: r["id"])
        chosen_canaries.extend(reps[:SMOKE_K_DUP])

    out = chosen_organic + chosen_canaries
    rng.shuffle(out)

    dst = DATA_DIR / "train_v0_smoke.jsonl"
    with dst.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    fp = hashlib.sha256(b"|".join(r["id"].encode() for r in out)).hexdigest()[:16]
    print(f"smoke train: {len(out)} records ({len(chosen_organic)} organic + {len(chosen_canaries)} canaries), fp={fp}")
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue-out", type=Path, default=CONFIG_DIR / "queue.json")
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args(argv)

    if not args.skip_smoke:
        build_smoke_trainfile()
    queue = build_queue()
    args.queue_out.write_text(json.dumps(queue, indent=2))
    print(f"wrote {len(queue)} configs ({len(WAVES)} waves × {len(MODELS)} models × {len(VARIANTS)} variants) → {args.queue_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
