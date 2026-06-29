"""Generate Exp 2 YAMLs (HMAC pseudonymization, Form B) per (model x variant x seed x wave)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
CONFIG_DIR = PILOT / "configs"
DATA_DIR = PILOT / "data"
TRAIN_FILE = str(DATA_DIR / "train_anon.jsonl")

MODELS = {
    "gemma3-1b":     "unsloth/gemma-3-1b-it",
    "qwen3-1.7b":    "Qwen/Qwen3-1.7B",
    "llama32-3b":    "unsloth/Llama-3.2-3B-Instruct",
    "vaultgemma-1b": "google/vaultgemma-1b",
}

# routes the heavy DP runs (Llama all, Qwen3 V2/V3) off the 12GB 3060
def goes_to_gpu2(model: str, variant: str) -> bool:
    if model == "llama32-3b":
        return False
    if model == "qwen3-1.7b" and variant in ("v2", "v3"):
        return False
    return True

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
    "bf16": True,
    "dp_target_epsilon": 8.0,
    "dp_target_delta": 1.0e-5,
    "dp_max_grad_norm": 1.0,
    "dp_lot_size": 32,
    "dp_grad_sample_mode": "ghost",
}

VARIANT_OVERRIDES = {
    "v0": {},
    "v1": {},
    "v2": {"dp_target_epsilon": 8.0},
    "v3": {"dp_target_epsilon": 2.0},
}

# Wave numbers outside the Exp 1 series (0-3) for a clean namespace.
WAVES = {
    4: {"seeds": [42]},
    5: {"seeds": [1337]},
    6: {"seeds": [2024]},
}


def run_id_for(model_key: str, variant: str, seed: int) -> str:
    return f"anon_{model_key}_{variant}_seed{seed}"


def build_config(model_key: str, variant: str, seed: int) -> dict:
    cfg = {
        "run_id": run_id_for(model_key, variant, seed),
        "model_name": MODELS[model_key],
        "variant": variant,
        "seed": seed,
        "train_file": TRAIN_FILE,
        **BASE,
        **VARIANT_OVERRIDES[variant],
    }
    if model_key == "llama32-3b" and variant in ("v2", "v3"):
        cfg["max_seq_length"] = 768
    return cfg


def write_config(cfg: dict) -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{cfg['run_id']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def build_queue() -> list[dict]:
    """Round-robin: variant -> model -> seed (same as Exp 1)."""
    queue: list[dict] = []
    slot = 0
    for wave in sorted(WAVES):
        spec = WAVES[wave]
        for variant in VARIANTS:
            for model_key in MODELS:
                for seed in spec["seeds"]:
                    cfg = build_config(model_key, variant, seed)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queue-out", type=Path, default=CONFIG_DIR / "queue_exp2.json")
    ap.add_argument("--gpu2-out", type=Path, default=CONFIG_DIR / "queue_exp2_gpu2.json")
    ap.add_argument("--main-out", type=Path, default=CONFIG_DIR / "queue_exp2_main.json")
    args = ap.parse_args(argv)

    if not Path(TRAIN_FILE).exists():
        print(f"ERROR: {TRAIN_FILE} does not exist - pseudonymize the splits first (see the anon stage of reproduce_full.sh)", file=sys.stderr)
        return 1

    queue = build_queue()
    args.queue_out.write_text(json.dumps(queue, indent=2))

    gpu2_q = [s for s in queue if goes_to_gpu2(s["model"], s["variant"])]
    main_q = [s for s in queue if not goes_to_gpu2(s["model"], s["variant"])]
    args.gpu2_out.write_text(json.dumps(gpu2_q, indent=2))
    args.main_out.write_text(json.dumps(main_q, indent=2))

    print(f"wrote {len(queue)} configs ({len(WAVES)} waves × {len(MODELS)} models × {len(VARIANTS)} variants)")
    print(f"  full queue:  {args.queue_out.relative_to(ROOT)}  ({len(queue)} slots)")
    print(f"  gpu2 queue:  {args.gpu2_out.relative_to(ROOT)}  ({len(gpu2_q)} slots)")
    print(f"  main queue:  {args.main_out.relative_to(ROOT)}  ({len(main_q)} slots)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
