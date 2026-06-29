"""LoRA SFT training driver - four variants V0/V1/V2/V3."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "cache" / "hf" / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer


LORA_EXCLUDE_PATTERN = r".*(vision_tower|audio_tower|embed_vision|embed_audio)\..*"


@dataclasses.dataclass(frozen=True)
class RunConfig:
    run_id: str
    model_name: str
    variant: str
    seed: int
    train_file: Path
    num_epochs: float
    batch_size: int
    grad_accum: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    max_seq_length: int
    save_steps: int
    save_total_limit: int
    logging_steps: int
    lossguard_min: float
    adapters_dir: Path
    telemetry_file: Path
    bf16: bool
    dp_target_epsilon: float
    dp_target_delta: float
    dp_max_grad_norm: float
    dp_lot_size: int
    dp_grad_sample_mode: str
    quant_mode: str = "auto"
    sampling_mode: str = "auto"

    @property
    def uses_qlora(self) -> bool:
        if self.quant_mode == "nf4":
            return True
        if self.quant_mode == "bf16":
            return False
        return self.variant in ("v1", "v2", "v3", "vb")

    @property
    def uses_dp(self) -> bool:
        # vd: DP path with noise=0
        return self.variant in ("v2", "v3", "vd")

    @property
    def is_ablation(self) -> bool:
        return self.variant in ("va", "vb", "vc", "vd")

    @property
    def use_poisson(self) -> bool:
        if self.sampling_mode == "poisson":
            return True
        if self.sampling_mode == "uniform":
            return False
        return False

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        raw = yaml.safe_load(path.read_text())
        return cls(
            run_id=raw["run_id"],
            model_name=raw["model_name"],
            variant=raw["variant"],
            seed=int(raw["seed"]),
            train_file=Path(raw["train_file"]),
            num_epochs=float(raw.get("num_epochs", 3)),
            batch_size=int(raw.get("batch_size", 1)),
            grad_accum=int(raw.get("grad_accum", 4)),
            learning_rate=float(raw.get("learning_rate", 1e-4)),
            lora_r=int(raw.get("lora_r", 16)),
            lora_alpha=int(raw.get("lora_alpha", 32)),
            lora_dropout=float(raw.get("lora_dropout", 0.05)),
            target_modules=tuple(raw.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
            max_seq_length=int(raw.get("max_seq_length", 1024)),
            save_steps=int(raw.get("save_steps", 500)),
            save_total_limit=int(raw.get("save_total_limit", 3)),
            logging_steps=int(raw.get("logging_steps", 10)),
            lossguard_min=float(raw.get("lossguard_min", 0.3)),
            adapters_dir=Path(raw.get("adapters_dir", ROOT / "experiment" / "adapters")),
            telemetry_file=Path(raw.get("telemetry_file", ROOT / "experiment" / "results" / "train_telemetry" / f"{raw['run_id']}.jsonl")),
            bf16=bool(raw.get("bf16", True)),
            dp_target_epsilon=float(raw.get("dp_target_epsilon", 8.0)),
            dp_target_delta=float(raw.get("dp_target_delta", 1e-5)),
            dp_max_grad_norm=float(raw.get("dp_max_grad_norm", 1.0)),
            dp_lot_size=int(raw.get("dp_lot_size", 32)),
            dp_grad_sample_mode=str(raw.get("dp_grad_sample_mode", "ghost")),
            quant_mode=str(raw.get("quant_mode", "auto")),
            sampling_mode=str(raw.get("sampling_mode", "auto")),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except RuntimeError:
        pass


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def serialize(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


class LossGuard(TrainerCallback):
    """Abort if loss < min_loss (deterministic collapse). V0 only."""
    def __init__(self, min_loss: float):
        self.min_loss = min_loss
        self.triggered = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or self.triggered:
            return
        loss = logs.get("loss")
        if isinstance(loss, (int, float)) and loss < self.min_loss:
            self.triggered = True
            control.should_training_stop = True
            print(f"[LossGuard] loss={loss:.4f} < {self.min_loss}; stopping", flush=True)


class Telemetry(TrainerCallback):
    """JSONL per step + per epoch. Compatible with SFTTrainer (V0) and the manual loop (V1/V2/V3)."""
    def __init__(self, run_id: str, variant: str, model_name: str, seed: int, path: Path):
        self.meta = {"run_id": run_id, "variant": variant, "model": model_name, "seed": seed}
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")
        self.start = time.monotonic()
        self.epoch_losses: list[float] = []

    def write(self, payload: dict) -> None:
        self.fh.write(json.dumps({**self.meta, **payload, "wallclock_s": time.monotonic() - self.start}) + "\n")
        self.fh.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        loss = logs.get("loss")
        if isinstance(loss, (int, float)):
            self.epoch_losses.append(loss)
        self.write({
            "kind": "step",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "loss": loss,
            "lr": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "vram_mb": _vram_mb(),
        })

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.epoch_losses:
            return
        arr = np.array(self.epoch_losses)
        self.write({
            "kind": "epoch",
            "epoch": state.epoch,
            "global_step": state.global_step,
            "train_loss_mean": float(arr.mean()),
            "train_loss_std": float(arr.std()),
            "train_loss_min": float(arr.min()),
            "train_loss_max": float(arr.max()),
        })
        self.epoch_losses = []

    def on_train_end(self, args, state, control, **kwargs):
        self.write({"kind": "end", "global_step": state.global_step, "epoch": state.epoch})
        self.fh.close()

    def close(self) -> None:
        if not self.fh.closed:
            self.fh.close()


def _vram_mb() -> int:
    return torch.cuda.max_memory_allocated() // (1024 ** 2) if torch.cuda.is_available() else 0


def _unwrap_custom_linears(model) -> None:
    """Replace custom Linear wrappers with the inner nn.Linear so PEFT can attach LoRA."""
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            inner = getattr(child, "linear", None)
            if isinstance(inner, nn.Linear) and not isinstance(child, nn.Linear):
                setattr(parent, child_name, inner)


def _lora_config(cfg: RunConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.target_modules),
        exclude_modules=LORA_EXCLUDE_PATTERN,
    )


def build_model_v0(cfg: RunConfig):
    """V0: bf16 base + LoRA + GC use_reentrant=False."""
    dtype = torch.bfloat16 if cfg.bf16 and torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    _unwrap_custom_linears(model)
    # Ghost Clipping needs untied embeddings
    if cfg.uses_dp:
        _untie_lm_head(model)
    if hasattr(model, "gradient_checkpointing_enable") and not cfg.uses_dp:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    model = get_peft_model(model, _lora_config(cfg))
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tok


def build_model_qlora(cfg: RunConfig):
    """V1/V2/V3: 4-bit NF4 + LoRA + GC + SDPA."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    _unwrap_custom_linears(model)
    # Ghost Clipping (V2/V3) rejects tying + grad-checkpointing; V1 keeps GC
    _untie_lm_head(model)
    use_gc = not cfg.uses_dp
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
    )
    model = get_peft_model(model, _lora_config(cfg))
    return model, tok


def _untie_lm_head(model) -> None:
    """Break input-embedding/lm_head weight tying for Ghost Clipping; no-op if untied."""
    in_emb = getattr(model, "get_input_embeddings", lambda: None)()
    lm_head = getattr(model, "lm_head", None) or getattr(model, "get_output_embeddings", lambda: None)()
    if in_emb is None or lm_head is None:
        return
    in_w = getattr(in_emb, "weight", None)
    out_w = getattr(lm_head, "weight", None)
    if in_w is None or out_w is None:
        return
    if in_w.data_ptr() == out_w.data_ptr():
        lm_head.weight = nn.Parameter(out_w.clone().detach())
        if hasattr(model, "config"):
            model.config.tie_word_embeddings = False


def _build_dataset(path: Path) -> Dataset:
    return Dataset.from_dict({"text": [serialize(r) for r in load_records(path)]})


def train_v0(cfg: RunConfig, resume: str | None) -> int:
    seed_everything(cfg.seed)
    dataset = _build_dataset(cfg.train_file)
    model, tok = build_model_v0(cfg)
    out_dir = cfg.adapters_dir / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sft = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16 and torch.cuda.is_available(),
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        dataset_text_field="text",
        max_length=cfg.max_seq_length,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tok,
        train_dataset=dataset,
        args=sft,
        callbacks=[
            LossGuard(cfg.lossguard_min),
            Telemetry(cfg.run_id, cfg.variant, cfg.model_name, cfg.seed, cfg.telemetry_file),
        ],
    )
    if resume:
        trainer.train(resume_from_checkpoint=resume)
    else:
        trainer.train()
    trainer.save_model(str(out_dir / "marker_final"))
    return 0


def _make_loader(records: list[dict], tok, max_len: int, batch_size: int,
                 poisson_lot: int | None = None) -> DataLoader:
    """DataLoader; poisson_lot switches to Opacus UniformWithReplacementSampler."""
    enc = tok([serialize(r) for r in records], max_length=max_len, truncation=True,
              padding="max_length", return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]

    class LMDataset(TorchDataset):
        def __len__(self) -> int:
            return int(ids.size(0))
        def __getitem__(self, i: int) -> dict:
            labels = ids[i].clone()
            labels[mask[i] == 0] = -100
            return {"input_ids": ids[i], "attention_mask": mask[i], "labels": labels}

    ds = LMDataset()
    if poisson_lot is not None:
        from opacus.utils.uniform_sampler import UniformWithReplacementSampler
        sampler = UniformWithReplacementSampler(
            num_samples=len(ds), sample_rate=poisson_lot / len(ds))
        return DataLoader(ds, batch_sampler=sampler, num_workers=0)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


def _safe_eps(engine, delta: float) -> float:
    try:
        eps = float(engine.get_epsilon(delta))
        return eps if math.isfinite(eps) else float("inf")
    except Exception:
        return float("inf")


def _shift_logits_labels(logits, labels):
    return (logits[..., :-1, :].contiguous(),
            labels[..., 1:].contiguous())


def train_manual(cfg: RunConfig, resume: str | None) -> int:
    """Manual loop: V1 (no DP) + V2/V3 (Ghost Clipping)."""
    if resume:
        raise NotImplementedError("resume not wired (Opacus state dict does not persist)")

    seed_everything(cfg.seed)
    records = load_records(cfg.train_file)
    if cfg.uses_qlora:
        model, tok = build_model_qlora(cfg)
    else:
        model, tok = build_model_v0(cfg)
    out_dir = cfg.adapters_dir / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # V1: batch=1 + manual accum; V2/V3: batch=lot_size split by BatchMemoryManager
    loader_batch = cfg.dp_lot_size if cfg.uses_dp else 1
    accum_steps = 1 if cfg.uses_dp else cfg.dp_lot_size
    poisson_lot = cfg.dp_lot_size if (cfg.use_poisson and not cfg.uses_dp) else None
    loader = _make_loader(records, tok, cfg.max_seq_length, loader_batch,
                          poisson_lot=poisson_lot)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    privacy_engine = None
    if cfg.uses_dp:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        # vd: noise~0 + clip~inf to isolate sampling/microbatching without DP noise
        if cfg.variant == "vd":
            wrapped = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=loader,
                noise_multiplier=0.0,
                max_grad_norm=1e9,
                grad_sample_mode=cfg.dp_grad_sample_mode,
            )
        else:
            wrapped = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                criterion=criterion,
                data_loader=loader,
                target_epsilon=cfg.dp_target_epsilon,
                target_delta=cfg.dp_target_delta,
                max_grad_norm=cfg.dp_max_grad_norm,
                epochs=int(cfg.num_epochs),
                grad_sample_mode=cfg.dp_grad_sample_mode,
            )
        if len(wrapped) == 4:
            model, optimizer, criterion, loader = wrapped
        else:
            model, optimizer, loader = wrapped

    # rounds = optimizer steps, not micro-batches
    rounds = max(1, (len(loader) // max(1, accum_steps)) * int(cfg.num_epochs))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=rounds)
    telem = Telemetry(cfg.run_id, cfg.variant, cfg.model_name, cfg.seed, cfg.telemetry_file)
    device = next(model.parameters()).device

    print(f"[{cfg.variant}] records={len(records)} lot={cfg.dp_lot_size} "
          f"epochs={int(cfg.num_epochs)} dp={'on' if cfg.uses_dp else 'off'}"
          + (f" ε_target={cfg.dp_target_epsilon} δ={cfg.dp_target_delta}" if cfg.uses_dp else ""),
          flush=True)

    global_step = 0
    micro_step = 0
    physical_bs = max(1, cfg.batch_size)

    for epoch in range(int(cfg.num_epochs)):
        model.train()
        losses: list[float] = []

        if cfg.uses_dp:
            from opacus.utils.batch_memory_manager import BatchMemoryManager
            ctx = BatchMemoryManager(data_loader=loader, max_physical_batch_size=physical_bs, optimizer=optimizer)
        else:
            ctx = _NullContext(loader)

        with ctx as iter_loader:
            for batch in iter_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                shift_logits, shift_labels = _shift_logits_labels(out.logits, batch["labels"])
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # V1 normalizes for accumulation; Opacus normalizes internally
                back_loss = loss / accum_steps if accum_steps > 1 else loss
                back_loss.backward()
                micro_step += 1
                if loss.item() != 0.0:
                    losses.append(loss.item())

                if cfg.uses_dp or micro_step % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    sched.step()
                    global_step += 1
                    if global_step % cfg.logging_steps == 0:
                        payload = {
                            "kind": "step",
                            "global_step": global_step,
                            "epoch": epoch + 1,
                            "loss": float(loss.item()),
                            "lr": sched.get_last_lr()[0],
                            "vram_mb": _vram_mb(),
                        }
                        if privacy_engine is not None:
                            payload["dp_epsilon"] = _safe_eps(privacy_engine, cfg.dp_target_delta)
                        telem.write(payload)

        arr = np.array(losses) if losses else np.array([float("nan")])
        epoch_payload = {
            "kind": "epoch",
            "epoch": float(epoch + 1),
            "global_step": global_step,
            "train_loss_mean": float(arr.mean()),
            "train_loss_std": float(arr.std()),
            "train_loss_min": float(arr.min()),
            "train_loss_max": float(arr.max()),
        }
        if privacy_engine is not None:
            epoch_payload["dp_epsilon"] = _safe_eps(privacy_engine, cfg.dp_target_delta)
        telem.write(epoch_payload)

        ckpt = out_dir / f"checkpoint-epoch{epoch + 1}"
        ckpt.mkdir(parents=True, exist_ok=True)
        peft_model = getattr(model, "_module", model)
        if hasattr(peft_model, "save_pretrained"):
            peft_model.save_pretrained(str(ckpt))
        eps_str = f", ε={epoch_payload.get('dp_epsilon', '—')}" if privacy_engine is not None else ""
        print(f"[{cfg.variant}] epoch {epoch + 1}/{int(cfg.num_epochs)} done{eps_str}", flush=True)

    telem.write({"kind": "end", "global_step": global_step, "epoch": float(cfg.num_epochs)})
    telem.close()
    return 0


class _NullContext:
    """Context-manager wrapper so the non-DP DataLoader uses the same `with` as BatchMemoryManager."""
    def __init__(self, loader):
        self.loader = loader
    def __enter__(self):
        return self.loader
    def __exit__(self, *exc):
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--resume-from-checkpoint", dest="resume", default=None)
    args = ap.parse_args(argv)
    cfg = RunConfig.from_yaml(args.config)
    if os.environ.get("SKIP_IF_DONE") == "1":
        out_dir = cfg.adapters_dir / cfg.run_id
        for marker in ("marker_final", "checkpoint-epoch3"):
            p = out_dir / marker / "adapter_config.json"
            if p.exists():
                print(f"[skip] {cfg.run_id} already done ({marker})", flush=True)
                return 0
    if cfg.variant == "v0":
        return train_v0(cfg, args.resume)
    if cfg.variant in ("v1", "v2", "v3", "va", "vb", "vc", "vd"):
        return train_manual(cfg, args.resume)
    raise ValueError(f"unknown variant {cfg.variant!r}; expected v0/v1/v2/v3/va/vb/vc/vd")


if __name__ == "__main__":
    sys.exit(main())
