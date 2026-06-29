#!/usr/bin/env bash
# Path B: full reproduction from scratch on a GPU (~222 GPU-h).
set -euo pipefail

need_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "WARNING: nvidia-smi not found. Path B trains models and needs a GPU."
        echo "WARNING: continuing, but expect this to be impractical on CPU."
    fi
}

need_gpu

echo "==> [setup] installing training dependencies"
pip install ".[train]"

echo "==> [data] preparing datasets (skipped when the shipped splits are present)"
[ -f experiment/data/train_v0.jsonl ] || python experiment/scripts/00_prepare_data.py

echo "==> [anon] fetching AnonShield and pseudonymizing datasets"
./fetch_anonshield.sh
ANON_SECRET_KEY="${ANON_SECRET_KEY:-artifact-anon-key-v1}"
export ANON_SECRET_KEY
docker run --rm -v "$PWD":/data anonshield/anon \
    --in /data/experiment/data/train_v0.jsonl \
    --out /data/experiment/data/train_anon.jsonl \
    --key "$ANON_SECRET_KEY"
docker run --rm -v "$PWD":/data anonshield/anon \
    --in /data/experiment/data/utility_held_out.jsonl \
    --out /data/experiment/data/utility_held_out_anon.jsonl \
    --key "$ANON_SECRET_KEY"

echo "==> [configs] building experiment configs"
python experiment/scripts/build_configs.py
python experiment/scripts/build_configs_exp2.py

echo "==> [train] training one model per config (skips configs whose eval result exists)"
for cfg in experiment/configs/*.yaml; do
    run_id="$(basename "$cfg" .yaml)"
    [ -f "experiment/results/eval_checkpoints/${run_id}.jsonl" ] && continue
    python experiment/scripts/train.py --config "$cfg"
done

echo "==> [eval] scanning and evaluating results"
python experiment/scripts/eval.py --scan

echo "==> [utility] evaluating utility (logits + prompt)"
python experiment/scripts/utility_eval_logits.py --scan
python experiment/scripts/utility_prompt_eval.py --scan

echo "==> [slug] running slug extraction attack"
python experiment/scripts/eval_slug_attack.py --scan

echo "==> [report] building tables and figures"
./reproduce.sh
