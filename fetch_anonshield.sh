#!/usr/bin/env bash
# Fetch the external AnonShield tool (idempotent).
set -euo pipefail

if command -v docker >/dev/null 2>&1; then
    docker pull anonshield/anon:latest
    echo "Selected backend: docker (anonshield/anon:latest)"
    echo "Usage example:"
    echo '  docker run --rm -v "$PWD":/data anonshield/anon \'
    echo '      --in /data/experiment/data/train_v0.jsonl \'
    echo '      --out /data/experiment/data/train_anon.jsonl \'
    echo '      --key "$ANON_SECRET_KEY"'
else
    if [ ! -d anonshield_ext ]; then
        git clone --depth 1 https://github.com/AnonShield/anonshield anonshield_ext
    fi
    (cd anonshield_ext && (uv sync || pip install -e .))
    echo "Selected backend: source checkout (anonshield_ext)"
fi
