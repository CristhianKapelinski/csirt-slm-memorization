#!/usr/bin/env bash
# Reproduce every numerical claim in the paper from the shipped eval data (no training, ~30s).

set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"

# Pick a Python interpreter
if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi
echo "==> Using $PY ($($PY --version 2>&1))"

$PY - <<'EOF'
import sys
need = ["numpy", "scipy"]
missing = []
for m in need:
    try: __import__(m)
    except ImportError: missing.append(m)
if missing:
    print("MISSING:", missing)
    print("Run: pip install numpy scipy   (or: pip install .[reproduce])")
    sys.exit(1)
EOF

# No training corpus needed here: this path only analyzes pre-computed eval results.

# Regenerate every report cited by the paper
mkdir -p experiment/reports

run() {
    local label="$1"; shift
    echo "==> [$label] $*"
    "$@"
}

run "Table 1       (Exp 1: V0/V1/V2/V3 exposure)" \
    $PY experiment/scripts/analyze_final.py
# analyze_final writes to experiment/; move it into reports/
[ -f experiment/FINAL_REPORT.md ] && mv experiment/FINAL_REPORT.md experiment/reports/FINAL_REPORT.md

run "Table 2       (Exp 2: AnonShield V0 reduction)" \
    $PY experiment/scripts/analyze_anonshield.py

run "Table 3      (Attack 2: slug exposure)" \
    $PY experiment/scripts/analyze_slug.py

run "Table 4      (format ablation)" \
    $PY experiment/scripts/analyze_ablation.py

run "Table 5      (utility cross-seed adapters)" \
    $PY experiment/scripts/analyze_utility_logits.py

run "Table 6      (utility prompt-eng base models)" \
    $PY experiment/scripts/analyze_prompt.py

run "Finding (i)   (matched-update control, C1)" \
    $PY experiment/scripts/analyze_matched_update.py

run "Sec 4.4       (cross-seed stack ablation + bootstrap CIs)" \
    $PY experiment/scripts/analyze_ablation_stack.py

# Verify paper claims numerically
echo
$PY scripts/verify_claims.py

echo
echo "================================================================="
echo "  Reports regenerated at experiment/reports/:"
echo "    FINAL_REPORT.md             (Table 1)"
echo "    ANONSHIELD_REPORT.md        (Table 2)"
echo "    SLUG_ATTACK_REPORT.md       (Table 3)"
echo "    UTILITY_ABLATION_REPORT.md  (Table 4)"
echo "    UTILITY_REPORT.md           (Table 5)"
echo "    UTILITY_PROMPT_REPORT.md    (Table 6)"
echo "    MATCHED_UPDATE_REPORT.md    (Finding i / C1)"
echo "    ABLATION_STACK_REPORT.md    (Sec 4.4 decomposition)"
echo "================================================================="
