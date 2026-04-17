#!/usr/bin/env bash
# =============================================================================
# eval_clean.sh
# Evaluate clean (unperturbed) accuracy across the standard 8-model suite
# (4 CNNs + 4 ViTs) defined in transferattack/utils.py.
#
# Usage:
#   bash experiments/eval_clean.sh [DATA_DIR] [GPU_ID]
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${REPO_ROOT}/data}"
GPU_ID="${2:-0}"
SURROGATE="resnet18"

echo ">>> Evaluating clean accuracy on ${DATA_DIR}/images"
python "${REPO_ROOT}/main.py" \
    --input_dir  "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/images" \
    --eval \
    --GPU_ID "${GPU_ID}" \
    | tee "${REPO_ROOT}/experiments/clean_acc.txt"

echo "Clean accuracy results saved to experiments/clean_acc.txt"
