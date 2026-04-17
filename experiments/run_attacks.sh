#!/usr/bin/env bash
# =============================================================================
# run_attacks.sh
# Generate adversarial examples for DIM, SGM, MIG, OPS, and MUMODIG using
# resnet18 as the surrogate model.  All five attacks are now implemented
# natively in this repository.
#
# Usage:
#   bash experiments/run_attacks.sh [DATA_DIR] [GPU_ID]
#
# Arguments:
#   DATA_DIR  Path to the ImageNet subset (must contain images/ and labels.csv).
#             Default: ./data
#   GPU_ID    CUDA device to use. Default: 0
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${REPO_ROOT}/data}"
GPU_ID="${2:-0}"
SURROGATE="resnet18"
ADV_ROOT="${REPO_ROOT}/adv_data"

echo "============================================================"
echo " Repository  : ${REPO_ROOT}"
echo " Data dir    : ${DATA_DIR}"
echo " Surrogate   : ${SURROGATE}"
echo " GPU         : ${GPU_ID}"
echo " Adv output  : ${ADV_ROOT}"
echo "============================================================"

# ---------- helper -----------------------------------------------------------
run_attack() {
    local ATTACK="$1"
    local OUT_DIR="${ADV_ROOT}/${ATTACK}/${SURROGATE}"
    echo ""
    echo ">>> [${ATTACK}] generating adversarial examples -> ${OUT_DIR}"
    python "${REPO_ROOT}/main.py" \
        --input_dir  "${DATA_DIR}" \
        --output_dir "${OUT_DIR}" \
        --attack     "${ATTACK}" \
        --model      "${SURROGATE}" \
        --epoch      10 \
        --eps        "$(python -c 'print(16/255)')" \
        --alpha      "$(python -c 'print(1.6/255)')" \
        --momentum   1.0 \
        --GPU_ID     "${GPU_ID}"
    echo "<<< [${ATTACK}] done."
}

# ---------- DIM --------------------------------------------------------------
run_attack "dim"

# ---------- SGM --------------------------------------------------------------
run_attack "sgm"

# ---------- MIG (integrated-gradient transfer) -------------------------------
echo ""
echo ">>> [mig] generating adversarial examples -> ${ADV_ROOT}/mig/${SURROGATE}"
python "${REPO_ROOT}/main.py" \
    --input_dir  "${DATA_DIR}" \
    --output_dir "${ADV_ROOT}/mig/${SURROGATE}" \
    --attack     "mig" \
    --model      "${SURROGATE}" \
    --epoch      10 \
    --eps        "$(python -c 'print(16/255)')" \
    --GPU_ID     "${GPU_ID}"
echo "<<< [mig] done."

# ---------- OPS (Operator-Perturbation Stochastic) ---------------------------
# Uses operator sampling (20 ops) + perturbation neighbourhood (10 neighbours).
# Set num_sample_neighbor / num_sample_operator via the OPS class defaults.
run_attack "ops"

# ---------- MUMODIG (Multi-baseline Monotone DIG) ----------------------------
# Uses LB-quantized baseline + expectation-over-transforms IG (6 transforms).
run_attack "mumodig"

echo ""
echo "============================================================"
echo " Attack generation complete."
echo " Adversarial examples saved to:"
for ATK in dim sgm mig ops mumodig; do
    echo "   ${ADV_ROOT}/${ATK}/${SURROGATE}/"
done
echo "============================================================"

