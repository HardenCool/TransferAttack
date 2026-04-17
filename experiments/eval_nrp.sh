#!/usr/bin/env bash
# =============================================================================
# eval_nrp.sh
# Evaluate NRP (Neural Representation Purifier) defense against DIM, SGM,
# MIG, OPS, and MUMODIG adversarial examples.
#
# Prerequisites:
#   - defense/models/NRP.pth  downloaded from
#     https://drive.google.com/drive/folders/1NfSjLzc-MtkYHLumcKYs6OqC2X_zWy3g
#
# Usage:
#   bash experiments/eval_nrp.sh [DATA_DIR] [GPU_ID]
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${REPO_ROOT}/data}"
GPU_ID="${2:-0}"
SURROGATE="resnet18"
NRP_MODEL="${REPO_ROOT}/defense/models/NRP.pth"
RESULTS_DIR="${REPO_ROOT}/experiments/results/nrp"

mkdir -p "${RESULTS_DIR}"

# Attacks to evaluate
ATTACKS="dim sgm mig ops mumodig"

echo "============================================================"
echo " NRP defense evaluation"
echo " NRP model  : ${NRP_MODEL}"
echo " Data dir   : ${DATA_DIR}"
echo " Results    : ${RESULTS_DIR}"
echo "============================================================"

for ATTACK in ${ATTACKS}; do
    ADV_DIR="${REPO_ROOT}/adv_data/${ATTACK}/${SURROGATE}"

    if [ ! -d "${ADV_DIR}" ]; then
        echo "[SKIP] ${ATTACK}: adversarial directory not found (${ADV_DIR})"
        continue
    fi

    PURIFIED_DIR="${REPO_ROOT}/defense/nrp/purified_data/${ATTACK}/${SURROGATE}"
    echo ""
    echo ">>> [NRP / ${ATTACK}]  purifying -> ${PURIFIED_DIR}"

    # Step 1: purify adversarial images
    python "${REPO_ROOT}/defense/nrp/purify.py" \
        --dir      "${ADV_DIR}" \
        --output   "${PURIFIED_DIR}" \
        --purifier NRP \
        --model_pth "${NRP_MODEL}" \
        --dynamic \
        --GPU_ID   "${GPU_ID}" \
        2>&1 | tee "${RESULTS_DIR}/${ATTACK}_purify.log"

    # Step 2: evaluate purified images (reports ASR on standard 8-model suite)
    echo "    evaluating purified images..."
    python "${REPO_ROOT}/main.py" \
        --input_dir  "${DATA_DIR}" \
        --output_dir "defense/nrp/purified_data/${ATTACK}/${SURROGATE}" \
        --eval \
        --GPU_ID     "${GPU_ID}" \
        2>&1 | tee "${RESULTS_DIR}/${ATTACK}_eval.log"

    echo "<<< [NRP / ${ATTACK}] done."
done

echo ""
echo "NRP evaluation complete. Logs in ${RESULTS_DIR}/"
