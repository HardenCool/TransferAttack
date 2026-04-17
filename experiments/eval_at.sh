#!/usr/bin/env bash
# =============================================================================
# eval_at.sh
# Evaluate AT (Adversarial Training, fast_adversarial, 4px) defense against
# DIM, SGM, MIG, OPS, and MUMODIG adversarial examples.
#
# Prerequisites:
#   - defense/models/imagenet_model_weights_4px.pth.tar downloaded from
#     https://drive.google.com/drive/folders/1NfSjLzc-MtkYHLumcKYs6OqC2X_zWy3g
#     (or Huggingface mirror in defense README)
#
# Usage:
#   bash experiments/eval_at.sh [DATA_DIR] [GPU_ID]
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${REPO_ROOT}/data}"
GPU_ID="${2:-0}"
SURROGATE="resnet18"
CHECKPOINT="${REPO_ROOT}/defense/models/imagenet_model_weights_4px.pth.tar"
LABEL_FILE="${DATA_DIR}/labels.csv"
RESULTS_DIR="${REPO_ROOT}/experiments/results/at"
AT_DIR="${REPO_ROOT}/defense/at"

mkdir -p "${RESULTS_DIR}"

# Attacks to evaluate (OPS and MUMODIG dirs must be populated externally)
ATTACKS="dim sgm mig ops mumodig"

echo "============================================================"
echo " AT defense evaluation"
echo " Checkpoint : ${CHECKPOINT}"
echo " Label file : ${LABEL_FILE}"
echo " Results    : ${RESULTS_DIR}"
echo "============================================================"

for ATTACK in ${ATTACKS}; do
    ADV_DIR="${REPO_ROOT}/adv_data/${ATTACK}/${SURROGATE}"

    if [ ! -d "${ADV_DIR}" ]; then
        echo "[SKIP] ${ATTACK}: adversarial directory not found (${ADV_DIR})"
        continue
    fi

    OUTPUT_FILE="at_results/${ATTACK}_${SURROGATE}.txt"
    echo ""
    echo ">>> [AT / ${ATTACK}]  adv_dir=${ADV_DIR}"

    (
        cd "${AT_DIR}"
        python main_fast.py "${ADV_DIR}" \
            --config  configs/configs_fast_4px_evaluate.yml \
            --output_prefix "${OUTPUT_FILE}" \
            --resume  "${CHECKPOINT}" \
            --evaluate \
            --restarts 10 \
            --GPU_ID "${GPU_ID}" \
            2>&1 | tee "${AT_DIR}/${OUTPUT_FILE}.log"
    )

    # Parse ASR and convert to RA
    python "${REPO_ROOT}/defense/check_output.py" \
        --output_file "${AT_DIR}/${OUTPUT_FILE}" \
        --label_file  "${LABEL_FILE}" \
        | tee "${RESULTS_DIR}/${ATTACK}.txt"

    echo "<<< [AT / ${ATTACK}] done."
done

echo ""
echo "AT evaluation complete. Results in ${RESULTS_DIR}/"
