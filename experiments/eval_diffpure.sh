#!/usr/bin/env bash
# =============================================================================
# eval_diffpure.sh
# Evaluate DiffPure (SDE variant, ImageNet) defense against DIM, SGM, MIG,
# OPS, and MUMODIG adversarial examples.
#
# Prerequisites:
#   - defense/models/256x256_diffusion_uncond.pt  (OpenAI guided-diffusion)
#     Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
#   - CUDA environment with torchsde, yaml, timm, etc. installed
#
# Usage:
#   bash experiments/eval_diffpure.sh [DATA_DIR] [GPU_ID]
#
# Note: DiffPure is slow (~1 h per 1000 images at batch-size 4 on a 4090).
#       Reduce --num_sub for a quick sanity check.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-${REPO_ROOT}/data}"
GPU_ID="${2:-0}"
SURROGATE="resnet18"
DIFFPURE_DIR="${REPO_ROOT}/defense/diffpure"
RESULTS_DIR="${REPO_ROOT}/experiments/results/diffpure"

mkdir -p "${RESULTS_DIR}"

# Classifier used inside DiffPure (cross-model eval target)
CLASSIFIER="resnet101"

# Attacks to evaluate
ATTACKS="dim sgm mig ops mumodig"

echo "============================================================"
echo " DiffPure defense evaluation (diffusion_type=sde)"
echo " Classifier  : ${CLASSIFIER}"
echo " Data dir    : ${DATA_DIR}"
echo " Results     : ${RESULTS_DIR}"
echo "============================================================"

for ATTACK in ${ATTACKS}; do
    ADV_DIR="${REPO_ROOT}/adv_data/${ATTACK}/${SURROGATE}"

    if [ ! -d "${ADV_DIR}" ]; then
        echo "[SKIP] ${ATTACK}: adversarial directory not found (${ADV_DIR})"
        continue
    fi

    echo ""
    echo ">>> [DiffPure / ${ATTACK}]  adv_dir=${ADV_DIR}"

    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python "${DIFFPURE_DIR}/diffpure.py" \
        --image_folder    "${DATA_DIR}" \
        --adv_dir         "${ADV_DIR}" \
        --config          imagenet.yml \
        --t               150 \
        --adv_eps         0.0627 \
        --adv_batch_size  4 \
        --num_sub         1000 \
        --domain          imagenet \
        --classifier_name "${CLASSIFIER}" \
        --diffusion_type  sde \
        --score_type      guided_diffusion \
        2>&1 | tee "${RESULTS_DIR}/${ATTACK}.log"

    echo "<<< [DiffPure / ${ATTACK}] done."
done

echo ""
echo "DiffPure evaluation complete. Logs in ${RESULTS_DIR}/"
echo "ASR lines are tagged as 'ASR:XX.XX%' in each log file."
