#!/usr/bin/env bash
# One-click pipeline:
#   Step 1 – Generate MOMUDIG adversarial examples
#   Step 2 – DiffPure purification  (existing script)
#   Step 3 – Ours purification      (SDEdit + SD2.1)
#   Step 4 – Generate comparison figures
#
# Prerequisites:
#   1. Download SD2.1 weights once:
#        python defense/ours/download_weights.py
#   2. Obtain WaveDM purified images and set WAVEDM_DIR below.
#
# Usage (run from repo root):
#   sh defense/ours_defense.sh

# ──────────────────────────────────────────────────────────────────────────────
# User-configurable paths
# ──────────────────────────────────────────────────────────────────────────────
ATTACK_METHOD=momudig
SOURCE_MODEL=resnet18

IMAGE_FOLDER=../../path/to/data/images   # clean ImageNet validation images
ADV_DIR=../../adv_data/${ATTACK_METHOD}/${SOURCE_MODEL}

DIFFPURE_OUTPUT=../../purified/${ATTACK_METHOD}/diffpure
WAVEDM_DIR=../../purified/${ATTACK_METHOD}/wavedm   # set to your WaveDM output
OURS_OUTPUT=../../purified/${ATTACK_METHOD}/ours
COMPARISON_OUTPUT=../../purified/${ATTACK_METHOD}/comparison

MODEL_PATH=defense/models/sd2.1   # path to downloaded SD2.1 weights

STRENGTH=0.40
NUM_STEPS=50
BATCH_SIZE=4

CUDA_VISIBLE_DEVICES=0

# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – Generate MOMUDIG adversarial examples
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Step 1: Generating adversarial examples with MOMUDIG ..."
python main.py \
    --attack ${ATTACK_METHOD} \
    --model  ${SOURCE_MODEL} \
    --input_dir  "${IMAGE_FOLDER}" \
    --output_dir "${ADV_DIR}" \
    --GPU_ID ${CUDA_VISIBLE_DEVICES}

# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – DiffPure purification (existing defense)
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Step 2: DiffPure purification ..."
mkdir -p "${DIFFPURE_OUTPUT}"
cd defense
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python diffpure/diffpure.py \
    --image_folder "${IMAGE_FOLDER}" \
    --adv_dir "${ADV_DIR}" \
    --config imagenet.yml \
    --t 150 --adv_eps 0.0157 --adv_batch_size ${BATCH_SIZE} \
    --domain imagenet --classifier_name resnet101 \
    --diffusion_type sde
cd ..

# ──────────────────────────────────────────────────────────────────────────────
# Step 3 – Ours purification (SDEdit + SD2.1)
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Step 3: Ours purification with SDEdit + SD2.1 ..."
mkdir -p "${OURS_OUTPUT}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python defense/ours/purify_ours.py \
    --adv_dir      "${ADV_DIR}" \
    --output_dir   "${OURS_OUTPUT}" \
    --model_path   "${MODEL_PATH}" \
    --strength     ${STRENGTH} \
    --num_inference_steps ${NUM_STEPS} \
    --batch_size   ${BATCH_SIZE}

# ──────────────────────────────────────────────────────────────────────────────
# Step 4 – Generate comparison figures
# ──────────────────────────────────────────────────────────────────────────────
echo "==> Step 4: Generating comparison figures ..."
mkdir -p "${COMPARISON_OUTPUT}"
python defense/ours/generate_comparison.py \
    --orig_dir     "${IMAGE_FOLDER}" \
    --adv_dir      "${ADV_DIR}" \
    --diffpure_dir "${DIFFPURE_OUTPUT}" \
    --wavedm_dir   "${WAVEDM_DIR}" \
    --ours_dir     "${OURS_OUTPUT}" \
    --output_dir   "${COMPARISON_OUTPUT}"

echo "==> Done!  Comparison figures saved to: ${COMPARISON_OUTPUT}"
