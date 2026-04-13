#!/usr/bin/env bash
# ===========================================================================
# download_weights.sh
#
# Downloads all pretrained weights required by the comparison pipeline:
#
#   1. 256x256_diffusion_uncond.pt  — shared by DiffPure AND WaveDM
#   2. Stable Diffusion v1-5        — used by "Ours" (FreqLDM)
#      (downloaded automatically by HuggingFace Hub on first run, or manually
#       below if you prefer a local copy)
#
# Usage:
#   bash defense/ours/download_weights.sh [--model-dir <dir>]
#
# Default model dir: defense/models/
# ===========================================================================

set -e

MODEL_DIR="defense/models"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)
            MODEL_DIR="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$MODEL_DIR"

# ── 1. OpenAI Guided Diffusion  (DiffPure + WaveDM) ─────────────────────────
DDPM_PT="$MODEL_DIR/256x256_diffusion_uncond.pt"
if [ -f "$DDPM_PT" ]; then
    echo "[✓] $DDPM_PT already exists, skipping."
else
    echo "[→] Downloading 256x256_diffusion_uncond.pt (~2.2 GB) ..."
    wget -q --show-progress \
        -O "$DDPM_PT" \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
    echo "[✓] Saved to $DDPM_PT"
fi

# ── 2. Stable Diffusion v1-5  (Ours / FreqLDM) ──────────────────────────────
# Option A (recommended): let HuggingFace Hub cache it automatically.
#   The first call to OursPurifier() will trigger the download (~4 GB).
#   No action needed here.
#
# Option B: pre-download to a custom directory with huggingface-cli.
#   Uncomment and run the block below if you prefer an explicit local copy:
#
# SD_DIR="$MODEL_DIR/stable-diffusion-v1-5"
# if [ -d "$SD_DIR" ]; then
#     echo "[✓] Stable Diffusion v1-5 already exists at $SD_DIR, skipping."
# else
#     echo "[→] Downloading Stable Diffusion v1-5 (~4 GB) to $SD_DIR ..."
#     python - <<'PYEOF'
# from huggingface_hub import snapshot_download
# import os, sys
# local_dir = sys.argv[1] if len(sys.argv) > 1 else "defense/models/stable-diffusion-v1-5"
# snapshot_download(
#     repo_id="runwayml/stable-diffusion-v1-5",
#     local_dir=local_dir,
#     ignore_patterns=["*.msgpack", "*.ot", "flax_model*"],
# )
# PYEOF
#     echo "[✓] Stable Diffusion v1-5 saved to $SD_DIR"
#     echo "    Pass model_id='$SD_DIR' to OursPurifier() or set --ours_model in compare_purification.py"
# fi

echo ""
echo "All required weights are ready."
echo "  DiffPure / WaveDM weights : $DDPM_PT"
echo "  Ours / FreqLDM weights    : downloaded automatically on first run"
echo "                              (or set HF_HOME to a custom cache directory)"
