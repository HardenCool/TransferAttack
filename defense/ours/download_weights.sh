#!/usr/bin/env bash
# ===========================================================================
# download_weights.sh
#
# Downloads all pretrained weights required by the comparison pipeline:
#
#   256x256_diffusion_uncond.pt  — shared by DiffPure, WaveDM, AND Ours
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

# ── Download helper: wget → curl → Python (works on Linux, macOS, Windows Git Bash) ──
_download() {
    local url="$1"
    local dest="$2"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url"
    elif command -v python &>/dev/null || command -v python3 &>/dev/null; then
        local py
        py=$(command -v python3 2>/dev/null || command -v python)
        echo "[i] wget/curl not found — falling back to Python urllib ..."
        "$py" - "$url" "$dest" <<'PYEOF'
import sys, urllib.request
url, dest = sys.argv[1], sys.argv[2]
def _progress(count, block, total):
    if total > 0:
        pct = min(100, count * block * 100 // total)
        print(f"\r    {pct}%", end="", flush=True)
urllib.request.urlretrieve(url, dest, _progress)
print()
PYEOF
    else
        echo "[✗] No download tool found (tried wget, curl, python). Please install one and retry."
        exit 1
    fi
}

# ── OpenAI Guided Diffusion  (DiffPure + WaveDM + Ours) ─────────────────────
DDPM_PT="$MODEL_DIR/256x256_diffusion_uncond.pt"
if [ -f "$DDPM_PT" ]; then
    echo "[✓] $DDPM_PT already exists, skipping."
else
    echo "[→] Downloading 256x256_diffusion_uncond.pt (~2.2 GB) ..."
    _download \
        "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt" \
        "$DDPM_PT"
    echo "[✓] Saved to $DDPM_PT"
fi

echo ""
echo "All required weights are ready."
echo "  DiffPure / WaveDM / Ours weights: $DDPM_PT"
