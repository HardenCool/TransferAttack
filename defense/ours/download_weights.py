"""
Download Stable Diffusion 2.1-base weights from HuggingFace Hub.

Usage:
    python defense/ours/download_weights.py [--model_dir defense/models/sd2.1]

The downloaded weights will be saved to ``model_dir`` and can be used
directly by ``sdedit_sd2.py`` and ``purify_ours.py``.

Requirements:
    pip install diffusers>=0.21 transformers accelerate huggingface_hub
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Download SD2.1 weights for Ours purification")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sd2.1"),
        help="Local directory to save the model weights",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="HuggingFace model repo ID",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace access token (required for gated models)",
    )
    return parser.parse_args()


def check_dependencies():
    missing = []
    for pkg in ("diffusers", "transformers", "accelerate", "huggingface_hub"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"       Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def download(repo_id: str, model_dir: str, token: str | None = None):
    from huggingface_hub import snapshot_download

    os.makedirs(model_dir, exist_ok=True)
    print(f"[INFO] Downloading '{repo_id}' -> '{model_dir}' ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=model_dir,
        token=token,
        ignore_patterns=["*.bin.index.json", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"[INFO] Download complete.  Weights saved to: {model_dir}")


def main():
    args = parse_args()
    check_dependencies()
    download(args.repo_id, args.model_dir, args.token)


if __name__ == "__main__":
    main()
