"""
Batch purification using SDEdit + Stable Diffusion 2.1 (Ours).

Usage example:
    python defense/ours/purify_ours.py \
        --adv_dir  /path/to/adv_images \
        --output_dir /path/to/purified_output \
        --model_path defense/models/sd2.1 \
        --strength 0.40 \
        --num_inference_steps 50 \
        --batch_size 4

All images in ``adv_dir`` that end with a common image extension (.png,
.jpg, .jpeg, .bmp, .webp) are processed in batches.  The purified
images are saved to ``output_dir`` with the *same filenames*.
"""

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Resolve imports when the script is executed from the repo root or from
# inside defense/ours/
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sdedit_sd2 import SDEditPurifier

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}


def get_image_paths(directory: str):
    paths = sorted(
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch adversarial purification with SDEdit + SD2.1 (Ours)"
    )
    parser.add_argument("--adv_dir", required=True, type=str,
                        help="Directory containing adversarial images")
    parser.add_argument("--output_dir", required=True, type=str,
                        help="Directory to save purified images")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sd2.1"),
        help="Local path to SD2.1 weights (or HF repo ID)",
    )
    parser.add_argument("--strength", type=float, default=0.40,
                        help="SDEdit noise strength [0,1].  Recommended: 0.35-0.45")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps (50=high quality, 20=faster)")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="CFG guidance scale (1.0 = unconditional)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: 'cuda', 'cuda:0', 'cpu', etc.")
    return parser.parse_args()


def main():
    args = parse_args()

    image_paths = get_image_paths(args.adv_dir)
    if not image_paths:
        print(f"[ERROR] No images found in '{args.adv_dir}'")
        sys.exit(1)

    print(f"[INFO] Found {len(image_paths)} images in '{args.adv_dir}'")
    os.makedirs(args.output_dir, exist_ok=True)

    purifier = SDEditPurifier(
        model_path=args.model_path,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        seed=args.seed,
    )

    for idx, img_path in enumerate(image_paths):
        print(f"[{idx + 1}/{len(image_paths)}] Purifying {img_path.name} ...")
        out_path = os.path.join(args.output_dir, img_path.name)
        if os.path.exists(out_path):
            print(f"  -> already exists, skipping.")
            continue
        purifier.purify_file(str(img_path), out_path)
        print(f"  -> saved to {out_path}")

    print(f"[INFO] Done.  Purified images saved to '{args.output_dir}'")


if __name__ == "__main__":
    main()
