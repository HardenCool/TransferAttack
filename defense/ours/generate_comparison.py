"""
Generate side-by-side comparison figures.

Layout per row (one row = one image):
  [Original] | [MOMUDIG Adv] | [DiffPure] | [WaveDM] | [Ours (SD2.1)]

Usage – pre-purified Ours results:
    python defense/ours/generate_comparison.py \
        --orig_dir      /path/to/original_images \
        --adv_dir       /path/to/momudig_adv_images \
        --diffpure_dir  /path/to/diffpure_output \
        --wavedm_dir    /path/to/wavedm_output \
        --ours_dir      /path/to/ours_purified_output \
        --output_dir    /path/to/comparison_output

Usage – on-the-fly Ours purification (no pre-computed ours_dir needed):
    python defense/ours/generate_comparison.py \
        --orig_dir      /path/to/original_images \
        --adv_dir       /path/to/momudig_adv_images \
        --diffpure_dir  /path/to/diffpure_output \
        --wavedm_dir    /path/to/wavedm_output \
        --output_dir    /path/to/comparison_output \
        --model_path    defense/models/sd2.1 \
        --strength      0.40

Two outputs are written to ``output_dir``:
  * ``individual/``  – one PNG per image (5 panels side-by-side)
  * ``grid.png``     – all images stacked vertically in a single file

Column headers can be customised with --col_labels.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}

DEFAULT_LABELS = ["Original", "MOMUDIG Adv", "DiffPure", "WaveDM", "Ours (SD2.1)"]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _get_sorted_names(directory: str) -> List[str]:
    return sorted(
        p.name for p in Path(directory).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _load_rgb(path: str, size: Optional[tuple] = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size is not None and img.size != size:
        img = img.resize(size, Image.LANCZOS)
    return img


def _make_header(width: int, height: int, text: str, font) -> Image.Image:
    header = Image.new("RGB", (width, height), color=(40, 40, 40))
    draw = ImageDraw.Draw(header)
    try:
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
    tx = (width - tw) // 2
    ty = (height - th) // 2
    draw.text((tx, ty), text, fill=(255, 255, 255), font=font)
    return header


def _try_load_font(size: int = 20):
    """Try to load a truetype font; fall back to default PIL font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    from PIL import ImageFont
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_row(
    images: List[Image.Image],
    labels: List[str],
    thumb_size: tuple,
    header_height: int,
    font,
    border: int = 4,
    border_color=(80, 80, 80),
) -> Image.Image:
    """Combine a list of images into a single horizontal strip with headers."""
    n = len(images)
    W, H = thumb_size
    total_w = n * (W + border) + border
    total_h = header_height + H + border * 2

    strip = Image.new("RGB", (total_w, total_h), color=(20, 20, 20))

    for i, (img, label) in enumerate(zip(images, labels)):
        x = border + i * (W + border)
        header = _make_header(W, header_height, label, font)
        strip.paste(header, (x, border))
        strip.paste(img.resize(thumb_size, Image.LANCZOS), (x, border + header_height))

    return strip


# ----------------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------------

def build_comparison(
    orig_dir: str,
    adv_dir: str,
    diffpure_dir: str,
    wavedm_dir: str,
    ours_dir: Optional[str],
    output_dir: str,
    col_labels: List[str],
    thumb_size: tuple,
    header_height: int,
    font_size: int,
    model_path: Optional[str],
    strength: float,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    device: Optional[str],
):
    # Determine which filenames exist in ALL required directories
    orig_names = set(_get_sorted_names(orig_dir))
    adv_names = set(_get_sorted_names(adv_dir))
    diffpure_names = set(_get_sorted_names(diffpure_dir))
    wavedm_names = set(_get_sorted_names(wavedm_dir))

    common = orig_names & adv_names & diffpure_names & wavedm_names

    if ours_dir is not None:
        ours_names = set(_get_sorted_names(ours_dir))
        common &= ours_names
    else:
        # We will run purification on-the-fly from adv_dir
        if model_path is None:
            print("[ERROR] Provide either --ours_dir or --model_path to run Ours purification.")
            sys.exit(1)

    if not common:
        print("[ERROR] No common image filenames found across all input directories.")
        sys.exit(1)

    filenames = sorted(common)
    print(f"[INFO] Processing {len(filenames)} image(s).")

    # Load purifier lazily (only if needed)
    purifier = None
    if ours_dir is None:
        from sdedit_sd2 import SDEditPurifier
        purifier = SDEditPurifier(
            model_path=model_path,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            device=device,
            seed=seed,
        )

    font = _try_load_font(font_size)
    individual_dir = os.path.join(output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    rows = []
    for idx, name in enumerate(filenames):
        print(f"[{idx + 1}/{len(filenames)}] {name}")

        orig_img = _load_rgb(os.path.join(orig_dir, name), thumb_size)
        adv_img = _load_rgb(os.path.join(adv_dir, name), thumb_size)
        diffpure_img = _load_rgb(os.path.join(diffpure_dir, name), thumb_size)
        wavedm_img = _load_rgb(os.path.join(wavedm_dir, name), thumb_size)

        if ours_dir is not None:
            ours_img = _load_rgb(os.path.join(ours_dir, name), thumb_size)
        else:
            print(f"  -> running Ours purification ...")
            adv_full = Image.open(os.path.join(adv_dir, name)).convert("RGB")
            ours_img = purifier.purify_pil(adv_full).resize(thumb_size, Image.LANCZOS)

        panel_images = [orig_img, adv_img, diffpure_img, wavedm_img, ours_img]
        row = make_row(panel_images, col_labels, thumb_size, header_height, font)

        stem = Path(name).stem
        row_path = os.path.join(individual_dir, f"{stem}_comparison.png")
        row.save(row_path)
        rows.append(row)

    # Stack all rows into a single grid image
    if rows:
        grid_w = rows[0].width
        grid_h = sum(r.height for r in rows)
        grid = Image.new("RGB", (grid_w, grid_h))
        y = 0
        for r in rows:
            grid.paste(r, (0, y))
            y += r.height
        grid_path = os.path.join(output_dir, "grid.png")
        grid.save(grid_path)
        print(f"[INFO] Grid saved to '{grid_path}'")

    print(f"[INFO] Individual comparisons saved to '{individual_dir}'")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 5-column comparison figures (Original | MOMUDIG | DiffPure | WaveDM | Ours)"
    )
    parser.add_argument("--orig_dir", required=True, help="Directory of original clean images")
    parser.add_argument("--adv_dir", required=True, help="Directory of MOMUDIG adversarial images")
    parser.add_argument("--diffpure_dir", required=True, help="Directory of DiffPure purified images")
    parser.add_argument("--wavedm_dir", required=True, help="Directory of WaveDM purified images")
    parser.add_argument(
        "--ours_dir", default=None,
        help="Directory of pre-computed Ours purified images.  "
             "If omitted, purification is run on-the-fly (requires --model_path).",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save comparison images")

    # On-the-fly purification options (used when --ours_dir is not provided)
    parser.add_argument(
        "--model_path",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sd2.1"),
        help="Local path to SD2.1 weights (or HF repo ID)",
    )
    parser.add_argument("--strength", type=float, default=0.40,
                        help="SDEdit noise strength [0,1]")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    # Visual options
    parser.add_argument(
        "--col_labels", nargs=5,
        default=DEFAULT_LABELS,
        metavar=("COL1", "COL2", "COL3", "COL4", "COL5"),
        help="Column header labels (5 values)",
    )
    parser.add_argument("--thumb_size", type=int, default=224,
                        help="Thumbnail size (square) for each panel in pixels")
    parser.add_argument("--header_height", type=int, default=30,
                        help="Height in pixels of the column header bar")
    parser.add_argument("--font_size", type=int, default=16,
                        help="Font size for column labels")
    return parser.parse_args()


def main():
    args = parse_args()
    build_comparison(
        orig_dir=args.orig_dir,
        adv_dir=args.adv_dir,
        diffpure_dir=args.diffpure_dir,
        wavedm_dir=args.wavedm_dir,
        ours_dir=args.ours_dir,
        output_dir=args.output_dir,
        col_labels=args.col_labels,
        thumb_size=(args.thumb_size, args.thumb_size),
        header_height=args.header_height,
        font_size=args.font_size,
        model_path=args.model_path,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
