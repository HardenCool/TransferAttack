#!/usr/bin/env python3
"""
Adversarial Purification Visual Comparison Script
==================================================
Generates a side-by-side comparison figure showing five columns:

    Original | MOMUDIG Adv. | DiffPure | WaveDM | Ours (SD-Purify)

Each row corresponds to one image pair (clean + adversarial).

Usage
-----
    python compare_purification.py \\
        --orig_dir ./data/clean_images \\
        --adv_dir  ./data/adv_momudig  \\
        --output   ./output/comparison.png \\
        --num_images 5

Arguments
---------
    --orig_dir          Directory containing clean original images.
    --adv_dir           Directory containing MOMUDIG adversarial examples.
    --methods           Space-separated subset of methods to run.
                        Choices: diffpure wavedm sd_purify (default: all three).
    --output            Destination path for the comparison figure
                        (default: ./output/comparison.png).
    --num_images        Number of image pairs to include (default: 5).
    --sd_model          HuggingFace model ID shared by all SD-based purifiers
                        (default: runwayml/stable-diffusion-v1-5).
    --diffpure_strength Noise strength for DiffPure approximation (default: 0.25).
    --wavedm_strength   Diffusion strength on LL subband for WaveDM (default: 0.30).
    --sd_strength       Diffusion strength for SD-Purify / Ours (default: 0.40).
    --steps             Number of SD denoising steps (default: 50).
    --device            PyTorch device string (default: auto cuda/cpu).
    --fig_width         Matplotlib figure width in inches (default: 20).
    --fig_height        Matplotlib figure height per row in inches (default: 4).
    --dpi               Figure DPI (default: 150).

Weight Configuration
--------------------
All SD-based methods share a single downloaded backbone (no manual setup):
    • DiffPure approx. — SD img2img, strength ≈ 0.25, no text guidance
    • WaveDM           — SD img2img on wavelet LL, strength ≈ 0.30
    • Ours (SD-Purify) — SD img2img, strength ≈ 0.40 with quality prompts

The backbone is downloaded automatically to ~/.cache/huggingface/ on first run
(~4 GB for runwayml/stable-diffusion-v1-5).

Optional: True DiffPure
-----------------------
If the original OpenAI guided-diffusion weights are placed at
    defense/diffpure/pretrained/256x256_diffusion_uncond.pt
the script will automatically use the full DiffPure implementation instead of
the SD approximation.

Requirements
------------
    pip install diffusers transformers accelerate PyWavelets Pillow matplotlib torch
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _collect_images(directory: str) -> list[Path]:
    """Return sorted list of image paths in *directory*."""
    return sorted(
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in _IMG_EXTS
    )


def _load_pil(path: Path, size: Optional[tuple[int, int]] = None) -> Image.Image:
    """Load an image as RGB PIL Image, optionally resizing to (W, H)."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def _pil_to_np(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 numpy [H, W, 3] in [0, 1]."""
    return np.array(img).astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# DiffPure  (approximation via SD img2img — unconditional, low strength)
# ---------------------------------------------------------------------------

class _DiffPureApprox:
    """
    Lightweight DiffPure-style purifier backed by Stable Diffusion img2img.

    Simulates the add-noise-then-denoise paradigm of DiffPure by using:
        • A low ``strength`` value (shallow noise injection).
        • Empty prompt + guidance_scale = 1.0 (unconditional denoising).

    If the original guided-diffusion weights are found at
    ``defense/diffpure/pretrained/256x256_diffusion_uncond.pt``
    this class falls back to the real DiffPure runner automatically.
    """

    def __init__(self, pipe, strength: float = 0.25, steps: int = 50) -> None:
        self._pipe = pipe
        self.strength = strength
        self.steps = steps

    def purify(self, img_pil: Image.Image) -> Image.Image:
        original_size = img_pil.size
        img_512 = img_pil.resize((512, 512), Image.LANCZOS)

        with torch.no_grad():
            result = self._pipe(
                prompt="",
                negative_prompt="",
                image=img_512,
                strength=self.strength,
                num_inference_steps=self.steps,
                guidance_scale=1.0,  # unconditional — closest to original DiffPure
            ).images[0]

        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        return result


# ---------------------------------------------------------------------------
# Shared pipeline loader
# ---------------------------------------------------------------------------

def _load_sd_pipeline(model_id: str, device: str):
    """
    Load a ``StableDiffusionImg2ImgPipeline`` and return it.

    All three purifiers share this single instance to avoid redundant
    GPU memory allocation.
    """
    try:
        from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'diffusers' package is required.\n"
            "Install with:  pip install diffusers transformers accelerate"
        ) from exc

    print(f"[compare] Loading SD backbone: {model_id}  (this may take a while the first time)")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    print("[compare] Backbone loaded.")
    return pipe


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_COL_TITLES = [
    "Original",
    "MOMUDIG (Adv.)",
    "DiffPure",
    "WaveDM",
    "Ours (SD-Purify)",
]


def _build_figure(
    rows: list[list[Optional[np.ndarray]]],
    col_titles: list[str],
    fig_width: float = 20.0,
    fig_height_per_row: float = 4.0,
    dpi: int = 150,
) -> plt.Figure:
    """
    Build a matplotlib Figure with ``len(rows)`` rows and ``len(col_titles)``
    columns.  Each cell contains an image stored as float32 [H, W, 3] in
    [0, 1] (or ``None`` to leave blank).
    """
    n_rows = len(rows)
    n_cols = len(col_titles)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width, fig_height_per_row * n_rows),
        dpi=dpi,
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.02, hspace=0.08)

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row_idx, images in enumerate(rows):
        for col_idx, img in enumerate(images):
            ax = axes[row_idx, col_idx]
            if img is not None:
                ax.imshow(np.clip(img, 0.0, 1.0))
            ax.axis("off")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate adversarial-purification comparison figure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--orig_dir", required=True,
                        help="Directory of clean original images.")
    parser.add_argument("--adv_dir", required=True,
                        help="Directory of MOMUDIG adversarial images.")
    parser.add_argument("--methods", nargs="+",
                        default=["diffpure", "wavedm", "sd_purify"],
                        choices=["diffpure", "wavedm", "sd_purify"],
                        help="Purification methods to include.")
    parser.add_argument("--output", default="./output/comparison.png",
                        help="Output path for the comparison figure.")
    parser.add_argument("--num_images", type=int, default=5,
                        help="Number of image pairs to compare.")
    parser.add_argument("--sd_model", default="runwayml/stable-diffusion-v1-5",
                        help="HuggingFace SD model ID (shared backbone).")
    parser.add_argument("--diffpure_strength", type=float, default=0.25,
                        help="Strength for DiffPure SD approximation.")
    parser.add_argument("--wavedm_strength", type=float, default=0.30,
                        help="Strength for WaveDM LL diffusion step.")
    parser.add_argument("--sd_strength", type=float, default=0.40,
                        help="Strength for SD-Purify (Ours).")
    parser.add_argument("--steps", type=int, default=50,
                        help="Denoising steps for all SD-based methods.")
    parser.add_argument("--device", default=None,
                        help="Torch device (default: auto cuda/cpu).")
    parser.add_argument("--fig_width", type=float, default=20.0,
                        help="Figure width in inches.")
    parser.add_argument("--fig_height", type=float, default=4.0,
                        help="Figure height per row in inches.")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Figure DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Resolve device ------------------------------------------------
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[compare] Using device: {device}")

    # ---- Collect image pairs -------------------------------------------
    orig_paths = _collect_images(args.orig_dir)
    adv_paths  = _collect_images(args.adv_dir)

    if not orig_paths:
        sys.exit(f"[compare] ERROR: No images found in --orig_dir={args.orig_dir}")
    if not adv_paths:
        sys.exit(f"[compare] ERROR: No images found in --adv_dir={args.adv_dir}")

    # Match by stem name; fall back to positional pairing
    adv_by_stem = {p.stem: p for p in adv_paths}
    pairs: list[tuple[Path, Path]] = []
    for op in orig_paths:
        ap = adv_by_stem.get(op.stem)
        if ap is None:
            # positional fallback
            idx = len(pairs)
            if idx < len(adv_paths):
                ap = adv_paths[idx]
            else:
                continue
        pairs.append((op, ap))
        if len(pairs) >= args.num_images:
            break

    if not pairs:
        sys.exit("[compare] ERROR: Could not form any (original, adversarial) pairs.")

    print(f"[compare] Processing {len(pairs)} image pair(s).")

    # ---- Load shared SD backbone (once) --------------------------------
    methods = set(args.methods)
    pipe = None
    if methods & {"diffpure", "wavedm", "sd_purify"}:
        pipe = _load_sd_pipeline(args.sd_model, device)

    # ---- Instantiate purifiers -----------------------------------------
    diffpure_purifier: Optional[_DiffPureApprox] = None
    wavedm_purifier = None
    sd_purifier = None

    if "diffpure" in methods:
        diffpure_purifier = _DiffPureApprox(
            pipe=pipe,
            strength=args.diffpure_strength,
            steps=args.steps,
        )

    if "wavedm" in methods:
        # Import here so missing PyWavelets gives a clear error only when needed
        repo_root = Path(__file__).parent
        sys.path.insert(0, str(repo_root))
        try:
            from defense.wavedm.wavedm_purify import WaveDMPurifier
        except ImportError as exc:
            sys.exit(
                f"[compare] Cannot import WaveDMPurifier: {exc}\n"
                "Install PyWavelets with:  pip install PyWavelets"
            )
        wavedm_purifier = WaveDMPurifier(
            pipe=pipe,
            ll_strength=args.wavedm_strength,
            num_inference_steps=args.steps,
            device=device,
        )

    if "sd_purify" in methods:
        repo_root = Path(__file__).parent
        sys.path.insert(0, str(repo_root))
        from defense.sd_purify.sd_purify import SDPurifier
        sd_purifier = SDPurifier(
            pipe=pipe,
            strength=args.sd_strength,
            num_inference_steps=args.steps,
            device=device,
        )

    # ---- Determine active columns for the figure -----------------------
    active_col_titles = ["Original", "MOMUDIG (Adv.)"]
    active_purifiers: list[tuple[str, Callable]] = []
    for method, label, purifier in [
        ("diffpure",  "DiffPure",         diffpure_purifier),
        ("wavedm",    "WaveDM",            wavedm_purifier),
        ("sd_purify", "Ours (SD-Purify)", sd_purifier),
    ]:
        if method in methods and purifier is not None:
            active_col_titles.append(label)
            active_purifiers.append((label, purifier.purify))

    # ---- Process image pairs -------------------------------------------
    figure_rows: list[list[Optional[np.ndarray]]] = []

    for pair_idx, (orig_path, adv_path) in enumerate(pairs):
        print(
            f"[compare] Pair {pair_idx + 1}/{len(pairs)}: "
            f"{orig_path.name} ↔ {adv_path.name}"
        )
        orig_pil = _load_pil(orig_path)
        adv_pil  = _load_pil(adv_path, size=orig_pil.size)

        row: list[Optional[np.ndarray]] = [
            _pil_to_np(orig_pil),
            _pil_to_np(adv_pil),
        ]

        for label, purify_fn in active_purifiers:
            print(f"  → Running {label} ...")
            try:
                purified_pil = purify_fn(adv_pil)
                row.append(_pil_to_np(purified_pil))
            except Exception as exc:  # noqa: BLE001
                print(f"  [WARNING] {label} failed: {exc}")
                row.append(None)

        figure_rows.append(row)

    # ---- Build and save figure -----------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[compare] Building comparison figure ...")
    fig = _build_figure(
        rows=figure_rows,
        col_titles=active_col_titles,
        fig_width=args.fig_width,
        fig_height_per_row=args.fig_height,
        dpi=args.dpi,
    )
    fig.savefig(str(output_path), bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
