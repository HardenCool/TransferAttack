#!/usr/bin/env python3
"""
Adversarial Purification Comparison Visualizer
================================================
Generates side-by-side comparison figures for five columns:

    Original | MOMUDIG Adv | DiffPure | WaveDM | Ours (DiT)

Usage
-----
Basic (wavelet + DiT only, no DiffPure)::

    python defense/compare_purification.py \\
        --orig_dir path/to/clean_images \\
        --adv_dir  path/to/adv_images \\
        --output_dir comparison_results \\
        --t 150 --num_images 8

With DiffPure (requires ``256x256_diffusion_uncond.pt``)::

    python defense/compare_purification.py \\
        --orig_dir path/to/clean_images \\
        --adv_dir  path/to/adv_images \\
        --output_dir comparison_results \\
        --model_dir path/to/models \\
        --t 150 --num_images 8 --use_diffpure

With WaveDM diffusion refinement::

    python defense/compare_purification.py \\
        --orig_dir path/to/clean_images \\
        --adv_dir  path/to/adv_images \\
        --output_dir comparison_results \\
        --model_dir path/to/models \\
        --t 150 --num_images 8 --wavedm_use_diffusion

With local DiT weights::

    python defense/compare_purification.py \\
        --orig_dir path/to/clean_images \\
        --adv_dir  path/to/adv_images \\
        --output_dir comparison_results \\
        --dit_model_path /path/to/DiT-XL-2-256 \\
        --t 150 --num_images 8

Image Folder Format
-------------------
Both ``--orig_dir`` and ``--adv_dir`` should contain image files with
**matching filenames** (PNG / JPG / JPEG / BMP / TIFF).  No CSV label
file is required; this script is for visual comparison only.

Output
------
* ``<output_dir>/individual/<name>_comparison.png``  — one strip per image
* ``<output_dir>/summary_grid.png``                 — all images in a grid
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy matplotlib import (non-interactive backend)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS = ["Original", "MOMUDIG Adv", "DiffPure", "WaveDM", "Ours (DiT)"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ===========================================================================
# Image I/O helpers
# ===========================================================================

def load_image(path: str, size: int = 256) -> torch.Tensor:
    """Load an image as a float32 tensor in [0, 1] with shape [1, C, H, W]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a [1, C, H, W] or [C, H, W] float32 tensor in [0,1] to PIL."""
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = (t.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))  # [H, W, C]


def collect_image_pairs(orig_dir: str, adv_dir: str):
    """Return a sorted list of (orig_path, adv_path, stem) tuples.

    Matching is done by filename stem regardless of extension.
    """
    orig_files = {
        Path(f).stem: f
        for f in os.listdir(orig_dir)
        if Path(f).suffix.lower() in IMG_EXTS
    }
    adv_files = {
        Path(f).stem: f
        for f in os.listdir(adv_dir)
        if Path(f).suffix.lower() in IMG_EXTS
    }
    common = sorted(orig_files.keys() & adv_files.keys())
    if not common:
        raise RuntimeError(
            f"No matching image pairs found.\n"
            f"  orig_dir: {orig_dir}  ({len(orig_files)} images)\n"
            f"  adv_dir:  {adv_dir}  ({len(adv_files)} images)\n"
            "Ensure both directories contain images with matching filenames."
        )
    return [
        (
            os.path.join(orig_dir, orig_files[s]),
            os.path.join(adv_dir, adv_files[s]),
            s,
        )
        for s in common
    ]


# ===========================================================================
# DiffPure wrapper
# ===========================================================================

class DiffPurePurifier:
    """Minimal standalone wrapper around the Guided Diffusion DDPM runner."""

    def __init__(self, model_dir: str, t: int = 150, device=None):
        self.t = t
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Make the guided_diffusion package importable from diffpure/
        diffpure_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "diffpure")
        )
        if diffpure_dir not in sys.path:
            sys.path.insert(0, diffpure_dir)

        from guided_diffusion.script_util import (
            create_model_and_diffusion,
            model_and_diffusion_defaults,
        )

        cfg = model_and_diffusion_defaults()
        cfg.update(
            {
                "attention_resolutions": "32,16,8",
                "class_cond": False,
                "diffusion_steps": 1000,
                "image_size": 256,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 256,
                "num_head_channels": 64,
                "num_res_blocks": 2,
                "resblock_updown": True,
                "use_fp16": True,
                "use_scale_shift_norm": True,
                "rescale_timesteps": True,
                "timestep_respacing": "1000",
            }
        )

        weight_path = os.path.join(model_dir, "256x256_diffusion_uncond.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Guided Diffusion checkpoint not found:\n  {weight_path}\n"
                "Download from:\n"
                "  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
                "256x256_diffusion_uncond.pt"
            )

        model, diffusion = create_model_and_diffusion(**cfg)
        model.load_state_dict(
            torch.load(weight_path, map_location="cpu", weights_only=True)
        )
        model.requires_grad_(False).eval()
        if cfg["use_fp16"]:
            model.convert_to_fp16()
        model.to(self.device)

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)
        print("DiffPure: Guided Diffusion model loaded.")

    def purify(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C, H, W] in [0, 1]; returns purified in [0, 1] on CPU."""
        orig_size = x.shape[-2:]
        if orig_size != (256, 256):
            x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)

        x_scaled = (x * 2.0 - 1.0).to(self.device)  # [-1, 1]
        B = x_scaled.shape[0]
        t = self.t

        a = (1 - self.betas).cumprod(dim=0)
        noise = torch.randn_like(x_scaled)
        x_noisy = x_scaled * a[t - 1].sqrt() + noise * (1.0 - a[t - 1]).sqrt()

        with torch.no_grad():
            for i in reversed(range(t)):
                t_batch = torch.tensor([i] * B, device=self.device)
                x_noisy = self.diffusion.p_sample(
                    self.model,
                    x_noisy,
                    t_batch,
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                )["sample"]

        x_out = (x_noisy + 1.0) * 0.5
        if orig_size != (256, 256):
            x_out = F.interpolate(
                x_out, size=orig_size, mode="bilinear", align_corners=False
            )
        return x_out.clamp(0, 1).cpu()

    def __call__(self, x):
        return self.purify(x)


# ===========================================================================
# Visualization helpers
# ===========================================================================

def make_strip(images: list, labels: list, title: str = None) -> plt.Figure:
    """Create a 1×N figure strip for a single input image.

    Parameters
    ----------
    images : list of PIL.Image
        One PIL image per column.
    labels : list of str
        Column header for each image.
    title : str or None
        Optional super-title.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, lbl in zip(axes, images, labels):
        ax.imshow(np.array(img))
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    return fig


def make_summary_grid(
    rows: list,          # list of (stem, [PIL images × 5])
    method_labels: list,
) -> plt.Figure:
    """Create an R×5 grid figure.

    Parameters
    ----------
    rows : list of (stem, images)
        Each entry is a (filename_stem, list-of-5-PIL-images) tuple.
    method_labels : list of str
        Column header for each of the 5 methods.
    """
    n_rows = len(rows)
    n_cols = len(method_labels)
    fig = plt.figure(figsize=(3.5 * n_cols, 3.5 * n_rows + 0.6))
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.08, wspace=0.04,
        top=0.94, bottom=0.01, left=0.01, right=0.99,
    )

    for col, lbl in enumerate(method_labels):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(lbl, fontsize=11, fontweight="bold", pad=4)
        ax.axis("off")

    for row_idx, (stem, imgs) in enumerate(rows):
        for col_idx, img in enumerate(imgs):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.imshow(np.array(img))
            if col_idx == 0:
                ax.set_ylabel(
                    stem, fontsize=7, rotation=0,
                    labelpad=4, va="center", ha="right",
                )
            ax.axis("off")

    return fig


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate purification comparison visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- I/O -----------------------------------------------------------------
    p.add_argument(
        "--orig_dir", required=True,
        help="Directory containing clean/original images.",
    )
    p.add_argument(
        "--adv_dir", required=True,
        help="Directory containing adversarial (MOMUDIG) images. "
             "Filenames must match those in --orig_dir.",
    )
    p.add_argument(
        "--output_dir", default="comparison_results",
        help="Directory where output figures are saved.",
    )
    p.add_argument(
        "--num_images", type=int, default=5,
        help="Maximum number of image pairs to process.",
    )
    p.add_argument(
        "--img_size", type=int, default=256,
        help="Resize all images to this square size before processing.",
    )

    # -- Diffusion noise scale -----------------------------------------------
    p.add_argument(
        "--t", type=int, default=150,
        help="SDEdit noise steps (forward diffusion strength) for DiffPure "
             "and DiT.  Range [50, 300].  Lower = more faithful, "
             "Higher = stronger purification.",
    )

    # -- DiffPure ------------------------------------------------------------
    p.add_argument(
        "--use_diffpure", action="store_true",
        help="Enable DiffPure column (requires --model_dir with "
             "256x256_diffusion_uncond.pt).",
    )
    p.add_argument(
        "--model_dir", default="defense/models",
        help="Directory containing 256x256_diffusion_uncond.pt "
             "(used by DiffPure and optional WaveDM refinement).",
    )

    # -- WaveDM --------------------------------------------------------------
    p.add_argument(
        "--wavedm_wavelet", default="db4",
        help="PyWavelets wavelet family for WaveDM (e.g. haar, db4, bior2.2).",
    )
    p.add_argument(
        "--wavedm_level", type=int, default=1,
        help="Wavelet decomposition levels for WaveDM.",
    )
    p.add_argument(
        "--wavedm_threshold", type=float, default=0.06,
        help="Soft-threshold applied to high-frequency wavelet coefficients. "
             "~2× adversarial epsilon is a sensible starting value.",
    )
    p.add_argument(
        "--wavedm_use_diffusion", action="store_true",
        help="Follow wavelet denoising with a light DDPM refinement pass "
             "(requires --model_dir with 256x256_diffusion_uncond.pt).",
    )
    p.add_argument(
        "--wavedm_diffusion_t", type=int, default=50,
        help="DDPM steps for the optional WaveDM refinement pass.",
    )

    # -- DiT -----------------------------------------------------------------
    p.add_argument(
        "--dit_model_id", default="facebook/DiT-XL-2-256",
        help="HuggingFace model ID for the DiT purifier.",
    )
    p.add_argument(
        "--dit_model_path", default=None,
        help="Local path to a saved DiT model (diffusers snapshot layout). "
             "Overrides --dit_model_id when set.",
    )
    p.add_argument(
        "--dit_class_label", type=int, default=0,
        help="ImageNet class label used to condition the DiT denoiser "
             "(0-999).  Use 1000 for the null/unconditional token.",
    )
    p.add_argument(
        "--dit_use_ddim", action="store_true", default=True,
        help="Use DDIM scheduler for faster DiT denoising (default True).",
    )
    p.add_argument(
        "--dit_ddim_steps", type=int, default=50,
        help="Number of DDIM steps for DiT denoising.",
    )

    # -- Device --------------------------------------------------------------
    p.add_argument(
        "--device", default=None,
        help="PyTorch device string (e.g. 'cuda', 'cuda:1', 'cpu'). "
             "Auto-detected when omitted.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
    )
    print(f"Using device: {device}")

    # -----------------------------------------------------------------------
    # Collect image pairs
    # -----------------------------------------------------------------------
    pairs = collect_image_pairs(args.orig_dir, args.adv_dir)
    pairs = pairs[: args.num_images]
    print(f"Processing {len(pairs)} image pair(s).")

    # -----------------------------------------------------------------------
    # Initialise purifiers
    # -----------------------------------------------------------------------
    # -- WaveDM --------------------------------------------------------------
    try:
        from wavedm.purify import WaveDMPurifier
    except ImportError:
        # Allow running from repo root
        sys.path.insert(0, os.path.dirname(__file__))
        from wavedm.purify import WaveDMPurifier

    wavedm = WaveDMPurifier(
        wavelet=args.wavedm_wavelet,
        level=args.wavedm_level,
        threshold=args.wavedm_threshold,
        use_diffusion=args.wavedm_use_diffusion,
        diffusion_t=args.wavedm_diffusion_t,
        model_dir=args.model_dir,
        device=device,
    )

    # -- DiffPure ------------------------------------------------------------
    diffpure = None
    if args.use_diffpure:
        diffpure = DiffPurePurifier(
            model_dir=args.model_dir,
            t=args.t,
            device=device,
        )

    # -- DiT -----------------------------------------------------------------
    try:
        from dit_purify.purify import DiTPurifier
    except ImportError:
        sys.path.insert(0, os.path.dirname(__file__))
        from dit_purify.purify import DiTPurifier

    dit = DiTPurifier(
        model_id=args.dit_model_id,
        model_path=args.dit_model_path,
        t=args.t,
        class_label=args.dit_class_label,
        use_ddim=args.dit_use_ddim,
        ddim_steps=args.dit_ddim_steps,
        device=device,
    )

    # -----------------------------------------------------------------------
    # Output directories
    # -----------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    individual_dir = os.path.join(args.output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Process each pair
    # -----------------------------------------------------------------------
    summary_rows = []

    for orig_path, adv_path, stem in pairs:
        print(f"\n[{stem}]")

        # Load images
        orig_t = load_image(orig_path, size=args.img_size)   # [1,3,H,W] ∈[0,1]
        adv_t  = load_image(adv_path,  size=args.img_size)

        # Purify: WaveDM
        print("  WaveDM …")
        with torch.no_grad():
            wavedm_t = wavedm(adv_t)

        # Purify: DiffPure
        if diffpure is not None:
            print("  DiffPure …")
            with torch.no_grad():
                diffpure_t = diffpure(adv_t)
        else:
            # Show a grey placeholder with a "N/A" label
            diffpure_t = torch.full_like(orig_t, 0.5)

        # Purify: DiT
        print("  DiT (Ours) …")
        with torch.no_grad():
            dit_t = dit(adv_t)

        # Convert to PIL
        pil_orig     = tensor_to_pil(orig_t)
        pil_adv      = tensor_to_pil(adv_t)
        pil_diffpure = tensor_to_pil(diffpure_t)
        pil_wavedm   = tensor_to_pil(wavedm_t)
        pil_dit      = tensor_to_pil(dit_t)

        method_images = [pil_orig, pil_adv, pil_diffpure, pil_wavedm, pil_dit]
        method_labels = METHODS[:]
        if diffpure is None:
            method_labels[2] = "DiffPure\n(N/A)"

        # Individual strip
        fig = make_strip(method_images, method_labels, title=stem)
        strip_path = os.path.join(individual_dir, f"{stem}_comparison.png")
        fig.savefig(strip_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Saved: {strip_path}")

        summary_rows.append((stem, method_images))

    # -----------------------------------------------------------------------
    # Summary grid
    # -----------------------------------------------------------------------
    print("\nGenerating summary grid …")
    method_labels = METHODS[:]
    if diffpure is None:
        method_labels[2] = "DiffPure\n(N/A: use --use_diffpure)"
    grid_fig = make_summary_grid(summary_rows, method_labels)
    grid_path = os.path.join(args.output_dir, "summary_grid.png")
    grid_fig.savefig(grid_path, bbox_inches="tight", dpi=150)
    plt.close(grid_fig)
    print(f"Summary grid saved: {grid_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
