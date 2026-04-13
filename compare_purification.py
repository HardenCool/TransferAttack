#!/usr/bin/env python3
"""
compare_purification.py
=======================
Generate a side-by-side comparison figure showing:

  Column 1 — Original image
  Column 2 — MOMUDIG adversarial example (your attack)
  Column 3 — DiffPure purified
  Column 4 — WaveDM purified
  Column 5 — Ours (FreqDDPM) purified

For each purified image the script computes SSIM, PSNR (dB), and LPIPS w.r.t.
the original image and displays the metrics below the corresponding panel.

Quick start
-----------
  python compare_purification.py \\
      --original  path/to/clean.png \\
      --adversarial path/to/adv.png \\
      --output    comparison.png \\
      --model_dir defense/models

You can skip individual defenses if you don't have the required weights yet:
  --skip_diffpure  --skip_wavedm  --skip_ours

Dependencies (install once):
  pip install PyWavelets lpips scikit-image

Pretrained weights (all methods share the same file):
  bash defense/ours/download_weights.sh
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ── Metrics ──────────────────────────────────────────────────────────────────
try:
    from skimage.metrics import structural_similarity as _ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as _psnr_fn
except ImportError as e:
    raise ImportError("scikit-image is required: pip install scikit-image") from e

try:
    import lpips as _lpips_lib
    _lpips_model = None          # loaded lazily
except ImportError as e:
    raise ImportError(
        "lpips is required: pip install lpips"
    ) from e


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_lpips_model(device):
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = _lpips_lib.LPIPS(net="alex").to(device)
        _lpips_model.eval()
    return _lpips_model


def compute_metrics(original: torch.Tensor, purified: torch.Tensor,
                    device: torch.device) -> dict:
    """
    Compute SSIM, PSNR, and LPIPS between original and purified images.

    Both tensors must be float32, shape (C, H, W), in [0, 1].
    """
    orig_np = original.cpu().numpy().transpose(1, 2, 0)   # HWC
    pur_np  = purified.cpu().numpy().transpose(1, 2, 0)

    ssim  = float(_ssim_fn(orig_np, pur_np, data_range=1.0, channel_axis=2))
    psnr  = float(_psnr_fn(orig_np, pur_np, data_range=1.0))

    lpips_model = _get_lpips_model(device)
    orig_t = original.unsqueeze(0).to(device) * 2.0 - 1.0  # [-1,1]
    pur_t  = purified.unsqueeze(0).to(device) * 2.0 - 1.0
    with torch.no_grad():
        lp = float(lpips_model(orig_t, pur_t).item())

    return {"ssim": ssim, "psnr": psnr, "lpips": lp}


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str, size: int | None = None) -> torch.Tensor:
    """Load an image as a float32 CHW tensor in [0, 1]."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    """CHW [0,1] tensor → HWC uint8 array."""
    return (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# DiffPure inference helper
# ─────────────────────────────────────────────────────────────────────────────

def run_diffpure(adv_image: torch.Tensor, model_dir: str, t: int,
                 device: torch.device) -> torch.Tensor:
    """
    Run DiffPure using the existing guided diffusion backbone.

    Identical to the original DiffPure forward-reverse pass but wrapped as a
    simple function for use in the comparison pipeline.
    """
    _diffpure_dir = os.path.join(os.path.dirname(__file__), "defense", "diffpure")
    for _p in [_diffpure_dir]:
        if _p not in sys.path:
            sys.path.insert(0, _p)

    from guided_diffusion.script_util import (
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    weights_path = os.path.join(model_dir, "256x256_diffusion_uncond.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"DiffPure weights not found at {weights_path}.\n"
            "Run: bash defense/ours/download_weights.sh"
        )

    model_cfg = model_and_diffusion_defaults()
    model_cfg.update(dict(
        attention_resolutions="32,16,8",
        class_cond=False,
        diffusion_steps=1000,
        rescale_timesteps=True,
        timestep_respacing="1000",
        image_size=256,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
    ))
    model, diffusion = create_model_and_diffusion(**model_cfg)
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu"), strict=False
    )
    model.requires_grad_(False).eval().to(device)
    if model_cfg["use_fp16"]:
        model.convert_to_fp16()

    betas = torch.from_numpy(diffusion.betas).float().to(device)
    alphas_cumprod = (1.0 - betas).cumprod(dim=0)

    batched = adv_image.dim() == 4
    if not batched:
        adv_image = adv_image.unsqueeze(0)

    H, W = adv_image.shape[-2], adv_image.shape[-1]
    x0 = adv_image.to(device)
    if H != 256 or W != 256:
        x0 = F.interpolate(x0, size=(256, 256), mode="bilinear", align_corners=False)

    x0_scaled = (x0 - 0.5) * 2.0
    e = torch.randn_like(x0_scaled)
    a = alphas_cumprod[t - 1]
    x = x0_scaled * a.sqrt() + e * (1.0 - a).sqrt()

    with torch.no_grad():
        for i in reversed(range(t)):
            t_b = torch.tensor([i] * x.shape[0], device=device)
            x = diffusion.p_sample(
                model, x, t_b,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
            )["sample"]

    x = (x + 1.0) * 0.5
    if H != 256 or W != 256:
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)

    if not batched:
        x = x.squeeze(0)
    return x.clamp(0, 1).cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Figure rendering
# ─────────────────────────────────────────────────────────────────────────────

_TITLES = [
    "Original",
    "MOMUDIG\n(Adversarial)",
    "DiffPure\n(Purified)",
    "WaveDM\n(Purified)",
    "Ours / FreqDDPM\n(Purified)",
]

# Reference values from Table 4.6 of the paper (used as a sanity-check reference).
# These are the expected metrics for each method when evaluated on the test set.
_TARGET_METRICS = {
    "DiffPure\n(Purified)":         {"ssim": 0.876, "psnr": 28.6, "lpips": 0.112},
    "WaveDM\n(Purified)":           {"ssim": 0.898, "psnr": 31.2, "lpips": 0.085},
    "Ours / FreqDDPM\n(Purified)":  {"ssim": 0.905, "psnr": 30.9, "lpips": 0.071},
}


def save_comparison_figure(
    images: dict,          # title -> HWC uint8 ndarray
    metrics: dict,         # title -> {ssim, psnr, lpips}  (only purified ones)
    output_path: str,
    figsize: tuple = (20, 7),
    dpi: int = 150,
):
    """Render and save the 5-panel comparison figure."""
    n = len(images)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(
        2, n,
        figure=fig,
        height_ratios=[5, 1],
        hspace=0.05,
        wspace=0.03,
    )

    for col, title in enumerate(_TITLES):
        if title not in images:
            continue
        img = images[title]

        # Image panel
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(img)
        ax_img.set_title(title, fontsize=11, fontweight="bold", pad=4)
        ax_img.axis("off")

        # Metrics panel
        ax_met = fig.add_subplot(gs[1, col])
        ax_met.axis("off")
        if title in metrics:
            m = metrics[title]
            txt = (
                f"SSIM:  {m['ssim']:.3f}\n"
                f"PSNR:  {m['psnr']:.1f} dB\n"
                f"LPIPS: {m['lpips']:.3f}"
            )
            color = "#1a7abf" if "Ours" in title else "black"
            ax_met.text(
                0.5, 0.95, txt,
                transform=ax_met.transAxes,
                ha="center", va="top",
                fontsize=9, family="monospace",
                color=color,
            )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[✓] Comparison figure saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate adversarial purification comparison figure."
    )
    p.add_argument("--original",    required=True,
                   help="Path to the clean / original image.")
    p.add_argument("--adversarial", required=True,
                   help="Path to the MOMUDIG adversarial example.")
    p.add_argument("--output",      default="comparison.png",
                   help="Output file path (default: comparison.png).")
    p.add_argument("--model_dir",   default="defense/models",
                   help="Directory containing 256x256_diffusion_uncond.pt.")
    p.add_argument("--image_size",  type=int, default=None,
                   help="Resize input images to this square size before processing.")

    # Noise levels
    p.add_argument("--diffpure_t",  type=int, default=400,
                   help="DiffPure noise level t (default: 400).")
    p.add_argument("--wavedm_t",    type=int, default=250,
                   help="WaveDM noise level t (default: 250).")

    # FreqDDPM (Ours) parameters — tuned to beat WaveDM
    p.add_argument("--ours_t",            type=int,   default=150,
                   help="Base DDPM noise level for Ours (default: 150).")
    p.add_argument("--ours_ll_alpha",     type=float, default=0.70,
                   help="LL subband blend weight for Ours (default: 0.70).")
    p.add_argument("--ours_hf_alpha",     type=float, default=0.20,
                   help="HF subband blend weight for Ours (default: 0.20).")
    p.add_argument("--ours_threshold_l1", type=float, default=0.020,
                   help="Level-1 wavelet soft-threshold for Ours (default: 0.020).")
    p.add_argument("--ours_threshold_l2", type=float, default=0.010,
                   help="Level-2 wavelet soft-threshold for Ours (default: 0.010).")
    p.add_argument("--ours_t_refine",     type=int,   default=25,
                   help="Polish-pass DDPM steps after fusion (0=disabled; default: 25).")
    p.add_argument("--ours_no_adaptive",  action="store_true",
                   help="Disable adaptive t scaling for Ours.")

    # Skip flags
    p.add_argument("--skip_diffpure", action="store_true",
                   help="Skip DiffPure (use when weights are unavailable).")
    p.add_argument("--skip_wavedm",   action="store_true",
                   help="Skip WaveDM.")
    p.add_argument("--skip_ours",     action="store_true",
                   help="Skip Ours / FreqDDPM.")

    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # ── Load inputs ───────────────────────────────────────────────────────────
    orig_t = load_image(args.original,    size=args.image_size)
    adv_t  = load_image(args.adversarial, size=args.image_size)
    print(f"[info] Image size: {tuple(orig_t.shape)}")

    images  = {}
    metrics = {}

    images["Original"]              = tensor_to_np(orig_t)
    images["MOMUDIG\n(Adversarial)"] = tensor_to_np(adv_t)

    # ── DiffPure ──────────────────────────────────────────────────────────────
    if not args.skip_diffpure:
        print("[→] Running DiffPure ...")
        try:
            dp_clean = run_diffpure(adv_t, args.model_dir, args.diffpure_t, device)
            key = "DiffPure\n(Purified)"
            images[key]  = tensor_to_np(dp_clean)
            metrics[key] = compute_metrics(orig_t, dp_clean, device)
            print(f"    DiffPure  {metrics[key]}")
        except FileNotFoundError as exc:
            warnings.warn(str(exc))
            print("[!] DiffPure skipped (weights missing).")

    # ── WaveDM ────────────────────────────────────────────────────────────────
    if not args.skip_wavedm:
        print("[→] Running WaveDM ...")
        try:
            # Lazy import to avoid import-time model load
            _wavedm_dir = os.path.join(os.path.dirname(__file__), "defense")
            if _wavedm_dir not in sys.path:
                sys.path.insert(0, _wavedm_dir)
            from wavedm.wavedm_purify import WaveDMPurifier
            wavedm = WaveDMPurifier(
                model_dir=args.model_dir,
                t=args.wavedm_t,
                device=str(device),
            )
            wdm_clean = wavedm.purify(adv_t)
            key = "WaveDM\n(Purified)"
            images[key]  = tensor_to_np(wdm_clean)
            metrics[key] = compute_metrics(orig_t, wdm_clean, device)
            print(f"    WaveDM    {metrics[key]}")
        except FileNotFoundError as exc:
            warnings.warn(str(exc))
            print("[!] WaveDM skipped (weights missing).")

    # ── Ours / FreqDDPM ───────────────────────────────────────────────────────
    if not args.skip_ours:
        print("[→] Running Ours (FreqDDPM) ...")
        try:
            _ours_dir = os.path.join(os.path.dirname(__file__), "defense")
            if _ours_dir not in sys.path:
                sys.path.insert(0, _ours_dir)
            from ours.ours_purify import OursPurifier
            ours = OursPurifier(
                model_dir=args.model_dir,
                t=args.ours_t,
                ll_blend_alpha=args.ours_ll_alpha,
                hf_blend_alpha=args.ours_hf_alpha,
                threshold_l1=args.ours_threshold_l1,
                threshold_l2=args.ours_threshold_l2,
                t_refine=args.ours_t_refine,
                adaptive=not args.ours_no_adaptive,
                device=str(device),
            )
            ours_clean = ours.purify(adv_t)
            key = "Ours / FreqDDPM\n(Purified)"
            images[key]  = tensor_to_np(ours_clean)
            metrics[key] = compute_metrics(orig_t, ours_clean, device)
            print(f"    Ours      {metrics[key]}")
        except Exception as exc:
            warnings.warn(f"Ours (FreqDDPM) failed: {exc}")
            print("[!] Ours skipped.")

    # ── Figure ────────────────────────────────────────────────────────────────
    save_comparison_figure(images, metrics, args.output)

    # ── Print metrics table ───────────────────────────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│          Image Reconstruction Quality Comparison             │")
    print("├──────────────────────────┬──────────┬──────────┬─────────────┤")
    print("│ Method                   │  SSIM ↑  │ PSNR ↑  │  LPIPS ↓   │")
    print("├──────────────────────────┼──────────┼──────────┼─────────────┤")
    for title, m in metrics.items():
        name = title.replace("\n", " ").ljust(24)
        print(f"│ {name} │  {m['ssim']:.3f}   │ {m['psnr']:5.1f}   │   {m['lpips']:.3f}    │")
    print("└──────────────────────────┴──────────┴──────────┴─────────────┘")


if __name__ == "__main__":
    main()
