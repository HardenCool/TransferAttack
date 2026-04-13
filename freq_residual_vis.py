#!/usr/bin/env python3
"""
freq_residual_vis.py
====================
Frequency-domain residual visualization for adversarial purification (3-D surface).

For each image the spectrum of its spatial-domain difference from the clean
reference is computed and rendered as a 3-D surface:

    z(u, v) = log(1 + |FFT2( img − original )|)   after fftshift

Why this formula?
-----------------
A well-crafted adversarial perturbation is perceptually invisible (L_inf norm
<= 8/255), meaning it barely changes the dominant LOW-frequency content of the
image.  The FFT of the perturbation (img - original) therefore has near-zero
energy at the centre (DC / low frequencies) → DEEP VALLEY.
At the same time, the attack algorithm must inject enough distortion to fool the
classifier, concentrating that energy in MID/HIGH frequencies → SHARP SPIKES at
the outer rim of the surface.

After purification the spikes collapse: the purifier suppresses the adversarial
high-frequency energy while leaving the low-frequency semantic content intact,
so the right-hand surface is nearly flat with very small amplitude throughout.

Quick start
-----------
  # Run purification live + show adversarial panel for comparison:
  python freq_residual_vis.py \\
      --original    path/to/clean.png \\
      --adversarial path/to/adv.png \\
      --model_dir   defense/models \\
      --show_adv \\
      --output      freq_residual_comparison.png

  # Use a pre-generated purified image (skip purification):
  python freq_residual_vis.py \\
      --original  path/to/clean.png \\
      --purified  path/to/ours_out.png \\
      --output    freq_residual.png

Dependencies
------------
  pip install matplotlib Pillow numpy scipy
"""

import argparse
import os
import sys
import warnings

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# CJK font detection
# ---------------------------------------------------------------------------

_CJK_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "SimSun", "FangSong",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
    "Noto Sans CJK SC", "Noto Sans SC",
    "Source Han Sans SC", "Source Han Sans CN",
    "PingFang SC", "Hiragino Sans GB",
    "Arial Unicode MS",
]


def _detect_cjk_font():
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


_CJK_FONT = _detect_cjk_font()
if _CJK_FONT:
    plt.rcParams["font.family"] = [_CJK_FONT, "DejaVu Sans"]


def _zh(chinese, english):
    return chinese if _CJK_FONT else english


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image_gray(path, size=None):
    """Load image as float32 HW array in [0,1] (BT.601 luma)."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def load_image_tensor(path, size=None):
    """Load image as CHW float32 torch.Tensor in [0,1]."""
    import torch
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# Frequency-domain core
# ---------------------------------------------------------------------------

def perturbation_spectrum(original, other):
    """
    Log-magnitude spectrum of the SPATIAL-DOMAIN difference: other − original.

        z(u, v) = log(1 + |fftshift(FFT2(other − original))|)

    This is the correct formula for showing WHERE the distortion energy lives:
    * Centre   (DC / low freq):  perturbation has near-zero energy → deep valley.
    * Periphery (high freq):     adversarial energy concentrates  → sharp spikes.

    Using the *spectrum of the difference* (not the difference of spectra) is
    essential — the latter mixes in the original image's own spectrum and produces
    a misleading result.
    """
    diff = other.astype(np.float32) - original.astype(np.float32)
    F = np.fft.fft2(diff)
    F = np.fft.fftshift(F)
    return np.log1p(np.abs(F)).astype(np.float32)


def _prepare_surface(z, target=150, smooth_sigma=0.8):
    """
    Subsample *z* to at most target×target for 3-D rendering.

    Uses STRIDE-based subsampling (pick every k-th sample) rather than
    average-pooling, so narrow spike peaks are preserved.  A very mild
    Gaussian is applied afterwards to remove single-pixel salt-and-pepper
    noise without flattening the spikes.
    """
    H, W = z.shape
    sh = max(1, H // target)
    sw = max(1, W // target)
    zd = z[::sh, ::sw]
    # Trim to exactly target size
    zd = zd[:target, :target]
    if smooth_sigma > 0:
        zd = gaussian_filter(zd, sigma=smooth_sigma)
    return zd.astype(np.float32)


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def _make_surface_axes(fig, pos, title, z,
                       vmax=None, elev=30, azim=225,
                       cmap="inferno"):
    """Add one 3-D surface panel to *fig*."""
    ax = fig.add_subplot(*pos, projection="3d")

    H, W = z.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))

    if vmax is None:
        # 99th-percentile clip avoids one extreme outlier dominating the scale
        vmax = float(np.percentile(z, 99))
    vmax = max(vmax, 1e-6)

    surf = ax.plot_surface(
        X, Y, z,
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        linewidth=0,
        antialiased=True,
        rcount=H,
        ccount=W,
        shade=True,
    )

    ax.view_init(elev=elev, azim=azim)
    ax.set_zlim(0, vmax * 1.05)
    ax.set_box_aspect([1, 1, 0.6])

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(_zh("频率 u", "Frequency u"), fontsize=9, labelpad=3)
    ax.set_ylabel(_zh("频率 v", "Frequency v"), fontsize=9, labelpad=3)
    ax.set_zlabel(_zh("log|F(Δ)|", "log|F(Delta)|"), fontsize=9, labelpad=3)
    ax.tick_params(labelsize=7)

    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor((0.94, 0.94, 0.94, 0.6))
    ax.yaxis.pane.set_facecolor((0.94, 0.94, 0.94, 0.6))
    ax.zaxis.pane.set_facecolor((0.94, 0.94, 0.94, 0.6))
    ax.grid(True, linewidth=0.4, color="white")

    return surf


def save_freq_residual_figure(
    original,
    purified,
    output_path,
    adversarial=None,
    dpi=180,
    elev=30,
    azim=225,
    cmap="inferno",
    smooth_sigma=0.8,
    shared_zlim=True,
    vmax_percentile=99,
):
    """
    Render and save the 3-D frequency-domain perturbation spectrum figure.

    Parameters
    ----------
    original    : HW float32 grayscale [0,1] -- clean reference.
    purified    : HW float32 grayscale -- Ours / FreqDDPM output.
    adversarial : HW float32 grayscale -- adversarial input (optional).
                  When given:
                    left  panel = adversarial perturbation spectrum
                                  (expected: deep valley at centre, spikes at rim)
                    right panel = purified perturbation spectrum
                                  (expected: spikes suppressed, nearly flat)
    shared_zlim : Use the same z/colour scale for both panels (default True),
                  making the spike suppression immediately visible.
    """
    z_pur = _prepare_surface(
        perturbation_spectrum(original, purified),
        smooth_sigma=smooth_sigma,
    )

    if adversarial is not None:
        z_adv = _prepare_surface(
            perturbation_spectrum(original, adversarial),
            smooth_sigma=smooth_sigma,
        )
        if shared_zlim:
            vmax = float(np.percentile(z_adv, vmax_percentile))
        else:
            vmax = None
    else:
        z_adv = None
        vmax = None

    ncols = 2 if z_adv is not None else 1
    fig = plt.figure(figsize=(7.5 * ncols, 6.2), dpi=dpi)
    fig.patch.set_facecolor("white")

    if z_adv is not None:
        _make_surface_axes(
            fig, (1, 2, 1),
            title=_zh(
                "对抗样本频谱残差\n(Adversarial \u2212 Original)",
                "Adversarial Perturbation Spectrum\n(Adversarial \u2212 Original)",
            ),
            z=z_adv,
            vmax=vmax,
            elev=elev, azim=azim,
            cmap=cmap,
        )
        _make_surface_axes(
            fig, (1, 2, 2),
            title=_zh(
                "\u51c0\u5316\u540e\u9891\u57df\u6b8b\u5dee\n(Purified \u2212 Original)",
                "Purified Perturbation Spectrum\n(Purified \u2212 Original)",
            ),
            z=z_pur,
            vmax=vmax,
            elev=elev, azim=azim,
            cmap=cmap,
        )

        fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.05,
                            wspace=0.08)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.65])
        _vmax_cb = vmax if vmax is not None else z_adv.max()
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=0, vmax=_vmax_cb),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(_zh("log|F(Δ)|", "log|F(Delta)|"), fontsize=9)

    else:
        _make_surface_axes(
            fig, (1, 1, 1),
            title=_zh(
                "\u51c0\u5316\u540e\u9891\u57df\u6b8b\u5dee\n(Purified \u2212 Original)",
                "Purified Perturbation Spectrum\n(Purified \u2212 Original)",
            ),
            z=z_pur,
            vmax=vmax,
            elev=elev, azim=azim,
            cmap=cmap,
        )
        fig.subplots_adjust(left=0.05, right=0.82, top=0.92, bottom=0.05)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=0, vmax=float(np.percentile(z_pur, 99))),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(_zh("log|F(Δ)|", "log|F(Delta)|"), fontsize=9)

    fig.savefig(output_path, bbox_inches="tight", dpi=dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


# ---------------------------------------------------------------------------
# OursPurifier inference helper
# ---------------------------------------------------------------------------

def run_ours_purify(adv_path, model_dir, image_size, device_str):
    """Run OursPurifier and return float32 HW grayscale [0,1]."""
    _defense_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defense")
    if _defense_dir not in sys.path:
        sys.path.insert(0, _defense_dir)

    from ours.ours_purify import OursPurifier

    adv_t = load_image_tensor(adv_path, size=image_size)
    purifier = OursPurifier(model_dir=model_dir, device=device_str)
    pur_t = purifier.purify(adv_t)

    arr = pur_t.numpy().transpose(1, 2, 0)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "3-D frequency-domain perturbation spectrum visualization "
            "(adversarial vs. purified)."
        )
    )
    p.add_argument("--original", required=True,
                   help="Path to the clean / original image.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adversarial",
                     help="Adversarial image path (OursPurifier will be run).")
    src.add_argument("--purified",
                     help="Pre-computed purified image path (skip purification).")

    p.add_argument("--output", default="freq_residual.png",
                   help="Output file path (default: freq_residual.png).")
    p.add_argument("--model_dir", default="defense/models",
                   help="Model weights directory (default: defense/models).")
    p.add_argument("--image_size", type=int, default=None,
                   help="Resize inputs to this square size.")
    p.add_argument("--show_adv", action="store_true",
                   help="Show adversarial panel (requires --adversarial).")
    p.add_argument("--dpi", type=int, default=180,
                   help="Output DPI (default: 180).")
    p.add_argument("--elev", type=float, default=30,
                   help="3-D view elevation degrees (default: 30).")
    p.add_argument("--azim", type=float, default=225,
                   help="3-D view azimuth degrees (default: 225).")
    p.add_argument("--cmap", default="inferno",
                   help="Matplotlib colormap (default: inferno).")
    p.add_argument("--smooth", type=float, default=0.8,
                   help="Gaussian smoothing sigma (0=off; default: 0.8).")
    p.add_argument("--vmax_pct", type=float, default=99,
                   help="Percentile for vmax clipping (default: 99).")
    p.add_argument("--no_shared_zlim", action="store_true",
                   help="Independent z-scale per panel.")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda).")
    return p.parse_args()


def main():
    args = parse_args()

    orig_gray = load_image_gray(args.original, size=args.image_size)
    print(f"[info] Image size: {orig_gray.shape}")

    if args.purified:
        pur_gray = load_image_gray(args.purified, size=args.image_size)
        print(f"[info] Loaded purified: {args.purified}")
    else:
        print("[->] Running Ours (FreqDDPM) purification ...")
        try:
            pur_gray = run_ours_purify(
                args.adversarial, args.model_dir, args.image_size, args.device
            )
            print("[OK] Purification complete.")
        except Exception as exc:
            print(f"[!!] Purification failed: {exc}")
            raise

    adv_gray = None
    if args.show_adv:
        if args.adversarial is None:
            warnings.warn("--show_adv requires --adversarial; ignored.")
        else:
            adv_gray = load_image_gray(args.adversarial, size=args.image_size)
            print(f"[info] Loaded adversarial: {args.adversarial}")

    save_freq_residual_figure(
        original=orig_gray,
        purified=pur_gray,
        output_path=args.output,
        adversarial=adv_gray,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        cmap=args.cmap,
        smooth_sigma=args.smooth,
        shared_zlim=not args.no_shared_zlim,
        vmax_percentile=args.vmax_pct,
    )


if __name__ == "__main__":
    main()
