#!/usr/bin/env python3
"""
freq_residual_vis.py
====================
Frequency-domain residual visualization for the purification process (3-D surface).

Computes the absolute log-magnitude spectrum difference between each image and
the clean reference, then renders them as 3-D surfaces:

    z(u, v) = |log(1 + |F_img(u,v)|) - log(1 + |F_original(u,v)|)|

After fftshift the surface centre corresponds to DC / low-frequency components
and the corners correspond to high frequencies.  A well-crafted adversarial
example therefore produces a smooth "valley" at the centre and dense sharp
"spikes" around the edges.  After purification those spikes collapse towards
zero -- the surface becomes nearly flat -- demonstrating that the purifier has
successfully neutralised the high-frequency adversarial energy.

Quick start
-----------
  # Run Ours purification live and show both adversarial + purified panels:
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# CJK font detection -- use Chinese labels if a suitable font is available
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
    """Return the name of the first available CJK font, or None."""
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in _CJK_CANDIDATES:
        if name in available:
            return name
    return None


_CJK_FONT = _detect_cjk_font()
if _CJK_FONT:
    plt.rcParams["font.family"] = [_CJK_FONT, "DejaVu Sans"]


def _zh(chinese, english):
    """Return Chinese label if a CJK font is available, otherwise English."""
    return chinese if _CJK_FONT else english


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_image_gray(path, size=None):
    """Load an image as a float32 HW array in [0, 1] (BT.601 luma)."""
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def load_image_tensor(path, size=None):
    """Load an image as a CHW float32 torch.Tensor in [0, 1]."""
    import torch
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# Frequency-domain helpers
# ---------------------------------------------------------------------------

def log_magnitude_spectrum(gray):
    """
    Centred log-magnitude spectrum of a 2-D grayscale image.

    Returns log(1 + |FFT2(gray)|) after fftshift, so the DC component is at
    the centre of the returned array.
    """
    F = np.fft.fft2(gray)
    F = np.fft.fftshift(F)
    return np.log1p(np.abs(F)).astype(np.float32)


def spectrum_diff_abs(original, other):
    """
    Absolute log-magnitude spectrum difference: |log|F_other| - log|F_orig||.

    Using the absolute value ensures the surface z >= 0 everywhere, giving a
    clear "mountain/valley" reading: z=0 means no distortion, large z means
    strong distortion.
    """
    return np.abs(log_magnitude_spectrum(other) - log_magnitude_spectrum(original))


def _prepare_surface(z, target=160, smooth_sigma=1.2):
    """
    Downsample and optionally smooth the surface for 3-D rendering.

    * Average-pool to at most target x target to reduce polygon count.
    * Apply a mild Gaussian blur to remove salt-and-pepper noise while
      preserving the large-scale spike / valley structure.
    """
    H, W = z.shape
    sh = max(1, H // target)
    sw = max(1, W // target)
    z_trim = z[: H - (H % sh), : W - (W % sw)]
    Ht, Wt = z_trim.shape
    zd = z_trim.reshape(Ht // sh, sh, Wt // sw, sw).mean(axis=(1, 3))
    if smooth_sigma > 0:
        zd = gaussian_filter(zd, sigma=smooth_sigma)
    return zd.astype(np.float32)


# ---------------------------------------------------------------------------
# Figure rendering
# ---------------------------------------------------------------------------

def _make_surface_axes(fig, pos, title, z,
                       vmax=None, elev=32, azim=225,
                       cmap="inferno"):
    """
    Add one 3-D surface subplot to *fig*.

    Parameters
    ----------
    pos   : (nrows, ncols, index) tuple passed to fig.add_subplot.
    z     : 2-D array of surface heights (already prepared / downsampled).
    vmax  : colour / z upper limit (0 is always the lower limit).
    elev, azim : viewing angles (degrees).
    cmap  : matplotlib colour map name.
    """
    ax = fig.add_subplot(*pos, projection="3d")

    H, W = z.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))

    if vmax is None:
        vmax = z.max()
    vmax = max(vmax, 1e-6)   # guard against all-zero surfaces

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
    ax.set_zlim(0, vmax)
    ax.set_box_aspect([1, 1, 0.55])

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(_zh("频率 u", "Frequency u"), fontsize=9, labelpad=3)
    ax.set_ylabel(_zh("频率 v", "Frequency v"), fontsize=9, labelpad=3)
    ax.set_zlabel(_zh("|Δ log|F||", "|Delta log|F||"), fontsize=9, labelpad=3)
    ax.tick_params(labelsize=7)

    # Light grey pane colour for better depth perception
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
    elev=32,
    azim=225,
    cmap="inferno",
    smooth_sigma=1.2,
    shared_zlim=True,
):
    """
    Render and save the 3-D log-spectrum absolute-residual figure.

    Parameters
    ----------
    original    : HW float32 grayscale [0,1] -- clean reference.
    purified    : HW float32 grayscale -- Ours / FreqDDPM output.
    adversarial : HW float32 grayscale -- adversarial input (optional).
                  When given the figure has two panels:
                    left  = adversarial residual (centre valley + edge spikes)
                    right = purified  residual (suppressed spikes)
    shared_zlim : If True (default) both panels share the same z/colour scale
                  so the spike suppression is directly visible.
    """
    z_pur = _prepare_surface(
        spectrum_diff_abs(original, purified),
        smooth_sigma=smooth_sigma,
    )

    if adversarial is not None:
        z_adv = _prepare_surface(
            spectrum_diff_abs(original, adversarial),
            smooth_sigma=smooth_sigma,
        )
        vmax = z_adv.max() if shared_zlim else None
    else:
        z_adv = None
        vmax = None

    ncols = 2 if z_adv is not None else 1
    figw  = 7.5 * ncols
    fig   = plt.figure(figsize=(figw, 6), dpi=dpi)
    fig.patch.set_facecolor("white")

    if z_adv is not None:
        _make_surface_axes(
            fig, (1, 2, 1),
            title=_zh(
                "对抗样本频域残差\n(Adversarial \u2212 Original)",
                "Adversarial Freq. Residual\n(Adversarial \u2212 Original)",
            ),
            z=z_adv,
            vmax=vmax,
            elev=elev, azim=azim,
            cmap=cmap,
        )
        surf_ours = _make_surface_axes(
            fig, (1, 2, 2),
            title=_zh(
                "\u51c0\u5316\u540e\u9891\u57df\u6b8b\u5dee\n(Ours / FreqDDPM \u2212 Original)",
                "Purified Freq. Residual\n(Ours / FreqDDPM \u2212 Original)",
            ),
            z=z_pur,
            vmax=vmax,
            elev=elev, azim=azim,
            cmap=cmap,
        )

        # Shared colour-bar on the right
        fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.05,
                            wspace=0.05)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.018, 0.65])
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=0, vmax=vmax),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(_zh("|Δ log|F||", "|Delta log|F||"), fontsize=9)

    else:
        surf_ours = _make_surface_axes(
            fig, (1, 1, 1),
            title=_zh(
                "\u51c0\u5316\u540e\u9891\u57df\u6b8b\u5dee\n(Ours / FreqDDPM \u2212 Original)",
                "Purified Freq. Residual\n(Ours / FreqDDPM - Original)",
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
            norm=plt.Normalize(vmin=0, vmax=z_pur.max()),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label(_zh("|Δ log|F||", "|Delta log|F||"), fontsize=9)

    fig.savefig(output_path, bbox_inches="tight", dpi=dpi,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")


# ---------------------------------------------------------------------------
# OursPurifier inference helper
# ---------------------------------------------------------------------------

def run_ours_purify(adv_path, model_dir, image_size, device_str):
    """
    Run OursPurifier on *adv_path* and return a float32 HW grayscale array.

    Lazily imported so the script is usable without guided-diffusion when
    --purified is supplied instead.
    """
    _defense_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defense")
    if _defense_dir not in sys.path:
        sys.path.insert(0, _defense_dir)

    from ours.ours_purify import OursPurifier

    adv_t = load_image_tensor(adv_path, size=image_size)
    purifier = OursPurifier(model_dir=model_dir, device=device_str)
    pur_t = purifier.purify(adv_t)   # CHW float32 [0,1]

    arr = pur_t.numpy().transpose(1, 2, 0)
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "3-D frequency-domain residual visualization "
            "(Ours / FreqDDPM vs. Original)."
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
                   help="Resize inputs to this square size before processing.")
    p.add_argument("--show_adv", action="store_true",
                   help="Add adversarial residual panel (requires --adversarial).")
    p.add_argument("--dpi", type=int, default=180,
                   help="Output DPI (default: 180).")
    p.add_argument("--elev", type=float, default=32,
                   help="3-D view elevation in degrees (default: 32).")
    p.add_argument("--azim", type=float, default=225,
                   help="3-D view azimuth in degrees (default: 225).")
    p.add_argument("--cmap", default="inferno",
                   help="Matplotlib colormap (default: inferno).")
    p.add_argument("--smooth", type=float, default=1.2,
                   help="Gaussian smoothing sigma (0 = disabled; default: 1.2).")
    p.add_argument("--no_shared_zlim", action="store_true",
                   help="Use independent z/colour limits per panel.")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda).")
    return p.parse_args()


def main():
    args = parse_args()

    orig_gray = load_image_gray(args.original, size=args.image_size)
    print(f"[info] Image size: {orig_gray.shape}")

    if args.purified:
        pur_gray = load_image_gray(args.purified, size=args.image_size)
        print(f"[info] Loaded purified image: {args.purified}")
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
            print(f"[info] Loaded adversarial image: {args.adversarial}")

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
    )


if __name__ == "__main__":
    main()
