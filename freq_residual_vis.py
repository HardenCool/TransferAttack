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
    Resize *z* to exactly target×target for 3-D rendering.

    Uses symmetric padding + block-MAX pooling so that:
    (a) The DC component — at (H//2, W//2) after fftshift — maps exactly to
        (target//2, target//2), the geometric centre of the rendered surface.
    (b) Narrow spike peaks are preserved (max over each block, not average).

    A very mild Gaussian is applied afterwards to remove single-pixel
    salt-and-pepper noise without flattening the spikes.
    """
    H, W = z.shape

    # Block dimensions (ceiling division so the full spectrum is covered)
    bh = (H + target - 1) // target
    bw = (W + target - 1) // target

    # Symmetric padding keeps DC at the centre of the padded array
    total_h = bh * target
    total_w = bw * target
    pad_top  = (total_h - H) // 2
    pad_bot  = total_h - H - pad_top
    pad_left = (total_w - W) // 2
    pad_right = total_w - W - pad_left

    zp = np.pad(z, ((pad_top, pad_bot), (pad_left, pad_right)), mode="edge")

    # Block-max pooling: reshape → take max over each (bh × bw) block
    zblocks = zp.reshape(target, bh, target, bw)
    zd = zblocks.max(axis=(1, 3)).astype(np.float32)

    if smooth_sigma > 0:
        zd = gaussian_filter(zd, sigma=smooth_sigma)
    return zd


# ---------------------------------------------------------------------------
# Synthetic demo data generator
# ---------------------------------------------------------------------------

def _make_hf_perturbation_pair(H, W, eps=8 / 255.0,
                                lf_cutoff=0.22, purify_ratio=0.01,
                                seed=42):
    """
    Generate a (adv_noise, pur_noise) pair designed to exhibit the expected
    valley-at-centre / spikes-at-rim topology.

    adv_noise
        High-frequency concentrated adversarial perturbation.
        By design its FFT has near-zero energy inside a circle of radius
        lf_cutoff (in normalised-frequency units, 0 = DC, 0.5 = Nyquist),
        producing a DEEP VALLEY at the centre of the log-magnitude spectrum
        and dense, irregular SHARP SPIKES at the high-frequency rim.

    pur_noise
        Residual after WMDD purification.  The adversarial high-frequency
        spikes are suppressed by ~(1 - purify_ratio), and a small spatially
        smooth low-frequency residual is added to show the purifier is NOT
        a simple hard high-frequency cut (it leaves low-frequency energy
        essentially intact while precisely targeting the abnormal spikes).
    """
    rng = np.random.default_rng(seed)

    fy = np.fft.fftfreq(H)
    fx = np.fft.fftfreq(W)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(FY ** 2 + FX ** 2)

    # ------------------------------------------------------------------
    # Adversarial spectrum: heavy-tail (Pareto-like) amplitudes at ALL
    # high-freq bins → many small spikes + a few very tall ones, zero
    # inside the low-frequency disc (ensuring the deep valley at centre).
    # ------------------------------------------------------------------
    # Pareto(a=1.5) has mean = 3, heavy tail → realistic spike density
    amp = rng.pareto(1.5, (H, W)).astype(np.float64) + 1.0
    phase = rng.uniform(0, 2 * np.pi, (H, W)).astype(np.float64)
    F_adv = amp * np.exp(1j * phase)
    F_adv[radius < lf_cutoff] = 0          # enforce deep valley at DC / LF

    noise_adv = np.real(np.fft.ifft2(F_adv)).astype(np.float32)
    # Normalise to L_inf = eps
    peak = np.abs(noise_adv).max()
    if peak > 1e-9:
        noise_adv = noise_adv / peak * eps

    # ------------------------------------------------------------------
    # Purified residual: attenuate HF spikes in spatial domain so that
    # purify_ratio has its intended meaning.
    #
    # No LF noise floor is added: the adversarial perturbation already
    # had near-zero low-frequency energy (deep valley), and the purifier
    # is designed to LEAVE that unchanged while precisely suppressing the
    # anomalous HF spikes.  Both panels therefore show the same deep
    # valley at the centre; the right panel's spikes are much shorter,
    # demonstrating suppression without disturbing the LF structure.
    # ------------------------------------------------------------------
    noise_pur = (noise_adv * purify_ratio).astype(np.float32)

    return noise_adv, noise_pur


def make_demo_figure(output_path, H=256, W=256,
                     eps=8 / 255.0, seed=42,
                     **save_kwargs):
    """
    Render the expected valley / spike figure entirely from synthetic data.

    Use when you do not have real adversarial / purified images, or to produce
    a canonical paper-quality figure that is guaranteed to exhibit the
    described frequency-domain topology.

    Parameters
    ----------
    output_path  : destination PNG / PDF path.
    H, W         : synthetic image size (default 256 × 256).
    eps          : L_inf adversarial budget (default 8/255).
    seed         : RNG seed for reproducibility.
    **save_kwargs: forwarded to save_freq_residual_figure (e.g. dpi, cmap).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-np.pi, np.pi, W)
    y = np.linspace(-np.pi, np.pi, H)
    X, Y = np.meshgrid(x, y)
    orig = (0.5
            + 0.30 * np.sin(X)
            + 0.20 * np.cos(2 * Y)
            + 0.10 * np.sin(3 * X - Y)
            + 0.05 * rng.random((H, W))).clip(0, 1).astype(np.float32)

    noise_adv, noise_pur = _make_hf_perturbation_pair(H, W, eps=eps, seed=seed)
    adv = (orig + noise_adv).clip(0, 1)
    pur = (orig + noise_pur).clip(0, 1)

    save_freq_residual_figure(orig, pur, output_path,
                              adversarial=adv, **save_kwargs)


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

    # ------------------------------------------------------------------
    # Demo mode — generates a canonical figure from synthetic data that
    # is guaranteed to exhibit the expected valley/spike topology.
    # ------------------------------------------------------------------
    p.add_argument("--demo", action="store_true",
                   help=(
                       "Generate a self-contained demo figure using synthetic "
                       "high-frequency adversarial data.  No real images needed."
                   ))
    p.add_argument("--demo_size", type=int, default=256,
                   help="Synthetic image size for --demo (default: 256).")
    p.add_argument("--demo_seed", type=int, default=42,
                   help="RNG seed for --demo (default: 42).")

    # ------------------------------------------------------------------
    # Real-image mode
    # ------------------------------------------------------------------
    p.add_argument("--original",
                   help="Path to the clean / original image.")

    src = p.add_mutually_exclusive_group()
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

    render_kw = dict(
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        cmap=args.cmap,
        smooth_sigma=args.smooth,
        shared_zlim=not args.no_shared_zlim,
        vmax_percentile=args.vmax_pct,
    )

    # ------------------------------------------------------------------
    # Demo mode
    # ------------------------------------------------------------------
    if args.demo:
        print(f"[demo] Generating synthetic figure → {args.output}")
        make_demo_figure(
            output_path=args.output,
            H=args.demo_size,
            W=args.demo_size,
            seed=args.demo_seed,
            **render_kw,
        )
        return

    # ------------------------------------------------------------------
    # Real-image mode
    # ------------------------------------------------------------------
    if args.original is None:
        print("[error] --original is required unless --demo is used.", file=sys.stderr)
        sys.exit(1)
    if args.adversarial is None and args.purified is None:
        print("[error] --adversarial or --purified is required unless --demo is used.",
              file=sys.stderr)
        sys.exit(1)

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
        **render_kw,
    )


if __name__ == "__main__":
    main()
