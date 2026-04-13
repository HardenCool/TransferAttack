#!/usr/bin/env python3
"""
freq_residual_vis.py
====================
净化过程的频域残差可视化（三维曲面图）。

对"Ours / FreqDDPM 净化图"与原始干净图分别计算灰度对数幅度频谱，
然后将两者之差渲染为三维曲面（Figure 4.6 风格）。

差值定义：
    Δ(u, v) = log(1 + |F_purified(u,v)|) − log(1 + |F_original(u,v)|)

频谱经零频中心化（fftshift），因此曲面中央对应直流/低频分量，
四角对应高频分量。

Quick start
-----------
  # 仅用原始图与净化图（自动运行 Ours 净化）：
  python freq_residual_vis.py \\
      --original   path/to/clean.png \\
      --adversarial path/to/adv.png \\
      --model_dir  defense/models \\
      --output     freq_residual.png

  # 已有净化结果，直接传入：
  python freq_residual_vis.py \\
      --original  path/to/clean.png \\
      --purified  path/to/ours_purified.png \\
      --output    freq_residual.png

  # 同时画对抗样本的频域残差（用于对比）：
  python freq_residual_vis.py \\
      --original   path/to/clean.png \\
      --adversarial path/to/adv.png \\
      --model_dir  defense/models \\
      --output     freq_residual.png \\
      --show_adv

Dependencies
------------
  pip install PyWavelets matplotlib Pillow numpy
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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import)


# ─────────────────────────────────────────────────────────────────────────────
# Image I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image_gray(path: str, size: int | None = None) -> np.ndarray:
    """
    Load an image as a float32 HW array in [0, 1] (luminance / grayscale).

    The grayscale conversion uses the standard ITU-R BT.601 coefficients, the
    same weighting used in most perceptual metrics.
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0          # HWC [0,1]
    # ITU-R BT.601 luma weights
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def load_image_tensor(path: str, size: int | None = None):
    """Load image as CHW float32 torch.Tensor in [0,1]."""
    import torch
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-domain helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_magnitude_spectrum(gray: np.ndarray) -> np.ndarray:
    """
    Compute the centred log-magnitude spectrum of a 2-D grayscale image.

    Returns
    -------
    log_spec : float32 HW array
        log(1 + |FFT2(gray)|) after zero-frequency centring.
    """
    F = np.fft.fft2(gray)
    F = np.fft.fftshift(F)
    return np.log1p(np.abs(F)).astype(np.float32)


def spectrum_diff(original: np.ndarray, other: np.ndarray) -> np.ndarray:
    """
    Return log-magnitude spectrum difference: log_spec(other) − log_spec(original).

    Both arrays must be the same shape (H, W), in [0, 1].
    """
    return log_magnitude_spectrum(other) - log_magnitude_spectrum(original)


def _downsample_surface(z: np.ndarray, target: int = 128) -> np.ndarray:
    """
    Uniformly subsample ``z`` to at most ``target × target`` for 3-D rendering.

    Using average-pooling via block-reduce keeps the visual impression of the
    surface while dramatically reducing the number of polygons and rendering
    time compared with plotting every pixel.
    """
    H, W = z.shape
    sh = max(1, H // target)
    sw = max(1, W // target)
    # Trim to multiples of stride
    z_trim = z[: H - (H % sh), : W - (W % sw)]
    Ht, Wt = z_trim.shape
    return z_trim.reshape(Ht // sh, sh, Wt // sw, sw).mean(axis=(1, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Figure rendering
# ─────────────────────────────────────────────────────────────────────────────

def _make_surface_axes(fig, pos: tuple, title: str, z: np.ndarray,
                       cmap: str = "RdBu_r",
                       zlim: tuple | None = None):
    """Add one 3-D surface subplot to *fig* at *pos* (nrows, ncols, index)."""
    ax = fig.add_subplot(*pos, projection="3d")

    zd = _downsample_surface(z)
    H, W = zd.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))

    vmax = np.abs(zd).max() if zlim is None else zlim[1]
    vmin = -vmax

    surf = ax.plot_surface(
        X, Y, zd,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        linewidth=0,
        antialiased=False,
        rcount=H,
        ccount=W,
    )

    # ── Styling ──────────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("频率 u", fontsize=9, labelpad=4)
    ax.set_ylabel("频率 v", fontsize=9, labelpad=4)
    ax.set_zlabel("Δ log|F|", fontsize=9, labelpad=4)
    ax.tick_params(labelsize=7)
    ax.set_box_aspect([1, 1, 0.5])   # flatten the z-axis for clarity

    if zlim is not None:
        ax.set_zlim(zlim)

    return surf


def save_freq_residual_figure(
    original: np.ndarray,
    purified: np.ndarray,
    output_path: str,
    adversarial: np.ndarray | None = None,
    dpi: int = 180,
):
    """
    Render and save the 3-D log-spectrum residual surface figure.

    Parameters
    ----------
    original    : HW float32 grayscale array in [0, 1] — clean reference.
    purified    : HW float32 grayscale — Ours / FreqDDPM output.
    adversarial : HW float32 grayscale — adversarial input (optional, for
                  side-by-side comparison).  When supplied the figure shows
                  two panels: left=adversarial residual, right=purified residual.
    output_path : file path for the saved figure.
    dpi         : output resolution.
    """
    diff_purified = spectrum_diff(original, purified)

    # Shared z-limits so the two panels are directly comparable.
    if adversarial is not None:
        diff_adv = spectrum_diff(original, adversarial)
        vmax = max(np.abs(diff_purified).max(), np.abs(diff_adv).max())
    else:
        diff_adv = None
        vmax = np.abs(diff_purified).max()

    zlim = (-vmax, vmax)

    ncols = 2 if diff_adv is not None else 1
    fig = plt.figure(figsize=(6 * ncols, 5), dpi=dpi)
    fig.patch.set_facecolor("white")

    if diff_adv is not None:
        surf_adv = _make_surface_axes(
            fig, (1, ncols, 1),
            title="对抗样本频域残差\n(Adversarial − Original)",
            z=diff_adv,
            cmap="RdBu_r",
            zlim=zlim,
        )
        surf_ours = _make_surface_axes(
            fig, (1, ncols, 2),
            title="净化后频域残差\n(Ours / FreqDDPM − Original)",
            z=diff_purified,
            cmap="RdBu_r",
            zlim=zlim,
        )
        # Shared colour-bar between the two panels
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(
            cmap="RdBu_r",
            norm=plt.Normalize(vmin=zlim[0], vmax=zlim[1]),
        )
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cbar_ax)
        cb.set_label("Δ log|F|", fontsize=9)
    else:
        surf_ours = _make_surface_axes(
            fig, (1, 1, 1),
            title="净化后频域残差\n(Ours / FreqDDPM − Original)",
            z=diff_purified,
            cmap="RdBu_r",
            zlim=zlim,
        )
        fig.colorbar(surf_ours, ax=fig.axes[0], shrink=0.5, pad=0.1,
                     label="Δ log|F|")

    plt.tight_layout(rect=[0, 0, 0.91 if diff_adv is not None else 1.0, 1])
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[✓] 频域残差三维曲面图已保存至: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# OursPurifier inference helper
# ─────────────────────────────────────────────────────────────────────────────

def run_ours_purify(adv_path: str, model_dir: str,
                    image_size: int | None,
                    device_str: str) -> np.ndarray:
    """
    Run OursPurifier on *adv_path* and return a float32 HW grayscale array.

    The purifier is imported lazily so the script still works (for the
    ``--purified`` path) even if guided-diffusion isn't installed.
    """
    import torch

    _defense_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defense")
    if _defense_dir not in sys.path:
        sys.path.insert(0, _defense_dir)

    from ours.ours_purify import OursPurifier

    adv_t = load_image_tensor(adv_path, size=image_size)
    purifier = OursPurifier(model_dir=model_dir, device=device_str)
    pur_t = purifier.purify(adv_t)   # CHW float32 [0,1]

    arr = pur_t.numpy().transpose(1, 2, 0)  # HWC
    return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="净化过程的频域残差三维可视化（Ours / FreqDDPM vs. Original）。"
    )
    p.add_argument("--original", required=True,
                   help="原始干净图像路径。")
    # Source of the purified image: either run Ours live or supply a pre-made file.
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--adversarial",
                     help="对抗样本路径（将自动调用 OursPurifier 生成净化图）。")
    src.add_argument("--purified",
                     help="已有的净化结果图像路径（跳过净化步骤）。")

    p.add_argument("--output", default="freq_residual.png",
                   help="输出文件路径（默认: freq_residual.png）。")
    p.add_argument("--model_dir", default="defense/models",
                   help="模型权重目录（默认: defense/models）。")
    p.add_argument("--image_size", type=int, default=None,
                   help="将输入图像缩放至此尺寸（正方形）。")
    p.add_argument("--show_adv", action="store_true",
                   help="同时显示对抗样本的频域残差面板（需要 --adversarial）。")
    p.add_argument("--dpi", type=int, default=180,
                   help="输出分辨率 DPI（默认: 180）。")
    p.add_argument("--device", default="cuda",
                   help="Torch 设备（默认: cuda）。")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load original ─────────────────────────────────────────────────────────
    orig_gray = load_image_gray(args.original, size=args.image_size)
    print(f"[info] 原始图像尺寸: {orig_gray.shape}")

    # ── Obtain purified grayscale ─────────────────────────────────────────────
    if args.purified:
        pur_gray = load_image_gray(args.purified, size=args.image_size)
        print(f"[info] 加载净化图像: {args.purified}")
    else:
        print("[→] 正在运行 Ours (FreqDDPM) 净化...")
        try:
            pur_gray = run_ours_purify(
                args.adversarial, args.model_dir, args.image_size, args.device
            )
            print("[✓] 净化完成。")
        except Exception as exc:
            print(f"[✗] 净化失败: {exc}")
            raise

    # ── Optionally load adversarial for the side-by-side panel ───────────────
    adv_gray = None
    if args.show_adv:
        if args.adversarial is None:
            warnings.warn(
                "--show_adv 需要 --adversarial 参数，已忽略。"
            )
        else:
            adv_gray = load_image_gray(args.adversarial, size=args.image_size)
            print(f"[info] 加载对抗样本: {args.adversarial}")

    # ── Render ────────────────────────────────────────────────────────────────
    save_freq_residual_figure(
        original=orig_gray,
        purified=pur_gray,
        output_path=args.output,
        adversarial=adv_gray,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
