"""
WaveDM: Wavelet-Domain Diffusion Purification for adversarial examples.

Algorithm:
  1. Apply 2-D Haar DWT to the adversarial image -> (LL, LH, HL, HH) subbands.
  2. Soft-threshold the high-frequency subbands to remove adversarial noise in the
     frequency domain (frequency pre-denoising).
  3. Reconstruct the image via IDWT to obtain a partially-denoised image.
  4. Apply the standard DDPM forward-reverse diffusion at a lighter noise level
     (t_wavedm < t_diffpure) for final polishing, using OpenAI's pretrained
     256x256_diffusion_uncond.pt model.

The lower noise level combined with the wavelet pre-denoising yields better SSIM
and LPIPS than vanilla DiffPure while keeping inference cost comparable.

Pretrained weight required:
  256x256_diffusion_uncond.pt  (identical to DiffPure)
  Download: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

Usage:
    from defense.wavedm.wavedm_purify import WaveDMPurifier

    purifier = WaveDMPurifier(model_dir='/path/to/weights', t=250, device='cuda')
    # clean_img is a float32 tensor in [0, 1], shape (C, H, W) or (B, C, H, W)
    clean_img = purifier.purify(adv_img)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# ── pywt is a lightweight dependency (PyWavelets) ────────────────────────────
try:
    import pywt
except ImportError as e:
    raise ImportError(
        "PyWavelets is required for WaveDM. Install it with: pip install PyWavelets"
    ) from e

# ── Add DiffPure paths so we can reuse its guided-diffusion backbone ──────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DIFFPURE_DIR = os.path.join(os.path.dirname(_THIS_DIR), "diffpure")
for _p in [_DIFFPURE_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


# ─────────────────────────────────────────────────────────────────────────────
# Wavelet helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dwt2(x: np.ndarray, wavelet: str = "haar") -> tuple:
    """2-D single-level DWT on a CHW float array.  Returns (LL, (LH, HL, HH))."""
    coeffs = []
    for c in range(x.shape[0]):
        LL_c, (LH_c, HL_c, HH_c) = pywt.dwt2(x[c], wavelet)
        coeffs.append((LL_c, LH_c, HL_c, HH_c))
    LL  = np.stack([c[0] for c in coeffs], axis=0)
    LH  = np.stack([c[1] for c in coeffs], axis=0)
    HL  = np.stack([c[2] for c in coeffs], axis=0)
    HH  = np.stack([c[3] for c in coeffs], axis=0)
    return LL, LH, HL, HH


def _idwt2(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray,
           HH: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """2-D single-level IDWT.  Returns CHW float array."""
    recon = []
    for c in range(LL.shape[0]):
        r = pywt.idwt2((LL[c], (LH[c], HL[c], HH[c])), wavelet)
        recon.append(r)
    return np.stack(recon, axis=0)


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    """Element-wise soft thresholding."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def wavelet_predenoise(img: np.ndarray, threshold: float = 0.02,
                       wavelet: str = "haar") -> np.ndarray:
    """
    Frequency-domain pre-denoising via wavelet soft thresholding.

    Parameters
    ----------
    img       : float32 CHW array in [0, 1].
    threshold : soft-threshold applied to LH / HL / HH subbands.
    wavelet   : mother wavelet (default 'haar').

    Returns
    -------
    denoised  : float32 CHW array in [0, 1].
    """
    LL, LH, HL, HH = _dwt2(img, wavelet)
    # Threshold high-frequency subbands only; LL (low-freq) is untouched.
    LH = _soft_threshold(LH, threshold)
    HL = _soft_threshold(HL, threshold)
    HH = _soft_threshold(HH, threshold)
    recon = _idwt2(LL, LH, HL, HH, wavelet)
    return np.clip(recon, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# WaveDM purifier class
# ─────────────────────────────────────────────────────────────────────────────

class WaveDMPurifier:
    """
    Wavelet-Domain Diffusion Purification (WaveDM).

    Parameters
    ----------
    model_dir       : folder that contains ``256x256_diffusion_uncond.pt``.
    t               : noise level for the DDPM forward-reverse pass (default 250,
                      lighter than DiffPure's 400).
    wavelet         : mother wavelet for frequency decomposition (default 'haar').
    wt_threshold    : soft-threshold for high-frequency subbands (default 0.02).
    device          : torch device string (default 'cuda').
    """

    # Default model configuration (mirrors DiffPure's imagenet.yml)
    _MODEL_CFG = dict(
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
    )

    def __init__(
        self,
        model_dir: str = "defense/diffpure/../models",
        t: int = 250,
        wavelet: str = "haar",
        wt_threshold: float = 0.02,
        device: str = "cuda",
    ):
        self.t = t
        self.wavelet = wavelet
        self.wt_threshold = wt_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        weights_path = os.path.join(model_dir, "256x256_diffusion_uncond.pt")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"WaveDM requires the pretrained weight file:\n  {weights_path}\n"
                "Download it with:\n"
                "  wget -O defense/models/256x256_diffusion_uncond.pt \\\n"
                "    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
            )

        cfg = model_and_diffusion_defaults()
        cfg.update(self._MODEL_CFG)
        model, diffusion = create_model_and_diffusion(**cfg)
        model.load_state_dict(
            torch.load(weights_path, map_location="cpu"), strict=False
        )
        model.requires_grad_(False).eval().to(self.device)
        if cfg["use_fp16"]:
            model.convert_to_fp16()

        self.model = model
        self.diffusion = diffusion
        self.betas = torch.from_numpy(diffusion.betas).float().to(self.device)
        self._alphas_cumprod = (1.0 - self.betas).cumprod(dim=0)

        print(f"[WaveDM] Model loaded from {weights_path}")

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def purify(self, adv_image: torch.Tensor) -> torch.Tensor:
        """
        Purify a batch of adversarial images.

        Parameters
        ----------
        adv_image : float32 tensor in **[0, 1]**, shape (B, C, H, W) or (C, H, W).

        Returns
        -------
        clean     : float32 tensor in [0, 1], same shape as input.
        """
        batched = adv_image.dim() == 4
        if not batched:
            adv_image = adv_image.unsqueeze(0)

        B, C, H, W = adv_image.shape

        # ── Step 1: Wavelet pre-denoising ────────────────────────────────
        wt_clean = []
        for b in range(B):
            arr = adv_image[b].cpu().numpy()            # CHW, [0,1]
            denoised = wavelet_predenoise(arr, self.wt_threshold, self.wavelet)
            wt_clean.append(torch.from_numpy(denoised))
        x0 = torch.stack(wt_clean, dim=0).to(self.device)  # (B,C,H,W) [0,1]

        # ── Step 2: Resize to 256×256 if needed ─────────────────────────
        orig_size = (H, W)
        if H != 256 or W != 256:
            x0 = F.interpolate(x0, size=(256, 256), mode="bilinear", align_corners=False)

        # ── Step 3: DDPM forward (add noise at level t) ─────────────────
        x0_scaled = (x0 - 0.5) * 2.0                       # rescale to [-1,1]
        e = torch.randn_like(x0_scaled)
        a = self._alphas_cumprod[self.t - 1]
        x_t = x0_scaled * a.sqrt() + e * (1.0 - a).sqrt()

        # ── Step 4: DDPM reverse (denoise back to t=0) ───────────────────
        x = x_t
        for i in reversed(range(self.t)):
            t_batch = torch.tensor([i] * B, device=self.device)
            x = self.diffusion.p_sample(
                self.model, x, t_batch,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
            )["sample"]

        # ── Step 5: Rescale back to [0,1] and original size ─────────────
        x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
        if orig_size != (256, 256):
            x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)

        if not batched:
            x = x.squeeze(0)
        return x
