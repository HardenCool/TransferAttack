"""
Ours: Frequency-Guided Latent Diffusion Purification (FreqLDM).

This method is the main contribution of the paper.  It improves upon WaveDM by
replacing the pixel-domain DDPM backbone with a Latent Diffusion Model (LDM),
specifically Stable Diffusion v1-5, and incorporating wavelet-based frequency
guidance during the reverse process.

Key innovations over WaveDM / DiffPure:
  1. **Latent-space diffusion** — operates in the compact perceptual latent space
     of the VAE encoder, which yields better LPIPS and SSIM scores while keeping
     inference fast.
  2. **Frequency-guided output blending** — the LL (low-frequency) subband of the
     original/wavelet-pre-denoised image is blended back into the LDM output to
     preserve large-scale structure, improving SSIM.
  3. **Adaptive strength** — the img2img noise strength is estimated from the
     measured l∞-norm of the adversarial perturbation so that clean images are
     modified minimally while heavily perturbed images receive stronger denoising.

Pretrained weight (downloaded automatically via HuggingFace Hub):
  runwayml/stable-diffusion-v1-5
  (~4 GB, stored in ~/.cache/huggingface by default)

Alternatively, set the environment variable HF_HOME to a custom cache directory,
or pass ``model_id`` to point at a local directory / alternative model.

Usage:
    from defense.ours.ours_purify import OursPurifier

    purifier = OursPurifier(device='cuda')          # loads SD weights once
    clean_img = purifier.purify(adv_img)            # (C,H,W) or (B,C,H,W) in [0,1]
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# ── PyWavelets (same as WaveDM) ───────────────────────────────────────────────
try:
    import pywt
except ImportError as e:
    raise ImportError(
        "PyWavelets is required.  Install with: pip install PyWavelets"
    ) from e

# ── HuggingFace Diffusers ─────────────────────────────────────────────────────
try:
    from diffusers import StableDiffusionImg2ImgPipeline
except ImportError as e:
    raise ImportError(
        "diffusers is required for OursPurifier.  Install with:\n"
        "  pip install diffusers[torch] transformers accelerate"
    ) from e

from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Wavelet helper (shared with WaveDM)
# ─────────────────────────────────────────────────────────────────────────────

def _dwt2(x: np.ndarray, wavelet: str = "haar") -> tuple:
    """Single-level 2-D DWT on a CHW float array."""
    LL_list, LH_list, HL_list, HH_list = [], [], [], []
    for c in range(x.shape[0]):
        LL_c, (LH_c, HL_c, HH_c) = pywt.dwt2(x[c], wavelet)
        LL_list.append(LL_c)
        LH_list.append(LH_c)
        HL_list.append(HL_c)
        HH_list.append(HH_c)
    return (
        np.stack(LL_list, axis=0),
        np.stack(LH_list, axis=0),
        np.stack(HL_list, axis=0),
        np.stack(HH_list, axis=0),
    )


def _idwt2(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray,
           HH: np.ndarray, wavelet: str = "haar") -> np.ndarray:
    """Single-level 2-D IDWT, returns CHW float array."""
    recon = []
    for c in range(LL.shape[0]):
        r = pywt.idwt2((LL[c], (LH[c], HL[c], HH[c])), wavelet)
        recon.append(r)
    return np.stack(recon, axis=0)


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert a CHW float [0,1] tensor to a PIL RGB image."""
    arr = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a CHW float [0,1] tensor."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-Guided Latent Diffusion (Ours)
# ─────────────────────────────────────────────────────────────────────────────

class OursPurifier:
    """
    Frequency-Guided Latent Diffusion Purification (FreqLDM — 'Ours').

    Parameters
    ----------
    model_id          : HuggingFace model id or local path to a
                        StableDiffusionImg2ImgPipeline-compatible checkpoint.
                        Default: ``"runwayml/stable-diffusion-v1-5"``.
    base_strength     : base img2img noise strength (fraction of total steps).
                        0.0 = no change; 1.0 = generate from scratch.
                        Values in [0.30, 0.45] give the best SSIM / LPIPS balance.
                        Default: 0.35.
    guidance_scale    : classifier-free guidance scale; 1.0 disables it so that
                        the model acts purely as a denoiser.
                        Default: 1.0.
    num_inf_steps     : number of DDIM denoising steps.  25-50 is usually enough.
                        Default: 30.
    ll_blend_alpha    : weight for low-frequency (LL subband) blending.
                        Higher alpha preserves more large-scale structure.
                        Default: 0.15.
    wavelet           : mother wavelet for frequency analysis.  Default: 'haar'.
    wt_threshold      : soft-threshold applied to HF subbands in pre-denoising.
                        Default: 0.018.
    sd_size           : resolution used internally by SD (512 × 512 for SD-v1.5;
                        768 × 768 for SD-v2-base).  Default: 512.
    device            : torch device.  Default: 'cuda'.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        base_strength: float = 0.35,
        guidance_scale: float = 1.0,
        num_inf_steps: int = 30,
        ll_blend_alpha: float = 0.15,
        wavelet: str = "haar",
        wt_threshold: float = 0.018,
        sd_size: int = 512,
        device: str = "cuda",
    ):
        self.base_strength = base_strength
        self.guidance_scale = guidance_scale
        self.num_inf_steps = num_inf_steps
        self.ll_blend_alpha = ll_blend_alpha
        self.wavelet = wavelet
        self.wt_threshold = wt_threshold
        self.sd_size = sd_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        print(f"[Ours] Loading Stable Diffusion model: {model_id} ...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("[Ours] Model ready.")

    # ------------------------------------------------------------------ #
    # Adaptive strength estimation
    # ------------------------------------------------------------------ #

    # Threshold below which the perturbation is considered benign (ε=4/255 ≈ 0.016)
    _BASE_EPS: float = 0.016
    # Each doubling of perturbation beyond _BASE_EPS adds this much extra strength
    _STRENGTH_SCALE_FACTOR: float = 0.15

    @staticmethod
    def _estimate_strength(adv: torch.Tensor, clean_approx: torch.Tensor,
                           base: float) -> float:
        """
        Adaptively scale the img2img strength based on the perturbation magnitude.

        adv and clean_approx are CHW or (1,C,H,W) tensors in [0,1].
        """
        _BASE_EPS = 0.016          # ε=4/255
        _STRENGTH_SCALE_FACTOR = 0.15
        diff = (adv.float() - clean_approx.float()).abs().max().item()
        scale = 1.0 + max(0.0, diff - _BASE_EPS) / _BASE_EPS * _STRENGTH_SCALE_FACTOR
        return float(min(base * scale, 0.65))

    # ------------------------------------------------------------------ #
    # Single-image purification (CHW, float32, [0,1])
    # ------------------------------------------------------------------ #

    def _purify_single(self, adv: torch.Tensor) -> torch.Tensor:
        C, H, W = adv.shape
        arr = adv.cpu().numpy()

        # ── Wavelet pre-denoising ────────────────────────────────────────
        LL, LH, HL, HH = _dwt2(arr, self.wavelet)
        LH_t = _soft_threshold(LH, self.wt_threshold)
        HL_t = _soft_threshold(HL, self.wt_threshold)
        HH_t = _soft_threshold(HH, self.wt_threshold)
        wt_clean = _idwt2(LL, LH_t, HL_t, HH_t, self.wavelet)
        wt_clean = np.clip(wt_clean, 0.0, 1.0).astype(np.float32)
        wt_tensor = torch.from_numpy(wt_clean)

        # ── Adaptive strength ─────────────────────────────────────────────
        strength = self._estimate_strength(adv, wt_tensor, self.base_strength)

        # ── Resize to SD internal resolution ──────────────────────────────
        pil_in = _tensor_to_pil(wt_tensor)
        orig_size = pil_in.size          # (W, H) PIL convention
        pil_resized = pil_in.resize((self.sd_size, self.sd_size), Image.LANCZOS)

        # ── Stable Diffusion img2img ───────────────────────────────────────
        result = self.pipe(
            prompt="",
            image=pil_resized,
            strength=strength,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inf_steps,
        ).images[0]

        # ── Resize back to original resolution ───────────────────────────
        result = result.resize(orig_size, Image.LANCZOS)
        out_tensor = _pil_to_tensor(result)

        # ── Frequency-guided LL blending ─────────────────────────────────
        # Replace a fraction of the LL subband in the LDM output with the
        # LL subband of the wavelet-pre-denoised input, helping to preserve
        # large-scale structure and boost SSIM.
        out_arr = out_tensor.numpy()
        LL_out, LH_out, HL_out, HH_out = _dwt2(out_arr, self.wavelet)

        # LL from wavelet-pre-denoised adversarial image (cleaner low-freq)
        LL_src, _, _, _ = _dwt2(wt_clean, self.wavelet)

        # Blend: alpha * LL_src + (1-alpha) * LL_out
        alpha = self.ll_blend_alpha
        LL_blended = alpha * LL_src + (1.0 - alpha) * LL_out

        blended_arr = _idwt2(LL_blended, LH_out, HL_out, HH_out, self.wavelet)
        blended_arr = np.clip(blended_arr, 0.0, 1.0).astype(np.float32)

        return torch.from_numpy(blended_arr)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def purify(self, adv_image: torch.Tensor) -> torch.Tensor:
        """
        Purify adversarial image(s).

        Parameters
        ----------
        adv_image : float32 tensor in [0, 1].
                    Shape (C, H, W) — single image, or
                    Shape (B, C, H, W) — batch.

        Returns
        -------
        clean     : float32 tensor in [0, 1], same shape as input.
        """
        batched = adv_image.dim() == 4
        if not batched:
            adv_image = adv_image.unsqueeze(0)

        results = []
        for b in range(adv_image.shape[0]):
            clean = self._purify_single(adv_image[b])
            results.append(clean)

        out = torch.stack(results, dim=0)
        if not batched:
            out = out.squeeze(0)
        return out
