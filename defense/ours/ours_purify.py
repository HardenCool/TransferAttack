"""
Ours: Multi-Scale Frequency-Guided DDPM Purification (FreqDDPM).

This method is the main contribution of the paper.  It improves upon WaveDM by
adding four key enhancements to the DDPM-based purification pipeline:

Key innovations over WaveDM / DiffPure:
  1. **Two-level wavelet pre-denoising** — adversarial noise is suppressed at
     two frequency scales (using a smooth db4 wavelet) before the diffusion
     step, yielding a cleaner input for the reverse pass.
  2. **Lower diffusion noise level** (t=150 vs. WaveDM's t=250) — the reverse
     pass restores local detail without overwriting global structure.
  3. **Post-diffusion LL + HF subband fusion** — after the reverse pass, the
     low-frequency (LL) subband from the wavelet-pre-denoised image is blended
     back at a high weight (alpha=0.70) to restore global structure (SSIM), and
     the high-frequency subbands are lightly blended (beta=0.20) so the DDPM
     generates natural-looking textures that improve LPIPS.
  4. **Perceptual polish pass** — a second very-short DDPM forward-reverse pass
     (t_refine=25) after the frequency fusion smooths any remaining wavelet
     reconstruction artefacts and injects natural fine-grain texture, which
     directly improves LPIPS without altering global structure.
  5. **Adaptive noise level** — t is scaled with the estimated l∞ perturbation
     magnitude so that lightly-perturbed images are minimally modified.

Pretrained weight (same file as DiffPure and WaveDM — no extra download):
  defense/models/256x256_diffusion_uncond.pt
  Download: bash defense/ours/download_weights.sh

Usage:
    from defense.ours.ours_purify import OursPurifier

    purifier = OursPurifier(model_dir='defense/models', device='cuda')
    clean_img = purifier.purify(adv_img)   # (C,H,W) or (B,C,H,W) in [0,1]
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# ── PyWavelets ────────────────────────────────────────────────────────────────
try:
    import pywt
except ImportError as e:
    raise ImportError(
        "PyWavelets is required.  Install with: pip install PyWavelets"
    ) from e

# ── Guided-diffusion backbone (shared with DiffPure and WaveDM) ───────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DIFFPURE_DIR = os.path.join(os.path.dirname(_THIS_DIR), "diffpure")
if _DIFFPURE_DIR not in sys.path:
    sys.path.insert(0, _DIFFPURE_DIR)

from guided_diffusion.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-level wavelet helpers
# ─────────────────────────────────────────────────────────────────────────────

_DWT_MODE = "periodization"
"""
DWT boundary mode.  'periodization' is the only pywt mode that is strictly
bijective: dwt2(N) → ceil(N/2) subbands, idwt2(ceil(N/2)) → N, regardless
of the wavelet filter length.  Using 'symmetric' with longer wavelets (e.g.
db4) on non-power-of-2 inputs can produce ambiguous subband sizes that cause
idwt2 to raise "coeffs must all be of equal size", which is what happens for
typical ImageNet images (224×224).
"""


def _dwt2(x: np.ndarray, wavelet: str = "db4") -> tuple:
    """Single-level 2-D DWT on a CHW float array."""
    LL_list, LH_list, HL_list, HH_list = [], [], [], []
    for c in range(x.shape[0]):
        LL_c, (LH_c, HL_c, HH_c) = pywt.dwt2(x[c], wavelet, mode=_DWT_MODE)
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
           HH: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """Single-level 2-D IDWT, returns CHW float array."""
    recon = []
    for c in range(LL.shape[0]):
        r = pywt.idwt2((LL[c], (LH[c], HL[c], HH[c])), wavelet, mode=_DWT_MODE)
        recon.append(r)
    return np.stack(recon, axis=0)


def _soft_threshold(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def _multilevel_wavelet_denoise(
    img: np.ndarray,
    wavelet: str = "db4",
    threshold_l1: float = 0.020,
    threshold_l2: float = 0.010,
) -> np.ndarray:
    """
    Two-level wavelet soft-thresholding denoising on a CHW [0,1] array.

    Level 1: DWT on the full image, threshold HF subbands at ``threshold_l1``.
    Level 2: DWT on the level-1 LL subband, threshold at ``threshold_l2``.
    Both levels are reconstructed via IDWT.
    """
    # Level-1 decomposition
    LL1, LH1, HL1, HH1 = _dwt2(img, wavelet)
    LH1 = _soft_threshold(LH1, threshold_l1)
    HL1 = _soft_threshold(HL1, threshold_l1)
    HH1 = _soft_threshold(HH1, threshold_l1)

    # Level-2 decomposition on the LL1 subband
    LL2, LH2, HL2, HH2 = _dwt2(LL1, wavelet)
    LH2 = _soft_threshold(LH2, threshold_l2)
    HL2 = _soft_threshold(HL2, threshold_l2)
    HH2 = _soft_threshold(HH2, threshold_l2)

    # Reconstruct LL1 from level-2 coefficients.
    # Crop to exactly match LH1/HL1/HH1 shape — 'periodization' mode keeps
    # the round-trip exact, but we guard against any off-by-one edge case.
    LL1_recon = _idwt2(LL2, LH2, HL2, HH2, wavelet)
    LL1_recon = LL1_recon[:, :LH1.shape[1], :LH1.shape[2]]

    # Reconstruct the full image from level-1 coefficients (using reconstructed LL1)
    recon = _idwt2(LL1_recon, LH1, HL1, HH1, wavelet)
    return np.clip(recon, 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-Guided DDPM Purification (Ours)
# ─────────────────────────────────────────────────────────────────────────────

class OursPurifier:
    """
    Multi-Scale Frequency-Guided DDPM Purification (FreqDDPM — 'Ours').

    Parameters
    ----------
    model_dir         : directory containing ``256x256_diffusion_uncond.pt``.
                        Default: ``"defense/models"``.
    t                 : base DDPM noise level.  Lower values are less destructive.
                        Default: 150 (vs. WaveDM's 250).
    wavelet           : mother wavelet for frequency analysis.  'db4' (default)
                        produces smooth reconstructions without Haar blocking
                        artefacts, which directly benefits LPIPS.
    threshold_l1      : level-1 wavelet soft-threshold applied to HF subbands.
                        Default: 0.020.
    threshold_l2      : level-2 wavelet soft-threshold applied to HF subbands of
                        the LL subband.  Default: 0.010.
    ll_blend_alpha    : fraction of the LL subband drawn from the wavelet-
                        pre-denoised image (vs. the DDPM output) after the
                        reverse pass.  Higher → better SSIM/PSNR.  Default: 0.70.
    hf_blend_alpha    : fraction of the HF subbands (LH/HL/HH) drawn from the
                        wavelet-pre-denoised image after the reverse pass.
                        Keeping this low (0.20) lets the DDPM generate natural
                        fine-grain textures, which improves LPIPS.  Default: 0.20.
    t_refine          : number of timesteps for the optional perceptual polish
                        pass — a second short DDPM forward-reverse pass applied
                        after the frequency fusion step.  Removes wavelet
                        reconstruction artefacts and injects natural texture.
                        Set to 0 to disable.  Default: 25.
    adaptive          : if True, scale ``t`` with the estimated perturbation
                        magnitude so that clean images are barely modified.
                        Default: True.
    device            : torch device string.  Default: 'cuda'.
    """

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
        model_dir: str = "defense/models",
        t: int = 150,
        wavelet: str = "db4",
        threshold_l1: float = 0.020,
        threshold_l2: float = 0.010,
        ll_blend_alpha: float = 0.70,
        hf_blend_alpha: float = 0.20,
        t_refine: int = 25,
        adaptive: bool = True,
        device: str = "cuda",
    ):
        self.t = t
        self.wavelet = wavelet
        self.threshold_l1 = threshold_l1
        self.threshold_l2 = threshold_l2
        self.ll_blend_alpha = ll_blend_alpha
        self.hf_blend_alpha = hf_blend_alpha
        self.t_refine = t_refine
        self.adaptive = adaptive
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        weights_path = os.path.join(model_dir, "256x256_diffusion_uncond.pt")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"OursPurifier requires the pretrained weight file:\n  {weights_path}\n"
                "Download it with:\n"
                "  bash defense/ours/download_weights.sh"
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
        betas = torch.from_numpy(diffusion.betas).float().to(self.device)
        self._alphas_cumprod = (1.0 - betas).cumprod(dim=0)
        print("[Ours] Model ready.")

    # ------------------------------------------------------------------ #
    # Adaptive noise-level estimation
    # ------------------------------------------------------------------ #

    _BASE_EPS: float = 0.016   # ε = 4/255
    _T_SCALE: float  = 0.20    # +20 % t for each additional ε above _BASE_EPS

    def _estimate_t(self, adv: torch.Tensor, wt_clean: torch.Tensor) -> int:
        """Scale t with the estimated l∞ perturbation magnitude."""
        diff = (adv.float() - wt_clean.float()).abs().max().item()
        scale = 1.0 + max(0.0, diff - self._BASE_EPS) / self._BASE_EPS * self._T_SCALE
        return max(1, int(min(round(self.t * scale), 300)))

    # ------------------------------------------------------------------ #
    # DDPM forward-reverse helper
    # ------------------------------------------------------------------ #

    def _ddpm_pass(self, x0: torch.Tensor, t_steps: int, B: int) -> torch.Tensor:
        """
        Run a DDPM forward-reverse pass on ``x0`` at noise level ``t_steps``.

        ``x0`` is a (B, C, H, W) tensor in [-1, 1].
        Returns a denoised (B, C, H, W) tensor in [-1, 1].
        """
        a = self._alphas_cumprod[t_steps - 1]
        noise = torch.randn_like(x0)
        x = x0 * a.sqrt() + noise * (1.0 - a).sqrt()
        for i in reversed(range(t_steps)):
            t_batch = torch.tensor([i] * B, device=self.device)
            x = self.diffusion.p_sample(
                self.model, x, t_batch,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
            )["sample"]
        return x

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

        B, C, H, W = adv_image.shape

        # ── Step 1: Multi-level wavelet pre-denoising ────────────────────
        wt_clean_list = []
        for b in range(B):
            arr = adv_image[b].cpu().numpy()
            denoised = _multilevel_wavelet_denoise(
                arr, self.wavelet, self.threshold_l1, self.threshold_l2
            )
            wt_clean_list.append(torch.from_numpy(denoised))
        wt_clean = torch.stack(wt_clean_list, dim=0).to(self.device)  # (B,C,H,W) [0,1]

        # ── Step 2: Adaptive noise level ─────────────────────────────────
        # Use the maximum t across the batch so the most-perturbed image is
        # fully denoised while lightly-perturbed images benefit from the same
        # conservative pass (over-denoising is mild at low t values).
        if self.adaptive:
            t_use = max(
                self._estimate_t(adv_image[b], wt_clean_list[b])
                for b in range(B)
            )
        else:
            t_use = self.t

        # ── Step 3: Resize to 256×256 if needed ──────────────────────────
        orig_size = (H, W)
        x0 = wt_clean
        if H != 256 or W != 256:
            x0 = F.interpolate(x0, size=(256, 256), mode="bilinear", align_corners=False)

        # ── Step 4: Main DDPM forward-reverse pass ────────────────────────
        x0_scaled = (x0 - 0.5) * 2.0
        x = self._ddpm_pass(x0_scaled, t_use, B)

        # ── Step 5: Rescale back to [0,1] ────────────────────────────────
        x = ((x + 1.0) * 0.5).clamp(0.0, 1.0)
        if orig_size != (256, 256):
            x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)

        # ── Step 6: Post-diffusion multi-scale frequency fusion ──────────
        # Blend LL (global structure) at high alpha=0.70 from the wavelet-
        # pre-denoised image to recover SSIM/PSNR.  Blend HF (edges/texture)
        # at low alpha=0.20 to avoid injecting adversarial residuals from the
        # thresholded subbands — the DDPM contribution dominates HF texture,
        # yielding more natural-looking output and better LPIPS.
        fused_list = []
        wt_clean_cpu = wt_clean.cpu()
        for b in range(B):
            out_arr = x[b].cpu().numpy()           # CHW [0,1]
            ref_arr = wt_clean_cpu[b].numpy()      # CHW [0,1]

            LL_out, LH_out, HL_out, HH_out = _dwt2(out_arr, self.wavelet)
            LL_ref, LH_ref, HL_ref, HH_ref = _dwt2(ref_arr, self.wavelet)

            LL_fused = self.ll_blend_alpha * LL_ref + (1.0 - self.ll_blend_alpha) * LL_out
            LH_fused = self.hf_blend_alpha * LH_ref + (1.0 - self.hf_blend_alpha) * LH_out
            HL_fused = self.hf_blend_alpha * HL_ref + (1.0 - self.hf_blend_alpha) * HL_out
            HH_fused = self.hf_blend_alpha * HH_ref + (1.0 - self.hf_blend_alpha) * HH_out

            fused_arr = _idwt2(LL_fused, LH_fused, HL_fused, HH_fused, self.wavelet)
            fused_arr = np.clip(fused_arr, 0.0, 1.0).astype(np.float32)
            fused_list.append(torch.from_numpy(fused_arr))

        fused = torch.stack(fused_list, dim=0).to(self.device)  # (B,C,H,W) [0,1]

        # ── Step 7: Perceptual polish pass (optional) ─────────────────────
        # A second very-short DDPM forward-reverse pass removes wavelet
        # reconstruction artefacts and injects natural fine-grain texture.
        # At t_refine=25 the forward pass adds only a tiny amount of noise
        # (alpha_24 ≈ 0.97), so global structure is untouched while
        # high-frequency artefacts are smoothed and replaced naturally.
        if self.t_refine > 0:
            fused_256 = fused
            if orig_size != (256, 256):
                fused_256 = F.interpolate(
                    fused, size=(256, 256), mode="bilinear", align_corners=False
                )
            fused_scaled = (fused_256 - 0.5) * 2.0
            x_polished = self._ddpm_pass(fused_scaled, self.t_refine, B)
            x_polished = ((x_polished + 1.0) * 0.5).clamp(0.0, 1.0)
            if orig_size != (256, 256):
                x_polished = F.interpolate(
                    x_polished, size=orig_size, mode="bilinear", align_corners=False
                )
            fused = x_polished

        if not batched:
            fused = fused.squeeze(0)
        return fused.cpu().float()
