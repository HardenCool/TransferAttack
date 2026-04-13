"""
WaveDM Adversarial Purification
================================
Wavelet-domain adversarial purification inspired by WaveDiff / WaveDM.

Strategy
--------
1. Apply 2-D Discrete Wavelet Transform (DWT) to decompose the input image
   into one low-frequency subband (LL) and three high-frequency detail
   subbands per level (LH, HL, HH).
2. Apply soft-thresholding to the high-frequency subbands to attenuate
   adversarial perturbations, which typically reside in fine-grained
   high-frequency structures.
3. Optionally follow with a light DDPM-based denoising pass on the
   reconstructed image for improved visual quality (requires the Guided
   Diffusion checkpoint ``256x256_diffusion_uncond.pt``).
4. Reconstruct via Inverse DWT.

Dependencies
------------
    pip install PyWavelets

Optional (for diffusion refinement)::
    pip install torch torchvision
    # plus the OpenAI Guided Diffusion checkpoint
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


class WaveDMPurifier:
    """Wavelet-domain adversarial purification (WaveDM style).

    Parameters
    ----------
    wavelet : str
        Wavelet family to use (default ``'db4'``).  ``'haar'`` is the
        fastest; ``'db4'`` / ``'bior2.2'`` give smoother reconstructions.
    level : int
        Decomposition levels (default ``1``).  Higher values decompose
        coarser frequencies but increase blurring for the same threshold.
    threshold : float
        Soft-threshold applied to each high-frequency coefficient.  A
        value around 2×(adversarial ε) works well in practice.
        Default ``0.06`` (~15/255).
    use_diffusion : bool
        Whether to apply a light DDPM-refinement pass after wavelet
        reconstruction (default ``False``).  Requires ``model_dir`` to
        contain ``256x256_diffusion_uncond.pt``.
    diffusion_t : int
        Number of DDPM forward-noise steps for the optional refinement
        (default ``50``).  Lower values preserve structure better.
    model_dir : str
        Directory containing ``256x256_diffusion_uncond.pt`` (only used
        when ``use_diffusion=True``).  Defaults to
        ``<this file's directory>/../models``.
    device : torch.device or None
        Inference device.  Auto-detected when ``None``.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        level: int = 1,
        threshold: float = 0.06,
        use_diffusion: bool = False,
        diffusion_t: int = 50,
        model_dir: str = None,
        device=None,
    ):
        if not PYWT_AVAILABLE:
            raise ImportError(
                "PyWavelets is required for WaveDMPurifier.\n"
                "Install it with:  pip install PyWavelets"
            )

        self.wavelet = wavelet
        self.level = level
        self.threshold = threshold
        self.use_diffusion = use_diffusion
        self.diffusion_t = diffusion_t

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.model_dir = os.path.abspath(model_dir)

        self._ddpm_model = None
        self._ddpm_diffusion = None
        self._ddpm_alphas_cumprod = None  # pre-computed ᾱ_t values

        if use_diffusion:
            self._load_ddpm()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_ddpm(self):
        """Load OpenAI Guided Diffusion model for optional refinement."""
        guided_dir = os.path.join(os.path.dirname(__file__), "..", "diffpure")
        guided_dir = os.path.abspath(guided_dir)
        if guided_dir not in sys.path:
            sys.path.insert(0, guided_dir)

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

        weight_path = os.path.join(self.model_dir, "256x256_diffusion_uncond.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Guided Diffusion checkpoint not found at:\n  {weight_path}\n"
                "Download it from:\n"
                "  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
                "256x256_diffusion_uncond.pt"
            )

        model, diffusion = create_model_and_diffusion(**cfg)
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        model.requires_grad_(False).eval()
        if cfg["use_fp16"]:
            model.convert_to_fp16()
        model.to(self.device)

        self._ddpm_model = model
        self._ddpm_diffusion = diffusion
        # Pre-compute ᾱ_t = cumprod(1 - β) once at load time
        betas = torch.from_numpy(diffusion.betas).float()
        self._ddpm_alphas_cumprod = (1 - betas).cumprod(dim=0).to(self.device)
        print("WaveDM: DDPM refinement model loaded.")

    def _wavelet_purify_np(self, img_np: np.ndarray) -> np.ndarray:
        """Purify a single-channel 2-D float32 array in [0, 1]."""
        coeffs = pywt.wavedec2(img_np, self.wavelet, level=self.level)
        # coeffs[0]   = LL approximation subband
        # coeffs[1..] = (LH, HL, HH) detail tuples per level
        new_coeffs = [coeffs[0]]
        for detail_tuple in coeffs[1:]:
            lh, hl, hh = detail_tuple
            lh = pywt.threshold(lh, self.threshold, mode="soft")
            hl = pywt.threshold(hl, self.threshold, mode="soft")
            hh = pywt.threshold(hh, self.threshold, mode="soft")
            new_coeffs.append((lh, hl, hh))

        rec = pywt.waverec2(new_coeffs, self.wavelet)
        # pywt may pad by 1 pixel; crop back to original size
        h, w = img_np.shape
        rec = rec[:h, :w]
        return np.clip(rec, 0.0, 1.0).astype(np.float32)

    def _ddpm_refine(self, x: torch.Tensor) -> torch.Tensor:
        """Light DDPM denoising pass.

        x : [B, C, H, W] in [0, 1]
        Returns purified tensor in [0, 1].
        """
        orig_size = x.shape[-2:]
        if orig_size != (256, 256):
            x_256 = F.interpolate(
                x, size=(256, 256), mode="bilinear", align_corners=False
            )
        else:
            x_256 = x

        x_scaled = (x_256 * 2 - 1).to(self.device)  # [-1, 1]
        B = x_scaled.shape[0]
        t = self.diffusion_t

        ac = self._ddpm_alphas_cumprod[t - 1]  # ᾱ_t (pre-computed)
        noise = torch.randn_like(x_scaled)
        x_noisy = x_scaled * ac.sqrt() + noise * (1.0 - ac).sqrt()

        with torch.no_grad():
            for i in reversed(range(t)):
                t_batch = torch.tensor([i] * B, device=self.device)
                x_noisy = self._ddpm_diffusion.p_sample(
                    self._ddpm_model,
                    x_noisy,
                    t_batch,
                    clip_denoised=True,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                )["sample"]

        x_out = (x_noisy + 1) * 0.5  # [0, 1]
        if orig_size != (256, 256):
            x_out = F.interpolate(
                x_out, size=orig_size, mode="bilinear", align_corners=False
            )
        return x_out.cpu()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def purify(self, x: torch.Tensor) -> torch.Tensor:
        """Purify a batch of adversarial images.

        Parameters
        ----------
        x : torch.Tensor
            Adversarial images of shape ``[B, C, H, W]`` with pixel values
            in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Purified images of the same shape and range.
        """
        x_np = x.detach().cpu().numpy()  # [B, C, H, W]
        B, C, H, W = x_np.shape

        purified = np.empty_like(x_np)
        for b in range(B):
            for c in range(C):
                purified[b, c] = self._wavelet_purify_np(x_np[b, c])

        x_purified = torch.from_numpy(purified).float()

        if self.use_diffusion:
            x_purified = self._ddpm_refine(x_purified)

        return x_purified.to(x.device)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.purify(x)
