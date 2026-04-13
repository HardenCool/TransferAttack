"""
WaveDM Purification Module
===========================
Principle (WaveDM-style):
  1. Apply 2D Discrete Wavelet Transform (DWT) to decompose the image into
     a low-frequency approximation subband (LL) and three high-frequency
     detail subbands (LH, HL, HH).
  2. Remove adversarial perturbations from the high-frequency subbands via
     soft-thresholding (wavelet shrinkage).
  3. Purify the low-frequency subband using a Stable Diffusion img2img
     pipeline (add a controlled amount of noise then denoise).
  4. Reconstruct the purified image via Inverse DWT (IDWT).

This approach targets adversarial noise at two levels:
  - High-frequency level: direct coefficient thresholding
  - Semantic level: diffusion-based regeneration of the approximation band

Requirements:
    pip install PyWavelets diffusers transformers accelerate Pillow torch

Usage:
    from defense.wavedm.wavedm_purify import WaveDMPurifier
    purifier = WaveDMPurifier()          # lazy-loads model on first call
    clean_img = purifier.purify(adv_img) # returns PIL.Image
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from PIL import Image


class WaveDMPurifier:
    """
    Wavelet-Domain Diffusion Model purifier.

    Decomposes an adversarial image in the wavelet domain, removes
    adversarial noise from high-frequency subbands via soft-thresholding,
    and purifies the low-frequency (LL) subband using a Stable Diffusion
    img2img pipeline.

    Args:
        model_id:           HuggingFace model ID for the SD img2img backbone.
        pipe:               A pre-loaded ``StableDiffusionImg2ImgPipeline``
                            instance.  When provided, *model_id* is ignored
                            and no model is downloaded.
        wavelet:            Wavelet family passed to PyWavelets (e.g. 'haar',
                            'db2', 'bior1.3').
        hf_threshold:       Soft-threshold value for high-frequency wavelet
                            coefficients.  Larger values remove more noise
                            (and more texture).
        ll_strength:        Diffusion strength applied to the LL subband
                            (0 = no change, 1 = full redraw).
        num_inference_steps: Number of denoising steps for the SD pipeline.
        device:             Torch device string; ``None`` auto-detects CUDA.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        pipe=None,
        wavelet: str = "haar",
        hf_threshold: float = 0.05,
        ll_strength: float = 0.30,
        num_inference_steps: int = 50,
        device: Optional[str] = None,
    ) -> None:
        try:
            import pywt  # type: ignore
            self._pywt = pywt
        except ImportError as exc:
            raise ImportError(
                "PyWavelets is required for WaveDMPurifier. "
                "Install with:  pip install PyWavelets"
            ) from exc

        self.model_id = model_id
        self.wavelet = wavelet
        self.hf_threshold = hf_threshold
        self.ll_strength = ll_strength
        self.num_inference_steps = num_inference_steps
        self._pipe = pipe  # may be None (lazy load) or a pre-loaded pipe

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the Stable Diffusion img2img pipeline."""
        if self._pipe is not None:
            return
        print(f"[WaveDM] Loading backbone model: {self.model_id}")
        from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        print("[WaveDM] Model loaded.")

    @staticmethod
    def _to_numpy(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert any supported image format to float32 numpy [H, W, 3] in [0, 1]."""
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB")).astype(np.float32) / 255.0
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 4:
                t = t.squeeze(0)
            arr = t.permute(1, 2, 0).numpy().astype(np.float32)
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        return arr / 255.0 if arr.max() > 1.5 else arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def purify(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
    ) -> Image.Image:
        """
        Purify an adversarial image using wavelet-domain diffusion.

        Args:
            image: Adversarial image as PIL Image, torch.Tensor [C, H, W]
                   in [0, 1], or numpy array [H, W, C] in [0, 1] or [0, 255].

        Returns:
            Purified PIL Image with the same spatial dimensions as the input.
        """
        self._load_model()

        img_np = np.clip(self._to_numpy(image), 0.0, 1.0)
        original_h, original_w = img_np.shape[:2]

        # ---- Step 1: Per-channel 2-D DWT --------------------------------
        ll_bands: list[np.ndarray] = []
        hf_bands: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for c in range(3):
            cA, (cH, cV, cD) = self._pywt.dwt2(img_np[:, :, c], self.wavelet)
            ll_bands.append(cA)
            hf_bands.append((cH, cV, cD))

        ll = np.stack(ll_bands, axis=2)  # [H/2, W/2, 3]

        # ---- Step 2: Soft-threshold high-frequency subbands -------------
        thr = self.hf_threshold
        purified_hf: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for cH, cV, cD in hf_bands:
            purified_hf.append((
                np.sign(cH) * np.maximum(np.abs(cH) - thr, 0.0),
                np.sign(cV) * np.maximum(np.abs(cV) - thr, 0.0),
                np.sign(cD) * np.maximum(np.abs(cD) - thr, 0.0),
            ))

        # ---- Step 3: SD img2img on the LL subband -----------------------
        ll_h, ll_w = ll.shape[:2]
        ll_pil = Image.fromarray(
            (np.clip(ll, 0.0, 1.0) * 255).astype(np.uint8)
        ).resize((512, 512), Image.LANCZOS)

        with torch.no_grad():
            purified_ll_pil = self._pipe(
                prompt="high quality photograph, clean, sharp, natural",
                negative_prompt="noise, artifacts, adversarial, blurry",
                image=ll_pil,
                strength=self.ll_strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=7.5,
            ).images[0]

        purified_ll = (
            np.array(
                purified_ll_pil.resize((ll_w, ll_h), Image.LANCZOS)
            ).astype(np.float32)
            / 255.0
        )

        # ---- Step 4: Per-channel IDWT reconstruction --------------------
        out_channels: list[np.ndarray] = []
        for c in range(3):
            out_channels.append(
                self._pywt.idwt2(
                    (purified_ll[:, :, c], purified_hf[c]),
                    self.wavelet,
                )
            )

        result = np.clip(np.stack(out_channels, axis=2), 0.0, 1.0)
        result_pil = Image.fromarray((result * 255).astype(np.uint8))

        if result_pil.size != (original_w, original_h):
            result_pil = result_pil.resize(
                (original_w, original_h), Image.LANCZOS
            )

        return result_pil
