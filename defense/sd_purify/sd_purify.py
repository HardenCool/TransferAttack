"""
SD-Purify: Stable Diffusion Adversarial Purification (Ours)
============================================================
SD-Purify uses a Stable Diffusion img2img pipeline to remove adversarial
perturbations from images.  A controlled amount of noise is injected into the
adversarial image (parameterised by ``strength``), then the SD denoiser
reconstructs a clean image guided by a quality-oriented text prompt and a
noise-suppressing negative prompt.

Advantages over DiffPure / WaveDM:
    1. Higher visual fidelity — trained on LAION-5B (5 billion diverse images).
    2. Better controllability — ``strength`` + text guidance balance
       purification depth against semantic preservation.
    3. Domain-agnostic — not restricted to ImageNet or CelebA-HQ distributions.
    4. Simple weight management — HuggingFace Hub downloads weights on first run.

Default backbone: ``runwayml/stable-diffusion-v1-5`` (auto-downloaded, ~4 GB).
To use a higher-quality model set ``model_id="stabilityai/stable-diffusion-2-1-base"``.

Requirements:
    pip install diffusers transformers accelerate Pillow torch

Usage:
    from defense.sd_purify.sd_purify import SDPurifier
    purifier = SDPurifier()             # lazy-loads model on first call
    clean_img = purifier.purify(adv_img)  # returns PIL.Image
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from PIL import Image


class SDPurifier:
    """
    Stable Diffusion img2img purifier ("Ours").

    Purifies adversarial examples by injecting controlled Gaussian noise and
    regenerating the image with a pre-trained SD model guided by a quality
    prompt.

    Args:
        model_id:            HuggingFace model ID.  Supported SD v1.x models
                             expect 512 × 512 input; SD v2.1 models expect
                             768 × 768 input.
        pipe:                A pre-loaded ``StableDiffusionImg2ImgPipeline``
                             instance.  When provided, *model_id* is ignored
                             and no download occurs.
        strength:            Noise injection level (0 = no change, 1 = full
                             redraw).  Values in [0.35, 0.45] strike a good
                             balance between purification and fidelity.
        num_inference_steps: Number of denoising steps.
        guidance_scale:      Classifier-free guidance scale.
        prompt:              Positive text prompt steering reconstruction.
        negative_prompt:     Negative prompt suppressing artifacts.
        device:              Torch device; ``None`` auto-detects CUDA.
    """

    _DEFAULT_PROMPT = (
        "high quality photograph, clean, natural, sharp details, no noise"
    )
    _DEFAULT_NEG_PROMPT = (
        "adversarial noise, artifacts, corrupted, distorted, blurry, "
        "low quality, ugly"
    )

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        pipe=None,
        strength: float = 0.40,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        prompt: str = _DEFAULT_PROMPT,
        negative_prompt: str = _DEFAULT_NEG_PROMPT,
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self._pipe = pipe  # may be None (lazy load) or pre-loaded

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
        print(f"[SD-Purify] Loading model: {self.model_id}")
        from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self._pipe.set_progress_bar_config(disable=True)
        print("[SD-Purify] Model loaded.")

    @staticmethod
    def _to_pil(image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Image.Image:
        """Convert any supported image format to RGB PIL Image."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 4:
                t = t.squeeze(0)
            arr = t.permute(1, 2, 0).numpy().astype(np.float32)
        elif isinstance(image, np.ndarray):
            arr = image.astype(np.float32)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        if arr.max() > 1.5:
            arr = arr / 255.0
        return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def purify(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
        strength: Optional[float] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
    ) -> Image.Image:
        """
        Purify an adversarial image using Stable Diffusion img2img.

        Args:
            image:           Adversarial image as PIL Image, torch.Tensor
                             [C, H, W] in [0, 1], or numpy array [H, W, C].
            strength:        Override instance ``strength`` for this call.
            prompt:          Override instance ``prompt`` for this call.
            negative_prompt: Override instance ``negative_prompt`` for this call.

        Returns:
            Purified PIL Image with the same spatial dimensions as the input.
        """
        self._load_model()

        img_pil = self._to_pil(image)
        original_size = img_pil.size  # (W, H)

        # Determine the resolution expected by the chosen model
        model_size = 768 if "2-1" in self.model_id else 512
        img_resized = img_pil.resize((model_size, model_size), Image.LANCZOS)

        with torch.no_grad():
            result = self._pipe(
                prompt=prompt if prompt is not None else self.prompt,
                negative_prompt=(
                    negative_prompt
                    if negative_prompt is not None
                    else self.negative_prompt
                ),
                image=img_resized,
                strength=strength if strength is not None else self.strength,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            ).images[0]

        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)

        return result
