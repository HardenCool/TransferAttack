"""
SDEdit-based adversarial purification using Stable Diffusion 2.1.

Core idea (SDEdit, Meng et al. 2022):
  1. Forward diffusion: add noise to the adversarial image up to timestep t.
  2. Reverse diffusion: denoise back to t=0.
  The adversarial perturbation is a high-frequency signal; moderate noise
  levels destroy it while preserving semantic content.

This module exposes a single class ``SDEditPurifier`` that can be used
both from ``purify_ours.py`` (batch mode) and from
``generate_comparison.py`` (interactive mode).

Requirements:
    pip install diffusers>=0.21 transformers accelerate
"""

import os
from typing import Union

import torch
import numpy as np
from PIL import Image


def _load_pipeline(model_path: str, device: torch.device):
    """Load the StableDiffusionImg2ImgPipeline from a local directory or HF Hub."""
    from diffusers import StableDiffusionImg2ImgPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    # Memory optimisation: attention slicing is free on accuracy
    if device.type == "cuda":
        pipe.enable_attention_slicing()
    return pipe


class SDEditPurifier:
    """
    Adversarial purification via SDEdit with Stable Diffusion 2.1.

    Parameters
    ----------
    model_path : str
        Path to a local directory containing SD2.1 weights (downloaded by
        ``download_weights.py``), or a HuggingFace repo ID such as
        ``'stabilityai/stable-diffusion-2-1-base'``.
    strength : float
        Noise strength in [0, 1].  Fraction of the total diffusion steps to
        run in the *forward* direction.  Higher values remove more adversarial
        noise but also destroy more semantic content.
        Recommended range: 0.35 – 0.45.
    num_inference_steps : int
        Total number of denoising steps.  50 gives good quality; 20 is faster.
    guidance_scale : float
        Classifier-free guidance scale.  1.0 disables guidance (pure
        unconditional generation), which avoids semantic drift.
    sd_resolution : int
        Internal processing resolution.  SD2.1-base was trained at 512×512.
    device : str or torch.device
        Device to run on.  Defaults to CUDA if available.
    seed : int or None
        Random seed for reproducibility.
    """

    SD_RESOLUTION = 512

    def __init__(
        self,
        model_path: str,
        strength: float = 0.40,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        sd_resolution: int = 512,
        device: Union[str, torch.device, None] = None,
        seed: int | None = 42,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(device)
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.sd_resolution = sd_resolution
        self.seed = seed

        print(f"[SDEditPurifier] Loading model from '{model_path}' on {self.device} ...")
        self.pipe = _load_pipeline(model_path, self.device)
        print("[SDEditPurifier] Model loaded.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def purify_pil(self, img: Image.Image) -> Image.Image:
        """
        Purify a single PIL image.

        Parameters
        ----------
        img : PIL.Image.Image
            Adversarial image in RGB mode (any resolution).

        Returns
        -------
        PIL.Image.Image
            Purified image at the *original* resolution.
        """
        original_size = img.size  # (W, H)
        resized = img.resize((self.sd_resolution, self.sd_resolution), Image.LANCZOS)

        generator = (
            torch.Generator(device=self.device).manual_seed(self.seed)
            if self.seed is not None
            else None
        )

        result = self.pipe(
            prompt="",
            image=resized,
            strength=self.strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
        ).images[0]

        # Restore original resolution
        if original_size != (self.sd_resolution, self.sd_resolution):
            result = result.resize(original_size, Image.LANCZOS)

        return result

    def purify_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Purify a batch of images represented as a float tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (N, C, H, W) with values in [0, 1].

        Returns
        -------
        torch.Tensor
            Purified tensor of the same shape and range [0, 1].
        """
        results = []
        for i in range(x.shape[0]):
            arr = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(arr, mode="RGB")
            purified = self.purify_pil(pil_img)
            out = torch.from_numpy(np.array(purified).astype(np.float32) / 255.0)
            results.append(out.permute(2, 0, 1))
        return torch.stack(results, dim=0)

    def purify_file(self, input_path: str, output_path: str):
        """Read an image file, purify it, and save the result."""
        img = Image.open(input_path).convert("RGB")
        purified = self.purify_pil(img)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        purified.save(output_path)
