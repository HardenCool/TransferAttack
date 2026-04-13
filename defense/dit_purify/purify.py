"""
DiT Adversarial Purification ("Ours")
=======================================
SDEdit-style adversarial purification using a Diffusion Transformer (DiT).

Model
-----
    facebook/DiT-XL-2-256  (ImageNet 256×256, class-conditional)
    https://huggingface.co/facebook/DiT-XL-2-256

The DiT-XL/2 transformer architecture achieves state-of-the-art FID on
ImageNet generation and delivers sharper, more semantically-coherent
reconstructions than U-Net DDPM baselines when used for SDEdit-style
purification.

Algorithm (SDEdit with DiT)
----------------------------
1. Resize input image to 256×256 and scale to [-1, 1].
2. Corrupt with DDPM forward diffusion for ``t`` steps.
3. Denoise back to step 0 using the DiT denoiser (class-conditional, with
   an optional class label).
4. Rescale to [0, 1] and resize to the original resolution.

Dependencies
------------
    pip install diffusers transformers accelerate

Weights are downloaded automatically to the HuggingFace cache on first
run, or can be supplied via ``--model_path`` pointing to a local
directory in the standard ``diffusers`` snapshot layout.
"""

import os
import torch
import torch.nn.functional as F


class DiTPurifier:
    """SDEdit adversarial purification with DiT-XL/2.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID (default ``'facebook/DiT-XL-2-256'``).
        Ignored when ``model_path`` is set.
    model_path : str or None
        Path to a local model directory (``diffusers`` snapshot layout).
        Takes precedence over ``model_id`` when set.
    t : int
        Number of forward-diffusion steps used for corruption (SDEdit
        noise scale).  Values in [100, 200] offer a good trade-off
        between purification strength and image fidelity.
        Default ``150``.
    class_label : int
        ImageNet class label to condition the denoiser on (0–999).
        Use ``1000`` to trigger the null / unconditional token when the
        model was trained with classifier-free guidance.
        Default ``0`` (tench).
    use_ddim : bool
        Use DDIM instead of DDPM for faster denoising (default ``True``).
        Requires fewer reverse steps while maintaining quality.
    ddim_steps : int
        Number of DDIM denoising steps (default ``50``).  Only used when
        ``use_ddim=True``.
    device : torch.device or None
        Inference device.  Auto-detected when ``None``.
    """

    def __init__(
        self,
        model_id: str = "facebook/DiT-XL-2-256",
        model_path: str = None,
        t: int = 150,
        class_label: int = 0,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        device=None,
    ):
        self.t = t
        self.class_label = class_label
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self._load_model(model_id, model_path)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_model(self, model_id: str, model_path: str):
        """Download / load the DiT model and configure the scheduler."""
        try:
            from diffusers import DiTPipeline, DDPMScheduler, DDIMScheduler
        except ImportError as exc:
            raise ImportError(
                "The 'diffusers' package is required for DiTPurifier.\n"
                "Install it with:  pip install diffusers transformers accelerate"
            ) from exc

        source = model_path if model_path else model_id
        print(f"DiTPurifier: loading model from '{source}' …")

        # Load the full pipeline; we only keep the transformer + scheduler
        pipe = DiTPipeline.from_pretrained(source, torch_dtype=torch.float32)
        self.transformer = pipe.transformer.eval().to(self.device)
        self.transformer.requires_grad_(False)

        # Build forward-noise (DDPM) scheduler — always 1000-step linear
        # schedule to match the training configuration.
        ddpm_cfg = pipe.scheduler.config
        self._ddpm_sched = DDPMScheduler.from_config(ddpm_cfg)
        self._ddpm_sched.set_timesteps(1000)
        self._alphas_cumprod = self._ddpm_sched.alphas_cumprod.to(self.device)

        if self.use_ddim:
            self._denoise_sched = DDIMScheduler.from_config(ddpm_cfg)
            self._denoise_sched.set_timesteps(self.ddim_steps)
        else:
            self._denoise_sched = DDPMScheduler.from_config(ddpm_cfg)
            self._denoise_sched.set_timesteps(1000)

        print("DiTPurifier: model loaded successfully.")

    # ------------------------------------------------------------------
    # Purification
    # ------------------------------------------------------------------

    def _forward_diffuse(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Add DDPM noise at step ``t`` (forward process).

        x : [B, C, H, W] in [-1, 1]
        Returns noisy tensor of same shape.
        """
        ac = self._alphas_cumprod[t - 1]  # ᾱ_t
        noise = torch.randn_like(x)
        return ac.sqrt() * x + (1.0 - ac).sqrt() * noise

    def _reverse_denoise(self, x_noisy: torch.Tensor, class_label: int) -> torch.Tensor:
        """Run the reverse denoising loop from step ``t`` to 0.

        x_noisy : [B, C, H, W] in [-1, 1]
        Returns denoised tensor of same shape.
        """
        B = x_noisy.shape[0]
        class_labels = torch.tensor(
            [class_label] * B, dtype=torch.long, device=self.device
        )

        sched = self._denoise_sched
        # Only denoise for timesteps ≤ self.t (SDEdit: start from t, not T)
        active_ts = [ts for ts in sched.timesteps if int(ts) <= self.t]

        curr = x_noisy
        with torch.no_grad():
            for ts in active_ts:
                ts_batch = torch.tensor([ts] * B, dtype=torch.long, device=self.device)
                # DiT transformer: (hidden_states, timestep, class_labels) → sample
                model_out = self.transformer(
                    curr,
                    timestep=ts_batch,
                    class_labels=class_labels,
                ).sample
                curr = sched.step(model_out, int(ts), curr).prev_sample

        return curr

    def purify(self, x: torch.Tensor, class_label: int = None) -> torch.Tensor:
        """Purify a batch of adversarial images.

        Parameters
        ----------
        x : torch.Tensor
            Adversarial images of shape ``[B, C, H, W]`` with pixel values
            in ``[0, 1]``.
        class_label : int or None
            Override the instance-level class label for this call.

        Returns
        -------
        torch.Tensor
            Purified images of the same shape, values in ``[0, 1]``.
            Returned on CPU.
        """
        label = class_label if class_label is not None else self.class_label

        orig_size = x.shape[-2:]  # (H, W)

        # Resize to 256×256 as required by DiT-XL-2-256
        if orig_size != (256, 256):
            x_256 = F.interpolate(
                x, size=(256, 256), mode="bilinear", align_corners=False
            )
        else:
            x_256 = x.clone()

        # Scale [0, 1] → [-1, 1] and send to device
        x_scaled = (x_256 * 2.0 - 1.0).to(self.device)

        # Forward diffusion (add noise at timestep t)
        x_noisy = self._forward_diffuse(x_scaled, self.t)

        # Reverse denoising
        x_denoised = self._reverse_denoise(x_noisy, label)

        # Scale back to [0, 1]
        x_out = (x_denoised + 1.0) * 0.5
        x_out = x_out.clamp(0.0, 1.0)

        # Resize back to original resolution if needed
        if orig_size != (256, 256):
            x_out = F.interpolate(
                x_out, size=orig_size, mode="bilinear", align_corners=False
            )

        return x_out.cpu()

    def __call__(self, x: torch.Tensor, class_label: int = None) -> torch.Tensor:
        return self.purify(x, class_label=class_label)
