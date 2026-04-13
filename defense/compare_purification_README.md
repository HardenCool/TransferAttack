# Adversarial Purification Comparison

Generates a side-by-side visual comparison of five purification methods:

| Column | Method |
|--------|--------|
| 1 | **Original** — clean reference image |
| 2 | **MOMUDIG Adv** — adversarial example |
| 3 | **DiffPure** — Guided Diffusion SDEdit (optional) |
| 4 | **WaveDM** — wavelet-domain soft-thresholding + optional DDPM |
| 5 | **Ours (DiT)** — DiT-XL/2 SDEdit purification |

---

## Quick Start

### 1. Install extra dependencies

```bash
pip install PyWavelets diffusers>=0.27.0 transformers>=4.38.0 accelerate>=0.27.0
```

### 2. Run comparison (WaveDM + DiT only, no checkpoint needed)

```bash
python defense/compare_purification.py \
    --orig_dir path/to/clean_images \
    --adv_dir  path/to/adv_images \
    --output_dir comparison_results \
    --t 150 --num_images 8
```

DiT weights (`facebook/DiT-XL-2-256`) are downloaded automatically from
HuggingFace on first run (~2 GB, cached in `~/.cache/huggingface/`).

### 3. Run with DiffPure

Download the Guided Diffusion checkpoint first:

```bash
mkdir -p defense/models
wget -P defense/models \
    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

Then add `--use_diffpure`:

```bash
python defense/compare_purification.py \
    --orig_dir path/to/clean_images \
    --adv_dir  path/to/adv_images \
    --output_dir comparison_results \
    --model_dir defense/models \
    --t 150 --num_images 8 --use_diffpure
```

---

## Output

```
comparison_results/
├── individual/
│   ├── image001_comparison.png   # 1×5 strip per image
│   ├── image002_comparison.png
│   └── ...
└── summary_grid.png              # All images in a single grid
```

---

## Module Overview

### `defense/wavedm/` — WaveDM Purifier

Wavelet-domain adversarial purification:
1. 2-D DWT (`db4` wavelet, 1 level) decomposes the image into LL / LH / HL / HH subbands.
2. Soft-thresholding (default threshold `0.06` ≈ 15/255) attenuates
   high-frequency adversarial noise concentrated in LH / HL / HH.
3. IDWT reconstructs the purified image.
4. Optionally followed by a light DDPM refinement pass (requires
   `256x256_diffusion_uncond.pt`).

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--wavedm_wavelet` | `db4` | Wavelet family (haar, db4, bior2.2, …) |
| `--wavedm_level` | `1` | Decomposition levels |
| `--wavedm_threshold` | `0.06` | Soft threshold (~2× adversarial ε) |
| `--wavedm_use_diffusion` | off | Enable DDPM refinement pass |
| `--wavedm_diffusion_t` | `50` | DDPM steps for refinement |

### `defense/dit_purify/` — DiT Purifier ("Ours")

SDEdit-style purification with [DiT-XL/2](https://github.com/facebookresearch/DiT):
1. Resize to 256×256, scale to [-1, 1].
2. Add DDPM forward noise at step `t`.
3. Denoise to step 0 using the DiT-XL/2 transformer (ImageNet class-conditional).
4. Resize back to original resolution.

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--t` | `150` | SDEdit noise steps |
| `--dit_model_id` | `facebook/DiT-XL-2-256` | HuggingFace model ID |
| `--dit_model_path` | — | Local model dir (overrides `--dit_model_id`) |
| `--dit_class_label` | `0` | ImageNet class label (0-999; 1000 = null) |
| `--dit_use_ddim` | on | DDIM scheduler (faster) |
| `--dit_ddim_steps` | `50` | DDIM reverse steps |

---

## Checkpoint Summary

| Method | Checkpoint | How to obtain |
|--------|-----------|---------------|
| DiffPure | `defense/models/256x256_diffusion_uncond.pt` | [OpenAI blob](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) |
| WaveDM (optional) | same as DiffPure | same as above |
| **Ours (DiT)** | `facebook/DiT-XL-2-256` | Auto-download via HuggingFace |
