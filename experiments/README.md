# Experiments — High-Transferability Attack Robustness Benchmark

This directory contains a self-contained pipeline for reproducing the
"robustness under high-transferability attacks" experiment described in
Section 4.5 of the paper.

---

## Directory layout

```
experiments/
├── run_attacks.sh        # Generate DIM / SGM / MIG / OPS / MUMODIG adversarial examples
├── eval_clean.sh         # Baseline clean-accuracy cross-model eval
├── eval_at.sh            # AT defense evaluation
├── eval_diffpure.sh      # DiffPure defense evaluation
├── eval_nrp.sh           # NRP defense evaluation (optional)
├── collect_results.py    # Parse logs → RA % → Markdown table
└── results/              # Auto-created by eval scripts
    ├── at/
    ├── diffpure/
    └── nrp/
```

Adversarial examples are written to:
```
adv_data/
├── dim/resnet18/
├── sgm/resnet18/
├── mig/resnet18/
├── ops/resnet18/       ← generated natively by run_attacks.sh
└── mumodig/resnet18/   ← generated natively by run_attacks.sh
```

---

## Step-by-step reproduction

### 1. Prerequisites

**Python environment** (all packages already in `requirements.txt`):
```bash
pip install -r requirements.txt
```

**Data** — ImageNet 1 000-image subset used by the paper:
- Google Drive: https://drive.google.com/file/d/1d-_PKYi3MBDPtJV4rfMCCtmsE0oWX7ZB/view
- Hugging Face: https://huggingface.co/datasets/Trustworthy-AI-Group/TransferAttack/blob/main/data.zip

Unzip so you have:
```
/path/to/data/
├── images/
│   ├── ILSVRC2012_val_00000293.JPEG
│   └── ...
└── labels.csv
```

**Defense model weights** — download `defense_model.zip` from:
- Google Drive: https://drive.google.com/drive/folders/1NfSjLzc-MtkYHLumcKYs6OqC2X_zWy3g
- Hugging Face: https://huggingface.co/Trustworthy-AI-Group/TransferAttack/blob/main/defense_model.zip

Extract into `defense/models/`:
```
defense/models/
├── imagenet_model_weights_4px.pth.tar   ← AT
├── NRP.pth                              ← NRP
└── rs_imagenet/                         ← RS (optional)
```

**DiffPure diffusion checkpoint** — download from OpenAI:
```bash
wget -P defense/models/ \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
```

> The DiffPure runner (`defense/diffpure/runners/diffpure_guided.py` and
> `diffpure_sde.py`) looks for this file at `defense/models/256x256_diffusion_uncond.pt`.

---

### 2. Generate adversarial examples (DIM / SGM / MIG / OPS / MUMODIG)

```bash
bash experiments/run_attacks.sh /path/to/data 0
# GPU_ID=0; change as needed
```

This writes adversarial images into
`adv_data/{dim,sgm,mig,ops,mumodig}/resnet18/`.

> **OPS** requires `kornia` for random rotation (already in `requirements.txt`
> for MUMODIG; install with `pip install kornia` if missing).
> **MUMODIG** also requires `kornia` for its `RandomRotation` transform.

### 3. Evaluate clean accuracy

```bash
bash experiments/eval_clean.sh /path/to/data 0
```

---

### 4. Evaluate AT defense

```bash
bash experiments/eval_at.sh /path/to/data 0
```

Results are written to `experiments/results/at/{dim,sgm,mig,ops,mumodig}.txt`.

---

### 5. Evaluate DiffPure defense

```bash
bash experiments/eval_diffpure.sh /path/to/data 0
```

> ⚠️ DiffPure is compute-intensive (~1 h per attack on a single A100/4090 for
> 1 000 images at `--adv_batch_size 4`).  Reduce `--num_sub` for quick tests.

Results (with `ASR:XX.XX%` lines) are written to
`experiments/results/diffpure/{attack}.log`.

---

### 6. Evaluate NRP defense (optional)

```bash
bash experiments/eval_nrp.sh /path/to/data 0
```

---

### 7. Collect results and print comparison table

```bash
python experiments/collect_results.py --results_root experiments/results
```

Example output:

```
## 鲁棒准确率对比 (RA %) — 高迁移攻击

| 防御方法 | DIM | SGM | MIG | OPS | MUMODIG | 平均鲁棒性 |
|---|---|---|---|---|---|---|
| AT (Adv. Training) | 61.5 | 56.4 | 52.8 | 50.5 | -- | 55.3 |
| DiffPure           | 72.8 | 68.1 | 64.2 | 61.8 | -- | 66.7 |
| NRP                | 70.1 | 65.3 | 61.0 | 59.2 | -- | 63.9 |
| **Ours**           | **Ours** | **Ours** | **Ours** | **Ours** | **Ours** | **Ours** |
```

Fill in the **Ours** column after running your own defense method.

---

## Attack details

| Attack | Category | Key idea | ε | Iterations | Surrogate |
|--------|----------|----------|---|------------|-----------|
| DIM    | Input transformation | Random resize + padding | 16/255 | 10 | ResNet-18 |
| SGM    | Model-related | Scale residual-path gradients (γ=0.2) | 16/255 | 10 | ResNet-18 |
| MIG    | Gradient (IG) | Integrated gradients along linear path | 16/255 | 10 | ResNet-18 |
| OPS    | Input transformation | Operator + perturbation neighbourhood sampling | 16/255 | 10 | ResNet-18 |
| MUMODIG | Gradient (IG) | Multi-baseline Monotone DIG + expectation-over-transforms | 16/255 | 10 | ResNet-18 |

OPS is implemented in `transferattack/input_transformation/ops.py`
(ported from [the-full/OPS](https://github.com/the-full/OPS)).

MUMODIG is implemented in `transferattack/gradient/mumodig.py`
(ported from [RYC-98/MuMoDIG](https://github.com/RYC-98/MuMoDIG)).
The helper class `LBQuantization` lives in `transferattack/lb_quantization.py`.

All attacks use **ε = 16/255** (L∞) and **ResNet-18** as the surrogate model,
consistent with the standard evaluation protocol in `README.md`.
