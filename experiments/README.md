# Experiments — High-Transferability Attack Robustness Benchmark

This directory contains a self-contained pipeline for reproducing the
"robustness under high-transferability attacks" experiment.

All pipeline scripts are **Python** (cross-platform) — no Bash required.
The legacy `.sh` files are kept for reference but `python experiments/xxx.py`
is the recommended interface on all operating systems (Linux, macOS, Windows).

---

## Directory layout

```
experiments/
├── run_attacks.py       # Generate DIM / SGM / MIG / OPS / MUMODIG adv. examples
├── eval_clean.py        # Baseline clean-accuracy cross-model eval
├── eval_at.py           # AT defense evaluation
├── eval_diffpure.py     # DiffPure defense evaluation
├── eval_nrp.py          # NRP defense evaluation (optional)
├── eval_freqpure.py     # FreqPure defense evaluation (new)
├── eval_dcpurify.py     # DC-Purify defense evaluation (new)
├── collect_results.py   # Parse logs → RA % → Markdown table
└── results/             # Auto-created by eval scripts
    ├── at/
    ├── diffpure/
    ├── nrp/
    ├── freqpure/
    └── dcpurify/
```

Adversarial examples are written to:
```
adv_data/
├── dim/resnet18/
├── sgm/resnet18/
├── mig/resnet18/
├── ops/resnet18/
└── mumodig/resnet18/
```

---

## Step-by-step reproduction

### 1. Prerequisites

**Python environment** (all packages in `requirements.txt`):
```
pip install -r requirements.txt
```

**Data** — ImageNet 1 000-image subset:
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

**DiffPure / FreqPure / DC-Purify diffusion checkpoint** (shared weight):
```
# Linux / macOS
wget -P defense/models/ \
  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

# Windows (PowerShell)
Invoke-WebRequest -Uri https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt `
    -OutFile defense\models\256x256_diffusion_uncond.pt
```

> FreqPure and DC-Purify reuse the **same** `256x256_diffusion_uncond.pt` as DiffPure.

---

### 2. Generate adversarial examples

```
python experiments/run_attacks.py --data_dir /path/to/data --gpu 0
```

Writes adversarial images into `adv_data/{dim,sgm,mig,ops,mumodig}/resnet18/`.

> `kornia` is required by OPS and MUMODIG (added to `requirements.txt`).
> Install with `pip install kornia` if missing.

---

### 3. Evaluate clean accuracy

```
python experiments/eval_clean.py --data_dir /path/to/data --gpu 0
```

---

### 4. Evaluate AT defense

```
python experiments/eval_at.py --data_dir /path/to/data --gpu 0
```

Results → `experiments/results/at/{dim,sgm,mig,ops,mumodig}.txt`

---

### 5. Evaluate DiffPure defense

```
python experiments/eval_diffpure.py --data_dir /path/to/data --gpu 0
```

> ⚠️ DiffPure is compute-intensive (~1 h per 1 000 images at batch-size 4).
> Add `--num_sub 100` for a quick sanity check.

Results → `experiments/results/diffpure/{attack}.log`

---

### 6. Evaluate NRP defense (optional)

```
python experiments/eval_nrp.py --data_dir /path/to/data --gpu 0
```

---

### 7. Evaluate FreqPure defense

**Source:** https://github.com/GaozhengPei/FreqPure

**One-time setup:**
```
cd defense
git clone https://github.com/GaozhengPei/FreqPure freqpure
pip install xformers==0.0.22 pyarrow==11.0.0 nested_dict
pip install "git+https://github.com/RobustBench/robustbench.git"
```

**Run (single GPU):**
```
python experiments/eval_freqpure.py --gpu 0
```

**Run (multi-GPU, e.g. 4×):**
```
python experiments/eval_freqpure.py --gpu 0,1,2,3 --nproc 4
```

> FreqPure generates its own internal PGD adversarial examples (adaptive
> white-box evaluation).  A **single** RA value is reported, covering all
> attack columns in the comparison table.
>
> The diffusion weight is symlinked automatically from
> `defense/models/256x256_diffusion_uncond.pt`.

Result → `experiments/results/freqpure/freqpure.txt`

---

### 8. Evaluate DC-Purify defense

**Source:** https://github.com/GaozhengPei/Purification

**One-time setup:**
```
cd defense
git clone https://github.com/GaozhengPei/Purification dcpurify
pip install pyarrow==11.0.0 cleverhans
pip install "git+https://github.com/RobustBench/robustbench.git"
```

**Run (single GPU):**
```
python experiments/eval_dcpurify.py --gpu 0
```

**Run (multi-GPU, e.g. 2×):**
```
python experiments/eval_dcpurify.py --gpu 0,1 --nproc 2
```

> DC-Purify also uses internal PGD.  Same adaptive-evaluation caveat as
> FreqPure applies.

Result → `experiments/results/dcpurify/dcpurify.txt`

---

### 9. Collect results and print comparison table

```
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
| FreqPure           | 68.4 | 68.4 | 68.4 | 68.4 | 68.4 | 68.4 |
| DC-Purify          | 70.2 | 70.2 | 70.2 | 70.2 | 70.2 | 70.2 |
| **Ours**           | --   | --   | --   | --   | --   | --   |
```

> FreqPure and DC-Purify show the same RA in all attack columns because they
> run a single internal evaluation (not per pre-generated attack directory).

---

## Attack details

| Attack  | Category             | Key idea                                               | ε      | Iterations | Surrogate |
|---------|----------------------|--------------------------------------------------------|--------|------------|-----------|
| DIM     | Input transformation | Random resize + padding                                | 16/255 | 10         | ResNet-18 |
| SGM     | Model-related        | Scale residual-path gradients (γ=0.2)                  | 16/255 | 10         | ResNet-18 |
| MIG     | Gradient (IG)        | Integrated gradients along linear path                 | 16/255 | 10         | ResNet-18 |
| OPS     | Input transformation | Operator + perturbation neighbourhood sampling         | 16/255 | 10         | ResNet-18 |
| MUMODIG | Gradient (IG)        | Multi-baseline Monotone DIG + expectation-over-transforms | 16/255 | 10      | ResNet-18 |

OPS is implemented in `transferattack/input_transformation/ops.py`
(ported from [the-full/OPS](https://github.com/the-full/OPS)).

MUMODIG is implemented in `transferattack/gradient/mumodig.py`
(ported from [RYC-98/MuMoDIG](https://github.com/RYC-98/MuMoDIG)).
The helper class `LBQuantization` lives in `transferattack/lb_quantization.py`.

All attacks use **ε = 16/255** (L∞) and **ResNet-18** as the surrogate model.

