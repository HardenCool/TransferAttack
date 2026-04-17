#!/usr/bin/env python3
"""
eval_dcpurify.py
================
Evaluate DC-Purify defense (https://github.com/GaozhengPei/Purification) on
ImageNet using its standard evaluation pipeline.

DC-Purify applies a selective diffusion-based purification controlled by
attention-based masks (threshold / threshold_percent). It uses the **same**
diffusion model weights as DiffPure (``256x256_diffusion_uncond.pt``).

Setup (one-time)
----------------
1. Clone the DC-Purify repository::

       cd defense
       git clone https://github.com/GaozhengPei/Purification dcpurify

2. Install extra dependencies::

       pip install pyarrow==11.0.0 cleverhans
       pip install git+https://github.com/RobustBench/robustbench.git

3. Ensure the diffusion model weight is present (shared with DiffPure)::

       defense/models/256x256_diffusion_uncond.pt

   Download from:
   https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

Notes
-----
* DC-Purify generates its own PGD adversarial examples internally.
  Results reflect robustness under adaptive attacks (stricter than our
  transfer-attack evaluation).
* Logs are saved to ``experiments/results/dcpurify/dcpurify_imagenet.log``.
  RA is parsed from ``acc_adv: XX.XXX%`` lines in the output.

Usage
-----
    python experiments/eval_dcpurify.py [--gpu GPU_ID] [--nproc N]
                                        [--n_iter N] [--eot N] [--ensemble N]
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[1]
DCPURIFY_DIR = REPO_ROOT / "defense" / "dcpurify"
RESULTS_DIR  = REPO_ROOT / "experiments" / "results" / "dcpurify"

# Regex for DC-Purify output: "acc_adv: 45.312%"
ACC_ADV_RE = re.compile(r"acc_adv\s*:\s*([\d.]+)\s*%", re.IGNORECASE)


def check_setup():
    if not DCPURIFY_DIR.is_dir() or not (DCPURIFY_DIR / "ddp_test.py").is_file():
        print(
            f"[ERROR] DC-Purify repo not found at {DCPURIFY_DIR}\n"
            "Please clone it first:\n"
            f"    cd {REPO_ROOT / 'defense'}\n"
            "    git clone https://github.com/GaozhengPei/Purification dcpurify\n"
        )
        sys.exit(1)

    weight = REPO_ROOT / "defense" / "models" / "256x256_diffusion_uncond.pt"
    pretrained_dst = DCPURIFY_DIR / "pretrained" / "guided_diffusion"
    pretrained_dst.mkdir(parents=True, exist_ok=True)
    target_link = pretrained_dst / "256x256_diffusion_uncond.pt"
    if not target_link.exists():
        if weight.is_file():
            # Prefer a symlink (saves disk space); fall back to copying on
            # systems where symlinks require elevated privileges (e.g. Windows).
            try:
                target_link.symlink_to(weight.resolve())
                print(f"[INFO] Symlinked diffusion weight: {target_link} -> {weight}")
            except (OSError, NotImplementedError):
                import shutil
                shutil.copy2(str(weight), str(target_link))
                print(f"[INFO] Copied diffusion weight to {target_link}")
        else:
            print(
                f"[WARNING] Diffusion weight not found at {weight}\n"
                "Download from:\n"
                "  https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
                "256x256_diffusion_uncond.pt\n"
                f"and place it at {weight}"
            )


def patch_imagenet_path(imagenet_dir: str):
    """
    DC-Purify reads ``imagenet_path`` from path.py; patch it before evaluation.
    """
    path_py  = DCPURIFY_DIR / "path.py"
    original = path_py.read_text()
    patched  = re.sub(
        r"(imagenet_path\s*=\s*)['\"].*?['\"]",
        f"imagenet_path = '{imagenet_dir}'",
        original,
    )
    path_py.write_text(patched)
    return original


def main():
    parser = argparse.ArgumentParser(description="Evaluate DC-Purify defense on ImageNet.")
    parser.add_argument("--imagenet_dir", default="",
                        help="Path to full ImageNet validation set root. "
                             "Leave empty to use DC-Purify's default path.py setting.")
    parser.add_argument("--gpu",       default="0", help="Comma-separated GPU IDs.  Default: 0")
    parser.add_argument("--nproc",     type=int, default=1,
                        help="Number of GPUs for torchrun (nproc_per_node).  Default: 1")
    parser.add_argument("--n_iter",    type=int, default=20,
                        help="PGD attack iterations inside DC-Purify eval.  Default: 20")
    parser.add_argument("--eot",       type=int, default=20,
                        help="EOT samples per PGD step.  Default: 20")
    parser.add_argument("--ensemble",  type=int, default=20,
                        help="Ensemble runs for purification.  Default: 20")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (must divide 512).  Default: 1")
    args = parser.parse_args()

    check_setup()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    original_path_py = None
    if args.imagenet_dir:
        original_path_py = patch_imagenet_path(args.imagenet_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_path = RESULTS_DIR / "dcpurify_imagenet.log"

    print("=" * 60)
    print(" DC-Purify defense evaluation (ImageNet)")
    print(f" DC-Purify dir : {DCPURIFY_DIR}")
    print(f" Log           : {log_path}")
    print(f" GPUs          : {args.gpu}  (nproc_per_node={args.nproc})")
    print("=" * 60)

    # DC-Purify must be launched with torchrun for DDP
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        "ddp_test.py",
        "--dataset",             "imagenet",
        "--batch_size",          str(args.batch_size),
        "--strength_l",          "0.4",
        "--strength_s",          "0.2",
        "--threshold_percent",   "0.15",
        "--classifier_name",     "ResNet50",
        "--attack_ddim_steps",   "10",
        "--defense_ddim_steps",  "500",
        "--forward_noise_steps", "3",
        "--attack_method",       "pgd",
        "--n_iter",              str(args.n_iter),
        "--eot",                 str(args.eot),
        "--num_ensemble_runs",   str(args.ensemble),
    ]

    print(f"\n>>> Running: {' '.join(cmd)}\n")

    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            cmd,
            cwd=str(DCPURIFY_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_fh.write(result.stdout)

    # Restore path.py
    if original_path_py is not None:
        (DCPURIFY_DIR / "path.py").write_text(original_path_py)

    # Parse final acc_adv
    lines = result.stdout.splitlines()
    ra_str = "--"
    for line in reversed(lines):
        m = ACC_ADV_RE.search(line)
        if m:
            ra_str = f"{float(m.group(1)):.1f}"
            break

    print("\n".join(lines[-20:]))

    ra_file = RESULTS_DIR / "dcpurify.txt"
    ra_file.write_text(f"ASR:{100.0 - float(ra_str):.2f}%\n" if ra_str != "--" else "ASR:--\n")

    if result.returncode != 0:
        print(f"[WARNING] DC-Purify evaluation exited with code {result.returncode}")

    print(f"\nDC-Purify RA (acc_adv): {ra_str}%")
    print(f"Result saved to {ra_file}")
    print(f"Full log saved to {log_path}")


if __name__ == "__main__":
    main()
