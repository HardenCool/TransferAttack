#!/usr/bin/env python3
"""
eval_freqpure.py
================
Evaluate FreqPure defense (https://github.com/GaozhengPei/FreqPure) on
ImageNet using its standard evaluation pipeline.

FreqPure applies frequency-domain filtering (amplitude & phase cutting)
followed by a guided-diffusion denoising step. It uses the **same** diffusion
model weights as DiffPure (``256x256_diffusion_uncond.pt``).

Setup (one-time)
----------------
1. Clone the FreqPure repository::

       cd defense
       git clone https://github.com/GaozhengPei/FreqPure freqpure

2. Install its extra dependencies (inside your existing conda/venv)::

       pip install xformers==0.0.22 pyarrow==11.0.0 nested_dict
       pip install git+https://github.com/RobustBench/robustbench.git

3. Ensure the diffusion model weight is present (shared with DiffPure)::

       defense/models/256x256_diffusion_uncond.pt

   Download from:
   https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

Notes
-----
* FreqPure generates its own PGD adversarial examples internally (white-box
  adaptive attack evaluation). The results reflect robustness under adaptive
  attacks, which is a **stricter** setting than our transfer-attack evaluation.
* Logs are saved to ``experiments/results/freqpure/freqpure_imagenet.log``.
  The script parses ``acc_adv: XX.XXX%`` lines for the Robust Accuracy (RA).

Usage
-----
    python experiments/eval_freqpure.py [--gpu GPU_ID] [--nproc N]
                                        [--n_iter N] [--eot N] [--ensemble N]
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FREQPURE_DIR = REPO_ROOT / "defense" / "freqpure"
RESULTS_DIR  = REPO_ROOT / "experiments" / "results" / "freqpure"

# Regex for FreqPure output: "acc_adv: 45.312%"
ACC_ADV_RE = re.compile(r"acc_adv\s*:\s*([\d.]+)\s*%", re.IGNORECASE)


def check_setup():
    if not FREQPURE_DIR.is_dir() or not (FREQPURE_DIR / "ddp_test.py").is_file():
        print(
            f"[ERROR] FreqPure repo not found at {FREQPURE_DIR}\n"
            "Please clone it first:\n"
            f"    cd {REPO_ROOT / 'defense'}\n"
            "    git clone https://github.com/GaozhengPei/FreqPure freqpure\n"
        )
        sys.exit(1)

    weight = REPO_ROOT / "defense" / "models" / "256x256_diffusion_uncond.pt"
    pretrained_dst = FREQPURE_DIR / "pretrained" / "guided_diffusion"
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
    FreqPure reads ``imagenet_path`` from path.py. We patch it in-place before
    running the evaluation and restore it afterwards.
    """
    path_py = FREQPURE_DIR / "path.py"
    original = path_py.read_text()
    patched  = re.sub(
        r"(imagenet_path\s*=\s*)['\"].*?['\"]",
        f"imagenet_path = '{imagenet_dir}'",
        original,
    )
    path_py.write_text(patched)
    return original  # caller restores this


def main():
    parser = argparse.ArgumentParser(description="Evaluate FreqPure defense on ImageNet.")
    parser.add_argument("--imagenet_dir", default="",
                        help="Path to full ImageNet validation set root. "
                             "Leave empty to use FreqPure's default path.py setting.")
    parser.add_argument("--gpu",       default="0", help="Comma-separated GPU IDs.  Default: 0")
    parser.add_argument("--nproc",     type=int, default=1,
                        help="Number of GPUs for torchrun (nproc_per_node).  Default: 1")
    parser.add_argument("--n_iter",    type=int, default=20,
                        help="PGD attack iterations inside FreqPure eval.  Default: 20")
    parser.add_argument("--eot",       type=int, default=5,
                        help="EOT samples per PGD step.  Default: 5")
    parser.add_argument("--ensemble",  type=int, default=5,
                        help="Ensemble runs for purification.  Default: 5")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (must divide 512).  Default: 1")
    args = parser.parse_args()

    check_setup()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Optionally patch imagenet_path in path.py
    original_path_py = None
    if args.imagenet_dir:
        original_path_py = patch_imagenet_path(args.imagenet_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    log_path = RESULTS_DIR / "freqpure_imagenet.log"

    print("=" * 60)
    print(" FreqPure defense evaluation (ImageNet)")
    print(f" FreqPure dir : {FREQPURE_DIR}")
    print(f" Log          : {log_path}")
    print(f" GPUs         : {args.gpu}  (nproc_per_node={args.nproc})")
    print("=" * 60)

    # FreqPure must be launched with torchrun for DDP
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        "ddp_test.py",
        "--dataset",              "imagenet",
        "--batch_size",           str(args.batch_size),
        "--amplitude_cut_range",  "10",
        "--phase_cut_range",      "10",
        "--delta",                "0.3",
        "--def_max_timesteps",    ",".join(["50"] * 8),
        "--def_num_denoising_steps", ",".join(["5"] * 8),
        "--att_max_timesteps",    "50",
        "--att_num_denoising_steps", "1",
        "--num_ensemble_runs",    str(args.ensemble),
        "--attack_method",        "pgd",
        "--n_iter",               str(args.n_iter),
        "--eot",                  str(args.eot),
    ]

    print(f"\n>>> Running: {' '.join(cmd)}\n")

    with open(log_path, "w") as log_fh:
        result = subprocess.run(
            cmd,
            cwd=str(FREQPURE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_fh.write(result.stdout)

    # Restore path.py
    if original_path_py is not None:
        (FREQPURE_DIR / "path.py").write_text(original_path_py)

    # Parse final acc_adv
    lines = result.stdout.splitlines()
    ra_str = "--"
    for line in reversed(lines):
        m = ACC_ADV_RE.search(line)
        if m:
            ra_str = f"{float(m.group(1)):.1f}"
            break

    print("\n".join(lines[-20:]))

    ra_file = RESULTS_DIR / "freqpure.txt"
    ra_file.write_text(f"ASR:{100.0 - float(ra_str):.2f}%\n" if ra_str != "--" else "ASR:--\n")

    if result.returncode != 0:
        print(f"[WARNING] FreqPure evaluation exited with code {result.returncode}")

    print(f"\nFreqPure RA (acc_adv): {ra_str}%")
    print(f"Result saved to {ra_file}")
    print(f"Full log saved to {log_path}")


if __name__ == "__main__":
    main()
