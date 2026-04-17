#!/usr/bin/env python3
"""
eval_diffpure.py
================
Evaluate DiffPure (SDE variant, ImageNet) defense against DIM, SGM, MIG,
OPS, and MUMODIG adversarial examples.
Cross-platform Python replacement for eval_diffpure.sh.

Prerequisites
-------------
Download defense/models/256x256_diffusion_uncond.pt from OpenAI:
    https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

Usage
-----
    python experiments/eval_diffpure.py [--data_dir DATA] [--gpu GPU_ID]

Note: DiffPure is slow (~1 h per 1000 images at batch-size 4 on a 4090).
      Use --num_sub to limit the number of images for a quick sanity check.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTACKS = ["dim", "sgm", "mig", "ops", "mumodig"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate DiffPure defense.")
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"),
                        help="Path to ImageNet subset.  Default: ./data")
    parser.add_argument("--gpu", default="0", help="CUDA device ID.  Default: 0")
    parser.add_argument("--num_sub", type=int, default=1000,
                        help="Number of images to evaluate (reduce for quick test).")
    parser.add_argument("--adv_batch_size", type=int, default=4,
                        help="Batch size for the adversarial evaluation.  Default: 4")
    args = parser.parse_args()

    data_dir     = Path(args.data_dir)
    diffpure_dir = REPO_ROOT / "defense" / "diffpure"
    results_dir  = REPO_ROOT / "experiments" / "results" / "diffpure"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" DiffPure defense evaluation (diffusion_type=sde)")
    print(f" Classifier  : resnet101")
    print(f" Data dir    : {data_dir}")
    print(f" Results     : {results_dir}")
    print("=" * 60)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for attack in ATTACKS:
        adv_dir = REPO_ROOT / "adv_data" / attack / "resnet18"
        if not adv_dir.is_dir():
            print(f"[SKIP] {attack}: adversarial directory not found ({adv_dir})")
            continue

        log_path = results_dir / f"{attack}.log"
        print(f"\n>>> [DiffPure / {attack}]  adv_dir={adv_dir}")

        cmd = [
            sys.executable,
            str(diffpure_dir / "diffpure.py"),
            "--image_folder",    str(data_dir),
            "--adv_dir",         str(adv_dir),
            "--config",          "imagenet.yml",
            "--t",               "150",
            "--adv_eps",         "0.0627",
            "--adv_batch_size",  str(args.adv_batch_size),
            "--num_sub",         str(args.num_sub),
            "--domain",          "imagenet",
            "--classifier_name", "resnet101",
            "--diffusion_type",  "sde",
            "--score_type",      "guided_diffusion",
        ]

        with open(log_path, "w") as log_fh:
            result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True)
            log_fh.write(result.stdout)

        # Print tail of output and look for ASR
        lines = result.stdout.splitlines()
        for line in lines[-20:]:
            print(line)
        if result.returncode != 0:
            print(f"[WARNING] DiffPure for {attack} exited {result.returncode}")
        print(f"<<< [DiffPure / {attack}] done. -> {log_path}")

    print(f"\nDiffPure evaluation complete. Logs in {results_dir}/")
    print("ASR lines are tagged as 'ASR:XX.XX%' in each log file.")


if __name__ == "__main__":
    main()
