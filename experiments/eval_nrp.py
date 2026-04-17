#!/usr/bin/env python3
"""
eval_nrp.py
===========
Evaluate NRP (Neural Representation Purifier) defense against DIM, SGM, MIG,
OPS, and MUMODIG adversarial examples.
Cross-platform Python replacement for eval_nrp.sh.

Prerequisites
-------------
Download defense/models/NRP.pth from:
    https://drive.google.com/drive/folders/1NfSjLzc-MtkYHLumcKYs6OqC2X_zWy3g

Usage
-----
    python experiments/eval_nrp.py [--data_dir DATA] [--gpu GPU_ID]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTACKS = ["dim", "sgm", "mig", "ops", "mumodig"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate NRP defense.")
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"),
                        help="Path to ImageNet subset.  Default: ./data")
    parser.add_argument("--gpu", default="0", help="CUDA device ID.  Default: 0")
    args = parser.parse_args()

    data_dir    = Path(args.data_dir)
    nrp_model   = REPO_ROOT / "defense" / "models" / "NRP.pth"
    results_dir = REPO_ROOT / "experiments" / "results" / "nrp"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not nrp_model.is_file():
        print(f"[ERROR] NRP model not found: {nrp_model}")
        print("Download from: https://drive.google.com/drive/folders/1NfSjLzc-MtkYHLumcKYs6OqC2X_zWy3g")
        sys.exit(1)

    print("=" * 60)
    print(" NRP defense evaluation")
    print(f" NRP model  : {nrp_model}")
    print(f" Data dir   : {data_dir}")
    print(f" Results    : {results_dir}")
    print("=" * 60)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for attack in ATTACKS:
        adv_dir = REPO_ROOT / "adv_data" / attack / "resnet18"
        if not adv_dir.is_dir():
            print(f"[SKIP] {attack}: adversarial directory not found ({adv_dir})")
            continue

        purified_dir = REPO_ROOT / "defense" / "nrp" / "purified_data" / attack / "resnet18"
        purified_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> [NRP / {attack}]  purifying -> {purified_dir}")

        # Step 1: purify
        cmd_purify = [
            sys.executable,
            str(REPO_ROOT / "defense" / "nrp" / "purify.py"),
            "--dir",       str(adv_dir),
            "--output",    str(purified_dir),
            "--purifier",  "NRP",
            "--model_pth", str(nrp_model),
            "--dynamic",
            "--GPU_ID",    args.gpu,
        ]
        purify_log = results_dir / f"{attack}_purify.log"
        with open(purify_log, "w") as log_fh:
            r = subprocess.run(cmd_purify, cwd=str(REPO_ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
            log_fh.write(r.stdout)
        if r.returncode != 0:
            print(f"[WARNING] NRP purification for {attack} exited {r.returncode}")

        # Step 2: evaluate purified images
        print("    evaluating purified images...")
        cmd_eval = [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "--input_dir",  str(data_dir),
            "--output_dir", str(purified_dir),
            "--eval",
            "--GPU_ID",    args.gpu,
        ]
        eval_log = results_dir / f"{attack}_eval.log"
        with open(eval_log, "w") as log_fh:
            r = subprocess.run(cmd_eval, cwd=str(REPO_ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
            log_fh.write(r.stdout)

        lines = r.stdout.splitlines()
        for line in lines[-10:]:
            print(line)
        print(f"<<< [NRP / {attack}] done. -> {eval_log}")

    print(f"\nNRP evaluation complete. Logs in {results_dir}/")


if __name__ == "__main__":
    main()
