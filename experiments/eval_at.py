#!/usr/bin/env python3
"""
eval_at.py
==========
Evaluate AT (Adversarial Training, fast_adversarial, 4px) defense against
DIM, SGM, MIG, OPS, and MUMODIG adversarial examples.
Cross-platform Python replacement for eval_at.sh.

Prerequisites
-------------
Download defense/models/imagenet_model_weights_4px.pth.tar from:
    https://drive.google.com/drive/folders/1NfSjLzc-MtkYHLumcKYs6OqC2X_zWy3g

Usage
-----
    python experiments/eval_at.py [--data_dir DATA] [--gpu GPU_ID]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTACKS = ["dim", "sgm", "mig", "ops", "mumodig"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate AT defense.")
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"),
                        help="Path to ImageNet subset.  Default: ./data")
    parser.add_argument("--gpu", default="0", help="CUDA device ID.  Default: 0")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    checkpoint = REPO_ROOT / "defense" / "models" / "imagenet_model_weights_4px.pth.tar"
    results_dir = REPO_ROOT / "experiments" / "results" / "at"
    at_dir      = REPO_ROOT / "defense" / "at"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        print(f"[ERROR] AT checkpoint not found: {checkpoint}")
        print("Download from: https://drive.google.com/drive/folders/1NfSjLzc-MtkYHLumcKYs6OqC2X_zWy3g")
        sys.exit(1)

    print("=" * 60)
    print(" AT defense evaluation")
    print(f" Checkpoint : {checkpoint}")
    print(f" Label file : {data_dir / 'labels.csv'}")
    print(f" Results    : {results_dir}")
    print("=" * 60)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for attack in ATTACKS:
        adv_dir = REPO_ROOT / "adv_data" / attack / "resnet18"
        if not adv_dir.is_dir():
            print(f"[SKIP] {attack}: adversarial directory not found ({adv_dir})")
            continue

        output_prefix = f"at_results/{attack}_resnet18.txt"
        print(f"\n>>> [AT / {attack}]  adv_dir={adv_dir}")

        # Step 1: run AT evaluation
        cmd_at = [
            sys.executable, "main_fast.py",
            str(adv_dir),
            "--config",        "configs/configs_fast_4px_evaluate.yml",
            "--output_prefix", output_prefix,
            "--resume",        str(checkpoint),
            "--evaluate",
            "--restarts",      "10",
            "--GPU_ID",        args.gpu,
        ]
        log_path_at = at_dir / f"{output_prefix}.log"
        log_path_at.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path_at, "w") as log_fh:
            result = subprocess.run(cmd_at, cwd=str(at_dir), env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True)
            log_fh.write(result.stdout)
        if result.returncode != 0:
            print(f"[WARNING] AT evaluation for {attack} exited {result.returncode}")

        # Step 2: parse output → RA
        cmd_parse = [
            sys.executable,
            str(REPO_ROOT / "defense" / "check_output.py"),
            "--output_file", str(at_dir / output_prefix),
            "--label_file",  str(data_dir / "labels.csv"),
        ]
        result_txt = results_dir / f"{attack}.txt"
        with open(result_txt, "w") as out_fh:
            r = subprocess.run(cmd_parse, cwd=str(REPO_ROOT), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True)
            out_fh.write(r.stdout)
        print(r.stdout.strip())
        print(f"<<< [AT / {attack}] done. -> {result_txt}")

    print(f"\nAT evaluation complete. Results in {results_dir}/")


if __name__ == "__main__":
    main()
