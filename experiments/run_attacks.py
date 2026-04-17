#!/usr/bin/env python3
"""
run_attacks.py
==============
Generate adversarial examples for DIM, SGM, MIG, OPS, and MUMODIG using
resnet18 as the surrogate model.  Cross-platform Python replacement for
run_attacks.sh.

Usage
-----
    python experiments/run_attacks.py [--data_dir DATA] [--gpu GPU_ID]

All five attacks are implemented natively in this repository.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_attack(attack: str, data_dir: Path, adv_root: Path, gpu: str, extra_args: list | None = None):
    out_dir = adv_root / attack / "resnet18"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [{attack}] generating adversarial examples -> {out_dir}")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--input_dir",  str(data_dir),
        "--output_dir", str(out_dir),
        "--attack",     attack,
        "--model",      "resnet18",
        "--epoch",      "10",
        "--eps",        str(16 / 255),
        "--alpha",      str(1.6 / 255),
        "--momentum",   "1.0",
        "--GPU_ID",     gpu,
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        print(f"[WARNING] [{attack}] exited with code {result.returncode}")
    else:
        print(f"<<< [{attack}] done.")


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples for all benchmark attacks.")
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"),
                        help="Path to ImageNet subset (images/ + labels.csv). Default: ./data")
    parser.add_argument("--gpu", default="0", help="CUDA device ID. Default: 0")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    adv_root = REPO_ROOT / "adv_data"

    print("=" * 60)
    print(f" Repository  : {REPO_ROOT}")
    print(f" Data dir    : {data_dir}")
    print(f" Surrogate   : resnet18")
    print(f" GPU         : {args.gpu}")
    print(f" Adv output  : {adv_root}")
    print("=" * 60)

    # DIM — input diversity
    run_attack("dim", data_dir, adv_root, args.gpu)

    # SGM — skip gradient method
    run_attack("sgm", data_dir, adv_root, args.gpu)

    # MIG — momentum integrated gradients (uses its own alpha schedule)
    mig_out = adv_root / "mig" / "resnet18"
    mig_out.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> [mig] generating adversarial examples -> {mig_out}")
    cmd_mig = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--input_dir",  str(data_dir),
        "--output_dir", str(mig_out),
        "--attack",     "mig",
        "--model",      "resnet18",
        "--epoch",      "10",
        "--eps",        str(16 / 255),
        "--GPU_ID",     args.gpu,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    r = subprocess.run(cmd_mig, cwd=str(REPO_ROOT), env=env)
    if r.returncode != 0:
        print(f"[WARNING] [mig] exited with code {r.returncode}")
    else:
        print("<<< [mig] done.")

    # OPS — operator + perturbation neighbourhood sampling
    run_attack("ops", data_dir, adv_root, args.gpu)

    # MUMODIG — multi-baseline monotone DIG + expectation-over-transforms
    run_attack("mumodig", data_dir, adv_root, args.gpu)

    print("\n" + "=" * 60)
    print(" Attack generation complete.  Adversarial examples saved to:")
    for atk in ["dim", "sgm", "mig", "ops", "mumodig"]:
        print(f"   {adv_root / atk / 'resnet18'}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
