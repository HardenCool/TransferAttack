#!/usr/bin/env python3
"""
eval_clean.py
=============
Evaluate clean (unperturbed) accuracy across the standard 8-model suite
(4 CNNs + 4 ViTs) defined in transferattack/utils.py.
Cross-platform Python replacement for eval_clean.sh.

Usage
-----
    python experiments/eval_clean.py [--data_dir DATA] [--gpu GPU_ID]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate clean accuracy across the 8-model suite.")
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data"),
                        help="Path to ImageNet subset.  Default: ./data")
    parser.add_argument("--gpu", default="0", help="CUDA device ID.  Default: 0")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_file = REPO_ROOT / "experiments" / "clean_acc.txt"

    print(f">>> Evaluating clean accuracy on {data_dir / 'images'}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--input_dir",  str(data_dir),
        "--output_dir", str(data_dir / "images"),
        "--eval",
        "--GPU_ID", args.gpu,
    ]

    with open(out_file, "w") as log_fh:
        result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True)
        log_fh.write(result.stdout)

    # Also print to console
    print(result.stdout)
    print(f"Clean accuracy results saved to {out_file}")


if __name__ == "__main__":
    main()
