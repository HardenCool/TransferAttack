#!/usr/bin/env python3
"""
collect_results.py
==================
Parse ASR log files produced by eval_at.sh, eval_diffpure.sh, and eval_nrp.sh,
convert to Robust Accuracy (RA = 100 - ASR), and print a Markdown comparison
table matching the format used in the paper.

Usage
-----
    python experiments/collect_results.py [--results_root RESULTS_ROOT]

The script expects the following directory layout (produced by the eval_*.sh
scripts):

    experiments/results/
        at/
            dim.txt      <- contains a line like "ASR:XX.XX%"
            sgm.txt
            mig.txt
            ops.txt      <- optional; skipped if missing
            mumodig.txt  <- optional; skipped if missing
        diffpure/
            dim.log
            sgm.log
            mig.log
            ops.log
            mumodig.log
        nrp/
            dim_eval.log
            sgm_eval.log
            mig_eval.log
            ops_eval.log
            mumodig_eval.log

All files are optional; missing entries are shown as "--".
"""

import argparse
import os
import re
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ATTACKS = ["dim", "sgm", "mig", "ops", "mumodig"]

# Column headers for the results table
ATTACK_LABELS = {
    "dim":     "DIM",
    "sgm":     "SGM",
    "mig":     "MIG",
    "ops":     "OPS",
    "mumodig": "MUMODIG",
}

DEFENSES = ["at", "diffpure", "nrp", "freqpure", "dcpurify"]

DEFENSE_LABELS = {
    "at":       "AT (Adv. Training)",
    "diffpure": "DiffPure",
    "nrp":      "NRP",
    "freqpure": "FreqPure",
    "dcpurify": "DC-Purify",
}

# Filename patterns per defense.
# FreqPure and DC-Purify run a single global evaluation (not per-attack),
# so they share one result file stored under their sub-directory.
FILE_PATTERNS = {
    "at":       "{attack}.txt",
    "diffpure": "{attack}.log",
    "nrp":      "{attack}_eval.log",
    "freqpure": "freqpure.txt",   # single file — same RA for every attack column
    "dcpurify": "dcpurify.txt",   # single file — same RA for every attack column
}

# Regex to find ASR value in a log / txt file
ASR_RE = re.compile(r"ASR\s*:\s*([\d.]+)\s*%", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_asr(filepath: str) -> float | None:
    """Return the first ASR percentage found in *filepath*, or None."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, "r", errors="replace") as fh:
        for line in fh:
            m = ASR_RE.search(line)
            if m:
                return float(m.group(1))
    return None


def asr_to_ra(asr: float | None) -> str:
    """Convert ASR to RA string; return '--' for missing values."""
    if asr is None:
        return "--"
    return f"{100.0 - asr:.1f}"


def avg_ra(values: list[str]) -> str:
    """Compute average RA ignoring '--' placeholders."""
    nums = [float(v) for v in values if v != "--"]
    if not nums:
        return "--"
    return f"{sum(nums) / len(nums):.1f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Collect and display RA results.")
    parser.add_argument(
        "--results_root",
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Root directory that contains at/, diffpure/, nrp/ sub-dirs.",
    )
    args = parser.parse_args()

    root = args.results_root

    # Defenses that share a single result file regardless of attack
    SINGLE_FILE_DEFENSES = {"freqpure", "dcpurify"}

    # Collect RA values:  table[defense][attack] = RA_string
    table: dict[str, dict[str, str]] = {}
    for defense in DEFENSES:
        table[defense] = {}
        if defense in SINGLE_FILE_DEFENSES:
            # One file, same RA reported for all attack columns
            filename = FILE_PATTERNS[defense]
            filepath = os.path.join(root, defense, filename)
            asr = parse_asr(filepath)
            ra  = asr_to_ra(asr)
            for attack in ATTACKS:
                table[defense][attack] = ra
        else:
            for attack in ATTACKS:
                filename = FILE_PATTERNS[defense].format(attack=attack)
                filepath = os.path.join(root, defense, filename)
                asr = parse_asr(filepath)
                table[defense][attack] = asr_to_ra(asr)

    # Add "Ours" row as placeholders (use "--" so avg_ra handles them cleanly)
    table["ours"] = {attack: "--" for attack in ATTACKS}
    DEFENSES_WITH_OURS = DEFENSES + ["ours"]
    DEFENSE_LABELS["ours"] = "**Ours** (placeholder)"

    # -----------------------------------------------------------------------
    # Print Markdown table
    # -----------------------------------------------------------------------
    attack_cols = [ATTACK_LABELS[a] for a in ATTACKS]
    header_row = "| 防御方法 | " + " | ".join(attack_cols) + " | 平均鲁棒性 |"
    sep_row    = "|" + "|".join(["---"] * (len(ATTACKS) + 2)) + "|"

    print("\n## 鲁棒准确率对比 (RA %) — 高迁移攻击\n")
    print(header_row)
    print(sep_row)

    for defense in DEFENSES_WITH_OURS:
        label = DEFENSE_LABELS[defense]
        vals = [table[defense][a] for a in ATTACKS]
        avg  = avg_ra(vals)
        row  = f"| {label} | " + " | ".join(vals) + f" | {avg} |"
        print(row)

    print()
    print("> Notes:")
    print("> - OPS: Operator-Perturbation Stochastic optimization")
    print(">   (transferattack/input_transformation/ops.py, ported from the-full/OPS).")
    print("> - MUMODIG: Multi-baseline Monotone DIG with expectation-over-transforms")
    print(">   (transferattack/gradient/mumodig.py, ported from RYC-98/MuMoDIG).")
    print("> - FreqPure (https://github.com/GaozhengPei/FreqPure): frequency-domain")
    print(">   filtering + guided-diffusion denoising; evaluated via eval_freqpure.py.")
    print(">   Uses the same 256x256_diffusion_uncond.pt weights as DiffPure.")
    print(">   Runs its own internal PGD attack (adaptive evaluation, stricter than")
    print(">   transfer-attack evaluation); a single RA value covers all attack columns.")
    print("> - DC-Purify (https://github.com/GaozhengPei/Purification): attention-mask-")
    print(">   guided selective diffusion purification; evaluated via eval_dcpurify.py.")
    print(">   Same weights & adaptive-evaluation caveat as FreqPure.")
    print("> - **Ours** rows are placeholders — fill in after running your own defense.")
    print()


if __name__ == "__main__":
    main()
