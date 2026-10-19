"""Print measured budget comparisons from an executed benchmark.

Run ``python -m experiments.run_experiments`` first. This command only reads
actual output; it does not create substitute or fabricated benchmark values.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "evidence" / "results")
    parser.add_argument("--budget", type=float, default=0.25, help="Fraction among 0, .1, .25, .5, .75, 1")
    args = parser.parse_args()
    path = args.results / "summary_by_kind.csv"
    if not path.is_file():
        parser.error(f"No executed results at {path}; run the experiment CLI first.")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if abs(float(row["budget_fraction"]) - args.budget) < 1e-10]
    if not rows:
        parser.error("That budget endpoint is not present in the executed results.")
    print(f"Measured comparison at {100 * args.budget:g}% budget; random seeds averaged within image, then equal image means.")
    print(f"{'kind':10s} {'method':12s} {'images':>6s} {'MSE':>11s} {'SSIM':>8s} {'MSE gain':>10s}")
    for row in rows:
        print(f"{row['kind']:10s} {row['strategy']:12s} {int(row['n_images']):6d} {float(row['mse_mean']):11.7f} {float(row['ssim_mean']):8.4f} {100 * float(row['relative_mse_reduction_mean']):9.2f}%")


if __name__ == "__main__":
    main()
