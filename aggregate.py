"""Pool the per-seed fold tables into mean +- standard deviation (Table 2)."""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from preprocess import COHORTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PTM-AGT result tables")
    parser.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--results-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir or os.path.join(args.root, "results")

    summary_rows = []
    for cohort in COHORTS:
        paths = sorted(glob.glob(os.path.join(results_dir, cohort, "seed*_folds.csv")))
        if not paths:
            continue

        pooled = {}
        for path in paths:
            with open(path, newline="") as handle:
                for row in csv.DictReader(handle):
                    if not row["fold"].isdigit():  # skip the mean and std rows
                        continue
                    for key, value in row.items():
                        if key not in ("seed", "fold"):
                            pooled.setdefault(key, []).append(float(value))

        entry = {"cohort": cohort, "seeds": len(paths), "runs": len(next(iter(pooled.values())))}
        for key, values in pooled.items():
            entry[key] = f"{np.mean(values):.2f} +- {np.std(values):.2f}"
        summary_rows.append(entry)

        print(f"\n{cohort}  (seeds={entry['seeds']}, seed-fold runs={entry['runs']})")
        for key, values in pooled.items():
            print(f"  {key:6s} = {np.mean(values):6.2f} +- {np.std(values):5.2f}  (n={len(values)})")

    if summary_rows:
        # cohorts report different metrics (F1/AUC vs F1_w/F1_m), so take the union
        metric_keys = []
        for row in summary_rows:
            metric_keys += [k for k in row if k not in ("cohort", "seeds", "runs") and k not in metric_keys]
        keys = ["cohort", "seeds", "runs"] + metric_keys
        out_path = os.path.join(results_dir, "summary_all_seeds.csv")
        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, restval="")
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
