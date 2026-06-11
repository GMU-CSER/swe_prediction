#!/usr/bin/env python
"""Aggregate sweep_results/*/metrics.json into a comparison table (stdout + CSV)."""
import argparse
import csv
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", default="/projects/kzhou6/bcui2/research/geo_project/sweep_results")
args = ap.parse_args()

rows = []
for mj in sorted(Path(args.out_dir).glob("*/metrics.json")):
    with open(mj) as f:
        rows.append(json.load(f))

if not rows:
    print(f"no metrics.json found under {args.out_dir}")
    raise SystemExit(0)

cols = ["run_name", "curve", "base_weight", "hypsometry_weight",
        "test_rmse", "test_r2", "final_val_r2", "mean_base", "mean_phys",
        "weighted_phys", "train_seconds"]

w = {c: max(len(c), max(len(f"{r.get(c,'')}") for r in rows)) for c in cols}
print("  ".join(c.ljust(w[c]) for c in cols))
print("  ".join("-" * w[c] for c in cols))
for r in sorted(rows, key=lambda r: (r.get("curve", ""), float(r.get("hypsometry_weight", 0)))):
    def fmt(c):
        v = r.get(c, "")
        if isinstance(v, float):
            return (f"{v:.4f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}")
        return str(v)
    print("  ".join(fmt(c).ljust(w[c]) for c in cols))

csv_path = Path(args.out_dir) / "sweep_summary.csv"
with open(csv_path, "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    wr.writeheader()
    wr.writerows(rows)
print(f"\nwrote {csv_path}")
