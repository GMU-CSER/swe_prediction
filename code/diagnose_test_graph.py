#!/usr/bin/env python
"""
Diagnose why trained GCN models predict near-zero SWE on the 2025 test-period
graphs despite high validation R2 (reported by Wei-Ting, 2026-06-10).

What it does (all read-only):
  1. Load the training graph and one test graph; check shapes and NaNs.
  2. Compare per-feature scaled distributions (train x is the StandardScaler
     output, so test |z| >> 3 means the feature is far outside the training
     distribution).
  3. Causal test: run a known-good model (our sweep run elv_hyps0, test
     R2=0.971 on the training-graph split) on the test graph
       (a) as-is, (b) with Slope neutralized, (c) Slope + fsca lags clipped.
  4. Print raw Slope/fsca stats from the train and test CSVs to identify
     which side is corrupted.

Usage:
  python diagnose_test_graph.py [--date 2025-01-15] [--model <path to .pth>]
  # writes everything to stdout; tee it to a log to keep a record, e.g.
  #   python diagnose_test_graph.py | tee ../..../sweep_results/diagnosis_2025-01-15.log

Findings as of 2026-06-10 (see sweep_results/diagnosis_2025-01-15.log):
  - Training CSV 'Slope' is degenerate (~constant 90 deg) while test CSV has
    real slopes (median 0.76 deg) -> scaled test Slope ~ -309 sigma.
  - Test CSV fsca_1..7 contain MODIS fill values (250) -> +6..14 sigma.
  - Fixing the single Slope column restores plausible predictions
    (max 3.0 in -> 38.7 in), proving the collapse is a data issue, not a
    model issue.
"""
import argparse
import numpy as np
import pandas as pd
import torch

from model_train import GCN_Model

# Sorted feature order produced by data_process.py (np.sort of final_columns).
LAG = lambda v: [v] + [f"{v}_{i}" for i in range(1, 8)]
COLS = (["Aspect", "Curvature", "Eastness", "Elevation", "Northness", "Slope"]
        + LAG("air_temperature_tmmn") + LAG("air_temperature_tmmx")
        + ["cos_day", "cos_lat", "cos_lon"]
        + LAG("fsca") + LAG("mean_vapor_pressure_deficit")
        + LAG("potential_evapotranspiration") + LAG("precipitation_amount")
        + LAG("relative_humidity_rmax") + LAG("relative_humidity_rmin")
        + ["sin_day", "sin_lat", "sin_lon", "snodas_mask"] + LAG("wind_speed"))
SLOPE_IDX = COLS.index("Slope")            # 5
FSCA_IDX = [COLS.index(c) for c in LAG("fsca")]  # 25..32

TRAIN_PT = "/groups/ESS/whung/swe_gnn/data/gnn_training_data.pt"
TEST_PT = "/groups/ESS/whung/swe_gnn/data/gnn_testing_data_{date}.pt"
TRAIN_CSV = "/groups/ESS/whung/swe_gnn/data/all_points_final_merged_training_snodas_mask_resnet_all_batch.csv"
TEST_CSV = ("/groups/ESS/whung/swe_gnn/data/testing_snodas_mask/"
            "testing_all_ready_{date}_merged.csv_snodas_mask.csv")
GOOD_MODEL = ("/projects/kzhou6/bcui2/research/geo_project/sweep_results/"
              "elv_hyps0/GCN_model.pth")


def pred_stats(p: torch.Tensor, name: str) -> None:
    p = p.numpy()
    print(f"{name:34s} min={p.min():8.2f} p50={np.percentile(p, 50):7.2f} "
          f"p95={np.percentile(p, 95):7.2f} max={p.max():8.2f} "
          f"| frac<0.1in: {(p < 0.1).mean() * 100:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-01-15")
    ap.add_argument("--model", default=GOOD_MODEL)
    args = ap.parse_args()

    tr = torch.load(TRAIN_PT, map_location="cpu", weights_only=False)
    te = torch.load(TEST_PT.format(date=args.date), map_location="cpu", weights_only=False)
    assert len(COLS) == tr.x.shape[1] == te.x.shape[1], "feature count mismatch"

    print(f"train x: {tuple(tr.x.shape)} | test x ({args.date}): {tuple(te.x.shape)}")
    print(f"NaNs in x  train: {torch.isnan(tr.x).sum().item()}  "
          f"test: {torch.isnan(te.x).sum().item()}")

    print("\n--- worst-shifted test features (train x is standardized, so "
          "|mean| >> 1 means out of training distribution) ---")
    ex = te.x.numpy()
    em, emax = np.nanmean(ex, axis=0), np.nanmax(np.abs(ex), axis=0)
    for i in np.argsort(-np.abs(em))[:12]:
        print(f"  {COLS[i]:34s} test_mean={em[i]:8.2f} test_max|z|={emax[i]:9.1f}")

    print(f"\n--- causal test with known-good model: {args.model} ---")
    m = GCN_Model(len(COLS), 64, 1)
    m.load_state_dict(torch.load(args.model, map_location="cpu"))
    m.eval()
    with torch.no_grad():
        p_tr = m(tr)
        vy = torch.var(tr.y, unbiased=False)
        vr = torch.var(tr.y - p_tr, unbiased=False)
        print(f"sanity, full-graph inference on TRAIN graph: R2={1 - (vr / vy).item():.4f}")
        pred_stats(p_tr, "  train pred")
        pred_stats(tr.y, "  train truth y")

        pred_stats(m(te), "TEST pred (as-is)")

        te2 = te.clone(); te2.x = te.x.clone()
        te2.x[:, SLOPE_IDX] = 0.0
        pred_stats(m(te2), "TEST pred (Slope neutralized)")

        te3 = te.clone(); te3.x = te2.x.clone()
        for i in FSCA_IDX:
            te3.x[:, i] = te3.x[:, i].clamp(-3, 3)
        pred_stats(m(te3), "TEST pred (Slope + fsca clipped)")

    print("\n--- raw CSV values (which side is corrupted?) ---")
    use = ["Slope", "fsca", "fsca_7", "Elevation"]
    t = pd.read_csv(TEST_CSV.format(date=args.date), usecols=lambda c: c in use)
    for c in use:
        s = pd.to_numeric(t[c], errors="coerce")
        print(f"  test  {c:10s} min={s.min():10.3f} p50={s.median():10.3f} max={s.max():10.3f}")
    t = pd.read_csv(TRAIN_CSV, usecols=use, nrows=2_000_000)
    for c in use:
        s = pd.to_numeric(t[c], errors="coerce")
        print(f"  train {c:10s} min={s.min():10.3f} p50={s.median():10.3f} max={s.max():10.3f}"
              "   (first 2M rows)")


if __name__ == "__main__":
    main()
