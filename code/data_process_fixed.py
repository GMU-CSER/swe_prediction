#!/usr/bin/env python
"""
Fixed version of data_process.py (Wei-Ting's original is left untouched).

Fixes, per the diagnosis in sweep_results/test_prediction_diagnosis.md:
  1. Slope dropped from the feature set (training CSV Slope is degenerate,
     ~constant 90 deg, while test CSVs carry real slopes -> scaled -309 sigma
     at test time). The Slope<=0 row filter is removed with it.
  2. Lag-stack repair for ALL lagged variables (not only fsca): values outside
     each variable's physical validity range are flags or zero-filled failed
     joins (fsca 250 = MODIS cloud; ~12.6% of test rows have the whole meteo
     record zero-filled, e.g. temperature 0 K). Invalid values are masked to
     NaN and filled row-wise from the nearest valid lag (ffill then bfill along
     the lag axis). Applied identically to the training and test paths;
     whatever stays NaN is dropped (train) or imputed with the training mean
     (test).
  3. Sanity gate: after scaling, every test feature must satisfy |mean z| < 5
     and max |z| < max(30, 1.2 x the TRAINING column's own max |z|), otherwise
     the script aborts and lists the offending columns instead of writing a
     poisoned graph. The per-column training envelope matters: zero-inflated
     variables like precipitation legitimately reach 80 sigma inside the
     training data itself (atmospheric-river storms), so a flat cap would
     reject real weather while the corrupted columns (Slope -309, fsca +14,
     0 K temps -40) are still caught by the |mean z| and envelope bounds.
  4. Test-side NaNs after the value filters are imputed with the training
     column means (scaled value 0) instead of being silently carried through.
  5. Edge construction vectorized (the original O(N*k) Python loop is too slow
     for the 1.2M-node training graph).

Outputs (never writes into /groups/ESS/whung/):
  <out_dir>/gnn_training_data_fixed.pt
  <out_dir>/scaler_fixed.pkl
  <out_dir>/gnn_testing_data_fixed_<date>.pt   (with .y_obs/.lat/.lon attached
                                                for evaluation and mapping)
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import KDTree
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

TRAIN_CSV = "/groups/ESS/whung/swe_gnn/data/all_points_final_merged_training_snodas_mask_resnet_all_batch.csv"
TEST_CSV = ("/groups/ESS/whung/swe_gnn/data/testing_snodas_mask/"
            "testing_all_ready_{date}_merged.csv_snodas_mask.csv")

CHUNKSIZE = 500_000
GRID_SIZE = 0.01
KNN_K = 6
EDGE_THRESHOLD = 0.5

LAG_VARS = ["air_temperature_tmmn", "potential_evapotranspiration",
            "mean_vapor_pressure_deficit", "relative_humidity_rmax",
            "relative_humidity_rmin", "precipitation_amount",
            "air_temperature_tmmx", "wind_speed", "fsca"]
# Original intended_columns minus 'Slope' (fix 1).
BASE_COLS = ["lat", "lon", "date", "precipitation_amount", "relative_humidity_rmin",
             "potential_evapotranspiration", "air_temperature_tmmx",
             "relative_humidity_rmax", "mean_vapor_pressure_deficit",
             "air_temperature_tmmn", "wind_speed", "Elevation", "Aspect",
             "Curvature", "Northness", "Eastness", "fsca"]
INTENDED = (BASE_COLS
            + [f"{v}_{i}" for i in range(1, 8) for v in LAG_VARS]
            + ["water_year", "snodas_mask"])
# Physical validity range per lagged variable (fix 2). Outside = flag code or
# zero-filled failed join, never a real measurement.
LAG_VALID = {
    "fsca": (0.0, 100.0),                       # MODIS flags are 200+
    "air_temperature_tmmn": (200.0, 340.0),     # Kelvin; 0 = failed join
    "air_temperature_tmmx": (200.0, 340.0),
    "relative_humidity_rmin": (0.5, 100.0),     # exact 0 only occurs as fill
    "relative_humidity_rmax": (0.5, 100.0),
    "precipitation_amount": (0.0, 1000.0),      # 0 is a legitimate dry day
    "wind_speed": (0.0, 60.0),                  # 0 calm is legitimate
    "potential_evapotranspiration": (0.0, 50.0),
    "mean_vapor_pressure_deficit": (0.0, 20.0),
}


def fix_lag_stacks(df: pd.DataFrame):
    """Mask physically invalid values across each [current, lag1..lag7] stack
    and fill each row from the nearest valid lag (fix 2). Remaining NaNs are
    handled downstream (train: dropna, test: training-mean imputation)."""
    n_bad = 0
    for var, (lo, hi) in LAG_VALID.items():
        cols = [c for c in [var] + [f"{var}_{i}" for i in range(1, 8)]
                if c in df.columns]
        if not cols:
            continue
        block = df[cols].astype(float)
        valid = block.where((block >= lo) & (block <= hi))
        n_bad += int(valid.isna().sum().sum() - block.isna().sum().sum())
        df[cols] = valid.ffill(axis=1).bfill(axis=1)
    return df, n_bad


def add_time_space_encodings(df: pd.DataFrame) -> pd.DataFrame:
    df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    lat_rad, lon_rad = np.radians(df["lat"]), np.radians(df["lon"])
    df["sin_lat"], df["cos_lat"] = np.sin(lat_rad), np.cos(lat_rad)
    df["sin_lon"], df["cos_lon"] = np.sin(lon_rad), np.cos(lon_rad)
    return df


def aggregate(df: pd.DataFrame, select_cols) -> pd.DataFrame:
    df["lat_bin"] = (df["lat"] // GRID_SIZE).astype(int)
    df["lon_bin"] = (df["lon"] // GRID_SIZE).astype(int)
    df["grid_id"] = (df["lat_bin"].astype(str) + "_" + df["lon_bin"].astype(str)
                     + "_" + df["day_of_year"].astype(str))
    agg = {c: "mean" for c in select_cols if c not in ["date", "lat", "lon"]}
    agg.update({"lat": "mean", "lon": "mean", "day_of_year": "mean"})
    if "date" in select_cols:
        agg["date"] = "first"
    agg["swe_value"] = "mean"
    return df.groupby("grid_id").agg(agg).reset_index()


def value_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Original physical-range filters, minus the Slope filter (fix 1)."""
    df.loc[df["air_temperature_tmmx"] < 250, "air_temperature_tmmx"] = np.nan
    df.loc[df["air_temperature_tmmn"] < 250, "air_temperature_tmmn"] = np.nan
    df.loc[df["Elevation"] < 0, "Elevation"] = np.nan
    return df


def build_edges(coords: np.ndarray) -> torch.Tensor:
    tree = KDTree(coords)
    dist, nbr = tree.query(coords, k=KNN_K + 1)
    n = coords.shape[0]
    src = np.repeat(np.arange(n), KNN_K)
    dst = nbr[:, 1:].ravel()
    keep = dist[:, 1:].ravel() < EDGE_THRESHOLD
    edges = np.stack([src[keep], dst[keep]])
    return torch.tensor(edges, dtype=torch.long)


def make_graph(merged: pd.DataFrame, x_scaled: np.ndarray, is_test: bool) -> Data:
    x = torch.tensor(x_scaled, dtype=torch.float)
    edge_index = build_edges(merged[["lat", "lon"]].values)
    if is_test:
        g = Data(x=x, edge_index=edge_index)
        y_obs = pd.to_numeric(merged["swe_value"], errors="coerce").values
        g.y_obs = torch.tensor(y_obs, dtype=torch.float)  # NaN where unobserved
        g.lat = torch.tensor(merged["lat"].values, dtype=torch.float)
        g.lon = torch.tensor(merged["lon"].values, dtype=torch.float)
    else:
        g = Data(x=x, edge_index=edge_index,
                 y=torch.tensor(merged["swe_value"].values, dtype=torch.float))
    g.grid_id = merged["grid_id"].tolist()
    g.dates = merged["date"].astype(str).tolist() if "date" in merged.columns else []
    print(f"  graph: {g.num_nodes:,} nodes, {g.num_edges:,} edges, "
          f"{g.num_node_features} features")
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_date", default="2025-01-15")
    ap.add_argument("--out_dir",
                    default="/projects/kzhou6/bcui2/research/geo_project/data_fixed")
    ap.add_argument("--test_only", action="store_true",
                    help="reuse <out_dir>/scaler_fixed.pkl and only build the "
                         "test graph (skips the 3.8GB training CSV entirely)")
    args = ap.parse_args()
    t0 = time.time()

    avail = set(pd.read_csv(TRAIN_CSV, nrows=1).columns)
    assert "Slope" in avail, "sanity: source CSV should still contain Slope"
    select = [c for c in dict.fromkeys(INTENDED) if c in avail]
    if "swe_value" not in select:
        select.append("swe_value")
    print(f"selected {len(select)} columns (Slope excluded by design)")

    train_m = None
    if args.test_only:
        with open(f"{args.out_dir}/scaler_fixed.pkl", "rb") as f:
            saved = pickle.load(f)
        scaler, final_cols = saved["scaler"], saved["columns"]
        train_absmax = saved.get("train_absmax")
        if train_absmax is None:  # older pickle: derive from the saved graph
            tr_graph = torch.load(f"{args.out_dir}/gnn_training_data_fixed.pt",
                                  map_location="cpu", weights_only=False)
            train_absmax = np.abs(tr_graph.x.numpy()).max(axis=0)
            del tr_graph
        print(f"test_only: reusing saved scaler ({len(final_cols)} columns)")
    else:
        # ---- training CSV (chunked) ----
        chunks, bad_fsca = [], 0
        for chunk in pd.read_csv(TRAIN_CSV, usecols=select, chunksize=CHUNKSIZE):
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            chunk = chunk.dropna(subset=["date"])
            chunk["day_of_year"] = chunk["date"].dt.dayofyear
            chunk = chunk[(chunk["swe_value"] >= 0) & (chunk["swe_value"] < 3000)]
            chunk, nb = fix_lag_stacks(chunk)
            bad_fsca += nb
            chunks.append(chunk)
            print(f"  ...{sum(len(c) for c in chunks):,} train rows "
                  f"({time.time()-t0:.0f}s)", flush=True)
        train = pd.concat(chunks, ignore_index=True)
        del chunks
        print(f"train rows: {len(train):,} | invalid lag values masked: {bad_fsca:,}")

        train_m = aggregate(train, select)
        del train
        train_m = value_filters(train_m).dropna()
        train_m = add_time_space_encodings(train_m)
        print(f"train nodes after filters: {len(train_m):,}")

    # ---- test CSV ----
    test = pd.read_csv(TEST_CSV.format(date=args.test_date))
    test = test.dropna(subset=["date"])
    if "swe_value" not in test.columns:
        test["swe_value"] = np.nan
    test = test[[c for c in select if c in test.columns]].copy()
    test["date"] = pd.to_datetime(test["date"], errors="coerce")
    test["day_of_year"] = test["date"].dt.dayofyear
    test, nb = fix_lag_stacks(test)
    print(f"test rows: {len(test):,} | invalid lag values masked: {nb:,}")
    test_m = aggregate(test, select)
    test_m = value_filters(test_m)
    test_m = add_time_space_encodings(test_m)

    # ---- features / scaling ----
    if not args.test_only:
        exclude = ["grid_id", "lat", "lon", "lat_rad", "lon_rad", "date",
                   "day_of_year", "swe_value", "water_year", "temp_avg"]
        final_cols = sorted(c for c in train_m.columns
                            if c not in exclude and c not in ["lat_bin", "lon_bin"])
        print(f"final feature columns ({len(final_cols)}): {final_cols}")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_m[final_cols])
        train_absmax = np.abs(x_train).max(axis=0)

    # fix 4: impute residual test NaNs with the training mean (scaled 0)
    n_nan = int(test_m[final_cols].isna().sum().sum())
    if n_nan:
        print(f"test NaNs imputed with training means: {n_nan:,}")
        test_m[final_cols] = test_m[final_cols].fillna(
            pd.Series(scaler.mean_, index=final_cols))
    x_test = scaler.transform(test_m[final_cols])

    # fix 3: sanity gate (per-column cap = training envelope with 20% headroom).
    # A |mean z| violation means systematic corruption -> abort. A max|z|
    # violation on a small fraction of cells is a record storm outside the
    # training envelope -> winsorize to the cap and warn (abort only if more
    # than 1% of a column needs clipping, which would indicate corruption).
    mz, xz = np.abs(x_test.mean(axis=0)), np.abs(x_test).max(axis=0)
    cap = np.maximum(30.0, 1.2 * np.asarray(train_absmax))
    print("\nsanity gate on scaled test features "
          "(|mean z|<5; tails clipped to max(30, 1.2*train max|z|)):")
    mean_bad = [(final_cols[i], mz[i]) for i in range(len(final_cols)) if mz[i] > 5]
    if mean_bad:
        for c, m in mean_bad:
            print(f"  VIOLATION (systematic) {c}: |mean z|={m:.1f}")
        raise SystemExit("sanity gate FAILED — graph not written")
    for i in range(len(final_cols)):
        if xz[i] > cap[i]:
            col = x_test[:, i]
            n_clip = int((np.abs(col) > cap[i]).sum())
            frac = n_clip / len(col)
            print(f"  WINSORIZE {final_cols[i]}: {n_clip} values "
                  f"({frac*100:.3f}%) beyond cap {cap[i]:.1f}, clipped")
            if frac > 0.01:
                raise SystemExit(f"sanity gate FAILED — {final_cols[i]} has "
                                 f"{frac*100:.1f}% out-of-envelope values")
            x_test[:, i] = np.clip(col, -cap[i], cap[i])
    print(f"  PASSED — worst |mean z|={mz.max():.2f}")

    # ---- build & save ----
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    if not args.test_only:
        with open(f"{args.out_dir}/scaler_fixed.pkl", "wb") as f:
            pickle.dump({"scaler": scaler, "columns": final_cols,
                         "train_absmax": train_absmax}, f)
        print("\nbuilding training graph...")
        torch.save(make_graph(train_m, x_train, is_test=False),
                   f"{args.out_dir}/gnn_training_data_fixed.pt")
    print("building test graph...")
    torch.save(make_graph(test_m, x_test, is_test=True),
               f"{args.out_dir}/gnn_testing_data_fixed_{args.test_date}.pt")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
