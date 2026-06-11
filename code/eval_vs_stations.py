#!/usr/bin/env python
"""
Evaluate SWE predictions against SNOTEL station observations for one test date,
for both pipelines:
  broken: Wei-Ting's original test graph (85 features, Slope/fsca/lag bugs)
          + our elv_hyps0 model trained on the original training graph
  fixed : data_fixed graphs (84 features, repaired lags, no Slope)
          + the fixed_rmse_baseline model

Station truth comes from the paired-data file
  testing_all_ready_<date>_paireddata_nopred.csv_snodas_mask.csv
matched to graph nodes via the same grid_id formula used in data_process.

Usage:
  python eval_vs_stations.py [--date 2025-01-15]
  # keep a record:  python eval_vs_stations.py | tee .../sweep_results/station_eval_<date>.log
"""
import argparse

import numpy as np
import pandas as pd
import torch

from model_train import GCN_Model

ROOT = "/projects/kzhou6/bcui2/research/geo_project"
OBS_CSV = ("/groups/ESS/whung/swe_gnn/data/testing_snodas_mask/"
           "testing_all_ready_{date}_paireddata_nopred.csv_snodas_mask.csv")

PIPELINES = {
    "broken": {
        "graph": "/groups/ESS/whung/swe_gnn/data/gnn_testing_data_{date}.pt",
        "model": f"{ROOT}/sweep_results/elv_hyps0/GCN_model.pth",
        "in_channels": 85,
    },
    "fixed": {
        "graph": f"{ROOT}/data_fixed/gnn_testing_data_fixed_{{date}}.pt",
        "model": f"{ROOT}/sweep_results/fixed_rmse_baseline/GCN_model.pth",
        "in_channels": 84,
    },
}


def evaluate(name: str, cfg: dict, obs: pd.DataFrame, date: str) -> None:
    te = torch.load(cfg["graph"].format(date=date), map_location="cpu",
                    weights_only=False)
    m = GCN_Model(cfg["in_channels"], 64, 1)
    m.load_state_dict(torch.load(cfg["model"], map_location="cpu"))
    m.eval()
    with torch.no_grad():
        pred = m(te).numpy()
    pred_by_grid = pd.Series(pred, index=te.grid_id)

    d = obs.copy()
    d["pred"] = d["grid_id"].map(pred_by_grid)
    d = d.dropna(subset=["pred", "swe_value"])
    y, p = d["swe_value"].values, d["pred"].values
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    r2 = float(1 - np.var(y - p) / np.var(y))
    corr = float(np.corrcoef(y, p)[0, 1])
    print(f"{name:7s} vs {len(d)} stations: RMSE={rmse:6.2f} in  R2={r2:7.4f}  "
          f"corr={corr:6.4f} | pred p50={np.percentile(p,50):5.2f} "
          f"p95={np.percentile(p,95):6.2f} max={p.max():6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2025-01-15")
    args = ap.parse_args()

    obs = pd.read_csv(OBS_CSV.format(date=args.date))
    obs["grid_id"] = ((obs["lat"] // 0.01).astype(int).astype(str) + "_"
                      + (obs["lon"] // 0.01).astype(int).astype(str) + "_"
                      + obs["day_of_year"].astype(int).astype(str))
    y = obs["swe_value"]
    print(f"date {args.date}: {len(obs)} station obs | "
          f"obs p50={y.median():.2f} p95={y.quantile(.95):.2f} max={y.max():.2f} in")

    for name, cfg in PIPELINES.items():
        try:
            evaluate(name, cfg, obs, args.date)
        except FileNotFoundError as e:
            print(f"{name:7s} skipped (missing file: {e.filename})")


if __name__ == "__main__":
    main()
