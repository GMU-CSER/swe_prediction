#!/usr/bin/env python
"""
Evaluate a model trained on the FIXED training graph against the FIXED test
graph for one date: prediction distribution, R2/RMSE vs observed swe_value
(where available), and a prediction map png.

Usage:
  python eval_fixed.py --model <path/GCN_model.pth> [--date 2025-01-15]
                       [--data_dir .../data_fixed] [--out_dir .../sweep_results]
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_train import GCN_Model


def stats(p: np.ndarray, name: str) -> None:
    print(f"{name:30s} min={p.min():8.2f} p50={np.percentile(p, 50):7.2f} "
          f"p95={np.percentile(p, 95):7.2f} max={p.max():8.2f} "
          f"| frac<0.1in: {(p < 0.1).mean()*100:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--date", default="2025-01-15")
    ap.add_argument("--data_dir",
                    default="/projects/kzhou6/bcui2/research/geo_project/data_fixed")
    ap.add_argument("--out_dir",
                    default="/projects/kzhou6/bcui2/research/geo_project/sweep_results")
    args = ap.parse_args()

    tr = torch.load(f"{args.data_dir}/gnn_training_data_fixed.pt",
                    map_location="cpu", weights_only=False)
    te = torch.load(f"{args.data_dir}/gnn_testing_data_fixed_{args.date}.pt",
                    map_location="cpu", weights_only=False)
    print(f"fixed train: {tuple(tr.x.shape)} | fixed test: {tuple(te.x.shape)}")

    m = GCN_Model(tr.num_node_features, 64, 1)
    m.load_state_dict(torch.load(args.model, map_location="cpu"))
    m.eval()

    with torch.no_grad():
        p_tr = m(tr)
        vy = torch.var(tr.y, unbiased=False)
        vr = torch.var(tr.y - p_tr, unbiased=False)
        print(f"sanity, full-graph inference on fixed TRAIN graph: "
              f"R2={1-(vr/vy).item():.4f}")
        pred = m(te).numpy()

    stats(pred, f"TEST {args.date} prediction")

    y = te.y_obs.numpy()
    ok = np.isfinite(y) & (y >= 0) & (y < 3000)
    print(f"observed swe_value available on {ok.sum():,}/{len(y):,} test nodes")
    if ok.sum() > 100:
        yo, po = y[ok], pred[ok]
        rmse = float(np.sqrt(np.mean((po - yo) ** 2)))
        r2 = float(1 - np.var(yo - po) / np.var(yo)) if np.var(yo) > 0 else float("nan")
        stats(yo, "  observed (on those nodes)")
        stats(po, "  predicted (on those nodes)")
        print(f"  TEST vs obs:  RMSE={rmse:.3f} inches   R2={r2:.4f}")

    lat, lon = te.lat.numpy(), te.lon.numpy()
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    sc = ax.scatter(lon, lat, c=np.clip(pred, 0, None), s=1, cmap="turbo",
                    vmin=0, vmax=20)
    fig.colorbar(sc, ax=ax, label="predicted SWE (inches)")
    ax.set_title(f"{args.date} SWE prediction — fixed pipeline")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    out_png = f"{args.out_dir}/fixed_prediction_map_{args.date}.png"
    fig.tight_layout(); fig.savefig(out_png, dpi=150)
    print(f"map saved: {out_png}")


if __name__ == "__main__":
    main()
