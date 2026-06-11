#!/usr/bin/env python
"""
SWELoss weight sensitivity-test driver.

Non-invasive wrapper around Wei-Ting's model_train.py: it imports the models,
loss, loaders and training loop from there and runs ONE configuration of
(base_weight, hypsometry_weight, curve weights) into its own output directory.

A launcher (run_sweep.sh) submits many of these as separate slurm jobs so the
ELV / TEMP curve families and the hypsometry-weight grid are explored in
parallel without overwriting each other.

Example:
  python run_sensitivity.py \
      --data_path /groups/ESS/whung/swe_gnn/data/gnn_training_data.pt \
      --model GCN --curve elv \
      --base_weight 1 --hypsometry_weight 1000 \
      --epochs 100 --out_dir /projects/.../sweep_results \
      --run_name elv_hyps1000 --print_every 200
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import torch
from sklearn.preprocessing import LabelEncoder

# Reuse everything from Wei-Ting's training script (importing does NOT run main())
import model_train as mt
from model_train import (
    SWELoss,
    GCN_Model,
    GraphSAGE_Model,
    EnhancedGraphSAGE_Model,
    GraphSAGE_v2,
    SimplifiedGraphSAGE_Model,
    build_loaders,
    split_masks,
    train as train_loop,
    eval_step,
    save_model,
    CONFIG,
)

MODELS = {
    "GCN": GCN_Model,
    "GraphSAGE": GraphSAGE_Model,
    "EnhancedGraphSAGE": EnhancedGraphSAGE_Model,
    "GraphSAGE_v2": GraphSAGE_v2,
    "SimplifiedGraphSAGE": SimplifiedGraphSAGE_Model,
}

# Feature-column indices into the SORTED 85-col feature matrix (see data_process.py).
# Verified: idx 3 = Elevation, 6 = air_temperature_tmmn, 49 = precipitation_amount.
ELEV_IDX = 3
TEMP_IDX = 6
PRECIP_IDX = 49


class LoggingSWELoss(SWELoss):
    """SWELoss that records base/phys magnitudes and prints them at an interval.

    Reproduces the 'turn on lines 300-302' diagnostic Wei-Ting mentioned, while
    swallowing the parent's per-step hypsometry print so logs stay readable.
    """

    def __init__(self, *args, print_every: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_every = int(print_every)
        self._step = 0
        self.base_sum = 0.0
        self.phys_sum = 0.0
        self.n = 0

    def forward(self, y_pred, y_true, batch=None):
        base = torch.sqrt(self._mse(y_pred, y_true))
        phys = torch.tensor(0.0, device=y_pred.device)
        if batch is not None and self.hypsometry_weight > 0.0:
            try:
                seeds = batch.batch_size if hasattr(batch, "batch_size") else y_pred.shape[0]
                # swallow the parent method's per-step stdout spam
                with contextlib.redirect_stdout(io.StringIO()):
                    phys = self._hypsometry_discrepancy(y_pred, y_true, batch, seeds)
            except Exception as e:  # noqa: BLE001
                if not self._warned_missing_context:
                    print(f"[SWELoss] phys term disabled this step due to: {e}", flush=True)
                    self._warned_missing_context = True

        b = float(base.detach())
        p = float(phys.detach()) if torch.is_tensor(phys) else float(phys)
        self._step += 1
        self.base_sum += b
        self.phys_sum += p
        self.n += 1
        if self.print_every and self._step % self.print_every == 0:
            print(
                f"[loss] step {self._step}: base={b:.4f} phys={p:.6f} | "
                f"w*base={self.base_weight * b:.4f} w*phys={self.hypsometry_weight * p:.4f}",
                flush=True,
            )
        return self.base_weight * base + self.hypsometry_weight * phys

    def mean_components(self):
        if self.n == 0:
            return 0.0, 0.0
        return self.base_sum / self.n, self.phys_sum / self.n


def load_graph(data_path: str, device: torch.device):
    graph = torch.load(data_path, map_location=device, weights_only=False)
    if not isinstance(graph.x, torch.Tensor):
        graph.x = torch.tensor(graph.x, dtype=torch.float)
    if not isinstance(graph.edge_index, torch.Tensor):
        graph.edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    if not isinstance(graph.y, torch.Tensor):
        graph.y = torch.tensor(graph.y, dtype=torch.float)
    graph = graph.to(device)
    # encode grid_id like main() does, drop the string dates list
    if hasattr(graph, "grid_id") and not torch.is_tensor(graph.grid_id):
        enc = LabelEncoder().fit_transform(graph.grid_id)
        graph.grid_id = torch.tensor(enc, dtype=torch.long)
    if "dates" in graph._store:
        del graph._store["dates"]
    return graph


def main():
    ap = argparse.ArgumentParser(description="SWELoss weight sensitivity run (single config)")
    ap.add_argument("--data_path", default=CONFIG["data_path"])
    ap.add_argument("--model", default="GCN", choices=list(MODELS.keys()))
    ap.add_argument("--curve", default="elv", choices=["elv", "temp", "both"],
                    help="which physical curve(s) the hypsometry term uses")
    ap.add_argument("--base_weight", type=float, default=1.0)
    ap.add_argument("--hypsometry_weight", type=float, default=1000.0)
    ap.add_argument("--num_bins", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=CONFIG["training_params"]["epochs"])
    ap.add_argument("--batch_size", type=int, default=CONFIG["training_params"]["batch_size"])
    ap.add_argument("--lr", type=float, default=CONFIG["training_params"]["learning_rate"])
    ap.add_argument("--weight_decay", type=float, default=CONFIG["training_params"]["weight_decay"])
    ap.add_argument("--out_dir", required=True, help="parent results dir; a run_name subdir is created")
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--print_every", type=int, default=200, help="print base/phys every N steps (0=off)")
    ap.add_argument("--seed", type=int, default=CONFIG["training_params"]["seed"])
    ap.add_argument("--feature_names_pkl", default=None,
                    help="pickle with {'columns': [...]} (e.g. data_fixed/scaler_fixed.pkl); "
                         "if given, elevation/temperature/precipitation indices are derived "
                         "from it instead of the legacy 85-column constants")
    args = ap.parse_args()

    elev_idx, temp_idx, precip_idx = ELEV_IDX, TEMP_IDX, PRECIP_IDX
    if args.feature_names_pkl:
        import pickle
        with open(args.feature_names_pkl, "rb") as f:
            cols = pickle.load(f)["columns"]
        elev_idx = cols.index("Elevation")
        temp_idx = cols.index("air_temperature_tmmn")
        precip_idx = cols.index("precipitation_amount")
        print(f"feature indices from {args.feature_names_pkl}: "
              f"elev={elev_idx} temp={temp_idx} precip={precip_idx}")

    # curve weights
    elev_w = 1.0 if args.curve in ("elv", "both") else 0.0
    temp_w = 1.0 if args.curve in ("temp", "both") else 0.0

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"=== run {args.run_name} | model={args.model} curve={args.curve} "
          f"base={args.base_weight} hyps={args.hypsometry_weight} "
          f"(elev_w={elev_w}, temp_w={temp_w}) | device={device} ===", flush=True)

    graph = load_graph(args.data_path, device)
    print(f"graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
          f"{graph.num_node_features} features", flush=True)

    split_masks(graph, train_ratio=CONFIG["training_params"]["train_ratio"],
                val_ratio=CONFIG["training_params"]["val_ratio"], seed=args.seed)

    in_channels = graph.num_node_features
    hidden = CONFIG["model_params"]["hidden_channels"]
    out_channels = CONFIG["model_params"]["out_channels"]
    model = MODELS[args.model](in_channels=in_channels, hidden_channels=hidden,
                               out_channels=out_channels).to(device)

    # neighbors per hop, same convention as main()
    nbm = CONFIG["training_params"].get("neighbors_by_model", {})
    default_nb = CONFIG["training_params"].get("num_neighbors", [16, 16])
    if args.model in nbm:
        num_neighbors = nbm[args.model]
    else:
        from torch_geometric.nn import SAGEConv, GCNConv
        L = sum(isinstance(m, (SAGEConv, GCNConv)) for m in model.modules())
        base_nb = default_nb[0] if isinstance(default_nb, list) and default_nb else 16
        num_neighbors = default_nb if L <= 0 else [base_nb] * L
    print(f"num_neighbors (hops): {num_neighbors}", flush=True)

    train_loader, val_loader, test_loader = build_loaders(
        data=graph, num_neighbors=num_neighbors, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_fn = LoggingSWELoss(
        base_weight=args.base_weight,
        hypsometry_weight=args.hypsometry_weight,
        elevation_feature_index=elev_idx,
        temperature_feature_index=temp_idx,
        precipitation_feature_index=precip_idx,
        elevation_curve_weight=elev_w,
        temperature_curve_weight=temp_w,
        precipitation_curve_weight=0.0,
        area_feature_index=None,
        num_bins=args.num_bins,
        print_every=args.print_every,
    )

    t0 = time.time()
    history = train_loop(model=model, train_loader=train_loader, val_loader=val_loader,
                         optimizer=optimizer, loss_fn=loss_fn, epochs=args.epochs, device=device)
    train_secs = time.time() - t0

    test_metrics = eval_step(model, test_loader, loss_fn, device, args.model,
                             str(run_dir / f"{args.model}_result_comparison.png"), plot_option=True)
    mean_base, mean_phys = loss_fn.mean_components()

    print(f"[{args.run_name}] TEST  R2={test_metrics['r2']:.4f}  RMSE={test_metrics['rmse']:.4f}  "
          f"mean_base={mean_base:.4f}  mean_phys={mean_phys:.6f}  "
          f"w*phys={args.hypsometry_weight*mean_phys:.4f}  train_secs={train_secs:.0f}", flush=True)

    save_model(model, str(run_dir / f"{args.model}_model.pth"))

    result = {
        "run_name": args.run_name,
        "model": args.model,
        "curve": args.curve,
        "base_weight": args.base_weight,
        "hypsometry_weight": args.hypsometry_weight,
        "elevation_curve_weight": elev_w,
        "temperature_curve_weight": temp_w,
        "num_bins": args.num_bins,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "data_path": args.data_path,
        "test_r2": test_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "final_val_r2": history["val_r2"][-1] if history["val_r2"] else None,
        "final_train_r2": history["train_r2"][-1] if history["train_r2"] else None,
        "mean_base": mean_base,
        "mean_phys": mean_phys,
        "weighted_phys": args.hypsometry_weight * mean_phys,
        "train_seconds": train_secs,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{args.run_name}] wrote {run_dir/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
