"""
lstm_trainer.py — PyTorch LSTM challenger
==========================================
Trains the LSTM on exactly the same engine-level split and seed as the
XGBoost model, so the two validation RMSEs are directly comparable, then
evaluates on the 100 test engines, logs everything to MLflow and (optionally)
registers it as  challenger-lstm.

  python lstm_trainer.py                       # auto device (cuda if available)
  python lstm_trainer.py --device cpu --epochs 20
  python lstm_trainer.py --register --fail-above-rmse 18

Why test a sequence model at all? XGBoost works from 183 hand-built features.
The LSTM sees only the raw sensor windows. If it wins, the feature engineering
was leaving signal on the table; if it doesn't, that is evidence the
interpretable model is the right choice, not just an opinion in the README.
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATASET, INFORMATIVE_SENSORS, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_LEARNING_RATE,
    LSTM_PATIENCE, LSTM_PREDICTOR_DIR, LSTM_RMSE_GATE, MAX_RUL, MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, RANDOM_SEED, VAL_FRACTION,
)
from data_processor import add_rul_labels, load_raw, load_rul_ground_truth
from lstm_model import add_cycle_norm, build_network, build_windows, input_columns, lstm_spec
from model_trainer import _git_sha, engine_split, nasa_score, raw_input_example
from rul_predictor import log_package, save_package


def resolve_torch_device(device: str | None) -> str:
    import torch

    device = device or os.getenv("PROGNOSTIC_DEVICE", "auto")
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int):
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(spec: dict):
    """Train windows (with engine ids), the fitted scaler, and the last test window per engine."""
    train_df = add_rul_labels(load_raw("train")).sort_values(["unit", "cycle"]).reset_index(drop=True)
    cols = input_columns(spec)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(add_cycle_norm(train_df, spec)[cols].values)
    windows, _ = build_windows(train_df, X_train, spec["sequence_length"])
    y = (train_df["rul"].values / spec["max_rul"]).astype(np.float32)
    groups = train_df["unit"].values

    test_df = load_raw("test").sort_values(["unit", "cycle"]).reset_index(drop=True)
    X_test = scaler.transform(add_cycle_norm(test_df, spec)[cols].values)
    test_windows, _ = build_windows(test_df, X_test, spec["sequence_length"])
    last_idx = test_df.groupby("unit", sort=True).tail(1).index.values
    return windows, y, groups, scaler, test_windows[last_idx], load_rul_ground_truth()


def _predict(net, windows: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    import torch

    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            out.append(net(torch.from_numpy(windows[i:i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out)


def train_lstm(epochs: int | None = None,
               device: str | None = None,
               log_to_mlflow: bool = True,
               register_model: bool = False,
               run_name: str | None = None,
               rmse_gate: float = LSTM_RMSE_GATE,
               hidden_size: int | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    epochs = epochs or LSTM_EPOCHS
    device = resolve_torch_device(device)
    _seed_everything(RANDOM_SEED)
    spec = lstm_spec()
    if hidden_size:
        spec["hidden_size"] = hidden_size

    print("=" * 58)
    print("  PrognosticAI — LSTM RUL Training (challenger)")
    print(f"  Dataset: NASA C-MAPSS {DATASET}  |  device: {device}")
    print("=" * 58)

    windows, y, groups, scaler, X_test, y_test = prepare_data(spec)
    tr_idx, val_idx = engine_split(np.zeros(len(y)), y, groups)   # same engines as XGBoost
    n_val_engines = len(np.unique(groups[val_idx]))
    print(f"Windows: {windows.shape}  |  Train: {len(tr_idx):,}  |  Val: {len(val_idx):,} "
          f"({n_val_engines} held-out engines)")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(windows[tr_idx]), torch.from_numpy(y[tr_idx])),
        batch_size=LSTM_BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    net = build_network(spec).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=LSTM_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=5)
    loss_fn = torch.nn.MSELoss()

    best_rmse, best_epoch, best_state, curve = float("inf"), 0, None, []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimiser.step()

        val_pred = _predict(net, windows[val_idx], device) * spec["max_rul"]
        val_rmse = float(np.sqrt(mean_squared_error(y[val_idx] * spec["max_rul"], val_pred)))
        scheduler.step(val_rmse)
        curve.append(val_rmse)
        if val_rmse < best_rmse - 1e-4:
            best_rmse, best_epoch = val_rmse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:>3}  val RMSE {val_rmse:6.2f}  (best {best_rmse:.2f} @ {best_epoch})")
        if epoch - best_epoch >= LSTM_PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break
    train_seconds = time.time() - t0

    net.load_state_dict(best_state)
    net = net.cpu()
    y_pred = (_predict(net, X_test, "cpu") * spec["max_rul"]).clip(0)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    s = nasa_score(y_test, y_pred)

    print("\n" + "─" * 44)
    print(f"  Val RMSE    : {best_rmse:.2f} cycles  (held-out training engines, epoch {best_epoch})")
    print(f"  Test RMSE   : {rmse:.2f} cycles  (official test set)")
    print(f"  MAE         : {mae:.2f}  cycles")
    print(f"  NASA Score S: {s:.1f}")
    print("─" * 44)

    example = raw_input_example()
    save_package("lstm", net, scaler, spec, example, LSTM_PREDICTOR_DIR)
    print(f"Packaged model saved → {LSTM_PREDICTOR_DIR}")

    metrics = {
        "val_rmse": best_rmse,
        "test_rmse": rmse,
        "test_mae": mae,
        "nasa_score": s,
        "best_epoch": best_epoch,
        "epochs_run": len(curve),
        "train_seconds": train_seconds,
    }
    if log_to_mlflow:
        predictions = pd.DataFrame({"engine": np.arange(1, len(y_test) + 1), "true_rul": y_test,
                                    "pred_rul": y_pred, "abs_error": np.abs(y_test - y_pred)})
        metrics["mlflow_run_id"] = _log_to_mlflow(
            net, scaler, spec, example, metrics, curve, predictions, device,
            n_val_engines, register_model, run_name, rmse_gate,
        )
    return metrics


def _log_to_mlflow(net, scaler, spec, example, metrics, curve, predictions, device,
                   n_val_engines, register_model, run_name, rmse_gate) -> str:
    import mlflow
    import torch

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "dataset": f"C-MAPSS {DATASET}",
            "split_strategy": "engine (GroupShuffleSplit)",
            "model_type": "lstm",
            "git_sha": _git_sha(),
            "runner": "github-actions" if os.getenv("GITHUB_ACTIONS") else
                      "kaggle" if os.getenv("KAGGLE_KERNEL_RUN_TYPE") else "local",
            "torch_version": torch.__version__,
        })
        mlflow.log_params({
            "device": device,
            "sequence_length": spec["sequence_length"],
            "hidden_size": spec["hidden_size"],
            "num_layers": spec["num_layers"],
            "dropout": spec["dropout"],
            "batch_size": LSTM_BATCH_SIZE,
            "learning_rate": LSTM_LEARNING_RATE,
            "patience": LSTM_PATIENCE,
            "max_rul_cap": MAX_RUL,
            "n_inputs": len(INFORMATIVE_SENSORS) + 1,
            "val_fraction": VAL_FRACTION,
            "n_val_engines": n_val_engines,
            "seed": RANDOM_SEED,
        })
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        for epoch, v in enumerate(curve, start=1):
            mlflow.log_metric("val_rmse_curve", v, step=epoch)
        with tempfile.TemporaryDirectory() as tmp:
            predictions.to_csv(os.path.join(tmp, "test_predictions.csv"), index=False)
            mlflow.log_artifacts(tmp, artifact_path="evaluation")
        log_package("lstm", net, scaler, spec, example)
        print(f"\nMLflow run logged → {run.info.run_id}  ({MLFLOW_TRACKING_URI})")

    if register_model:
        from registry import register_run
        register_run(run.info.run_id, "lstm", metrics, rmse_gate)
    return run.info.run_id


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the PrognosticAI LSTM challenger")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None, help="auto / cuda / cpu / mps")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--register", action="store_true", help="register as challenger-lstm")
    p.add_argument("--run-name", default=None)
    p.add_argument("--metrics-out", default=None)
    p.add_argument("--fail-above-rmse", type=float, default=None)
    args = p.parse_args(argv)

    metrics = train_lstm(epochs=args.epochs, device=args.device, log_to_mlflow=not args.no_mlflow,
                         register_model=args.register, run_name=args.run_name,
                         rmse_gate=args.fail_above_rmse or LSTM_RMSE_GATE)
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
    if args.fail_above_rmse is not None and metrics["test_rmse"] > args.fail_above_rmse:
        print(f"\nQUALITY GATE FAILED: test RMSE {metrics['test_rmse']:.2f} > {args.fail_above_rmse:.2f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
