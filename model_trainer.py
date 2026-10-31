"""
model_trainer.py — XGBoost RUL Predictor (v5)
==============================================
Changes from v4:
  - Trend features (rolling slopes over 10-80 cycles) and longer rolling
    windows, plus shallower trees (depth 5). Chosen with 5-fold engine-level
    cross-validation on the training set only; test RMSE went from ~18.4 to ~12.5
Changes from v3:
  - Every training run is tracked in MLflow: parameters, metrics, the
    validation curve, feature importance, per-engine test predictions and
    the model itself (optionally registered in the model registry)
  - Validation split is by engine (GroupShuffleSplit), not by row
  - Device is configurable (auto / cuda / cpu) so CI can train on CPU
  - CLI flags for quick runs and a CI quality gate (--fail-above-rmse)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from config import (
    DATASET, LEARNING_RATE, MAX_RUL, MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_EXPERIMENT, MLFLOW_REGISTERED_MODEL, MLFLOW_TRACKING_URI,
    MODEL_PATH, N_ESTIMATORS, RANDOM_SEED, RMSE_GATE, RMSE_TARGET,
    ROLLING_WINDOWS, SCALER_PATH, SLOPE_WINDOWS, VAL_FRACTION, XGB_DEVICE,
)
from data_processor import build_test_dataset, build_train_dataset, get_feature_columns


def resolve_device(device: str | None = None) -> str:
    device = device or XGB_DEVICE
    if device != "auto":
        return device
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def model_params(n_estimators: int, learning_rate: float, device: str) -> dict:
    return dict(
        n_estimators          = n_estimators,
        max_depth             = 5,      # was 8; shallower trees generalise better (engine-level CV)
        learning_rate         = learning_rate,
        subsample             = 0.75,
        colsample_bytree      = 0.65,
        min_child_weight      = 20,     # was 8
        gamma                 = 0.1,
        reg_alpha             = 0.2,
        reg_lambda            = 1.5,
        device                = device,
        tree_method           = "hist",
        random_state          = RANDOM_SEED,
        n_jobs                = -1,
        early_stopping_rounds = 80,
    )


def build_model(n_estimators: int = N_ESTIMATORS,
                learning_rate: float = LEARNING_RATE,
                device: str | None = None) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**model_params(n_estimators, learning_rate, resolve_device(device)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Asymmetric PHM'08 scoring function. Lower = better.
    Late predictions (predicted RUL HIGHER than true RUL, so maintenance
    comes too late) are penalised more than early ones.
    """
    diff = y_pred - y_true
    s = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return float(np.sum(s))


def engine_split(X, y, groups, val_fraction=VAL_FRACTION, seed=RANDOM_SEED):
    """Split so that no engine appears in both train and validation."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, val_idx = next(splitter.split(X, y, groups))
    return tr_idx, val_idx


def _git_sha() -> str:
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def train(n_estimators: int | None = None,
          learning_rate: float | None = None,
          device: str | None = None,
          log_to_mlflow: bool = True,
          register_model: bool = False,
          run_name: str | None = None) -> dict:
    """Train, evaluate on the 100 test engines, and (optionally) log to MLflow."""
    n_estimators = n_estimators or N_ESTIMATORS
    learning_rate = learning_rate or LEARNING_RATE
    device = resolve_device(device)

    print("=" * 58)
    print("  PrognosticAI — XGBoost RUL Training (v5)")
    print(f"  Dataset: NASA C-MAPSS {DATASET}  |  device: {device}")
    print("=" * 58)

    X_full, y_full, scaler, groups = build_train_dataset(return_groups=True)
    tr_idx, val_idx = engine_split(X_full, y_full, groups)
    X_tr, y_tr = X_full[tr_idx], y_full[tr_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    n_val_engines = len(np.unique(groups[val_idx]))
    print(f"\nTrain: {len(tr_idx):,} rows  |  Val: {len(val_idx):,} rows "
          f"({n_val_engines} held-out engines)")

    model = build_model(n_estimators, learning_rate, device)
    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
    train_seconds = time.time() - t0

    best = int(model.best_iteration)
    val_rmse = float(model.best_score)
    print(f"\nBest iteration : {best}")
    print(f"Val RMSE (best): {val_rmse:.4f} cycles")
    if best >= n_estimators - 100:
        print("\n⚠  Model hit the ceiling — consider raising n_estimators.")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")

    # ── Test Evaluation ───────────────────────────────────────────────────
    print("\nEvaluating on TEST set (last cycle per engine)...")
    X_test, y_test = build_test_dataset(scaler)
    y_pred = model.predict(X_test).clip(0)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    s = nasa_score(y_test, y_pred)

    print("\n" + "─" * 44)
    print(f"  Val RMSE    : {val_rmse:.2f} cycles  (held-out training engines)")
    print(f"  Test RMSE   : {rmse:.2f} cycles  (official test set)")
    print(f"  MAE         : {mae:.2f}  cycles")
    print(f"  NASA Score S: {s:.1f}")
    print("─" * 44)

    if rmse < 13:
        verdict = "EXCELLENT — matches published deep-learning baselines"
    elif rmse < RMSE_TARGET:
        verdict = f"GOOD — within target (< {RMSE_TARGET:.0f} cycles)"
    elif rmse < 20:
        verdict = "ACCEPTABLE — operationally useful, close to good range"
    else:
        verdict = "BELOW TARGET"
    print(f"  Verdict     : {verdict}")
    print("─" * 44)

    feat_names = get_feature_columns()
    importance = pd.DataFrame(
        {"feature": feat_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\n  Top 10 features (% contribution):")
    for rank, row in enumerate(importance.head(10).itertuples(), 1):
        print(f"  {rank:>2}. {row.feature:<20} {row.importance * 100:5.1f}%")

    predictions = pd.DataFrame({
        "engine": np.arange(1, len(y_test) + 1),
        "true_rul": y_test,
        "pred_rul": y_pred,
        "abs_error": np.abs(y_test - y_pred),
    })

    metrics = {
        "val_rmse": val_rmse,
        "test_rmse": rmse,
        "test_mae": mae,
        "nasa_score": s,
        "best_iteration": best,
        "train_seconds": train_seconds,
    }

    if log_to_mlflow:
        metrics["mlflow_run_id"] = _log_to_mlflow(
            model=model, params=model_params(n_estimators, learning_rate, device),
            metrics=metrics, importance=importance, predictions=predictions,
            X_example=X_test[:5], n_features=X_full.shape[1],
            n_val_engines=n_val_engines, register_model=register_model,
            run_name=run_name,
        )
    return metrics


def _log_to_mlflow(model, params, metrics, importance, predictions, X_example,
                   n_features, n_val_engines, register_model, run_name) -> str:
    import mlflow
    import mlflow.xgboost
    from mlflow.entities import Metric

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "dataset": f"C-MAPSS {DATASET}",
            "split_strategy": "engine (GroupShuffleSplit)",
            "git_sha": _git_sha(),
            "runner": "github-actions" if os.getenv("GITHUB_ACTIONS") else
                      "kaggle" if os.getenv("KAGGLE_KERNEL_RUN_TYPE") else "local",
        })
        mlflow.log_params({k: v for k, v in params.items()})
        mlflow.log_params({
            "max_rul_cap": MAX_RUL,
            "rolling_windows": ",".join(map(str, ROLLING_WINDOWS)),
            "slope_windows": ",".join(map(str, SLOPE_WINDOWS)),
            "n_features": n_features,
            "val_fraction": VAL_FRACTION,
            "n_val_engines": n_val_engines,
        })
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        # Validation curve: one batch call instead of thousands of single calls
        curve = model.evals_result()["validation_0"]["rmse"]
        now = int(time.time() * 1000)
        batch = [Metric("val_rmse_curve", float(v), now, i)
                 for i, v in enumerate(curve) if i % 10 == 0 or i == len(curve) - 1]
        client = mlflow.tracking.MlflowClient()
        for start in range(0, len(batch), 900):
            client.log_batch(run.info.run_id, metrics=batch[start:start + 900])

        with tempfile.TemporaryDirectory() as tmp:
            importance.to_csv(os.path.join(tmp, "feature_importance.csv"), index=False)
            predictions.to_csv(os.path.join(tmp, "test_predictions.csv"), index=False)
            shutil.copy(SCALER_PATH, os.path.join(tmp, "scaler.pkl"))
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        mlflow.xgboost.log_model(
            model,
            name="model",
            input_example=X_example,
            registered_model_name=MLFLOW_REGISTERED_MODEL if register_model else None,
        )
        print(f"\nMLflow run logged → {run.info.run_id}  ({MLFLOW_TRACKING_URI})")
        return run.info.run_id


def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model at '{MODEL_PATH}'. Run:  python model_trainer.py"
        )
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def predict_rul(X_scaled: np.ndarray, model) -> tuple[float, float, float]:
    """Point prediction plus a simple ±15% band (a heuristic, not a statistical interval)."""
    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(1, -1)
    rul = float(model.predict(X_scaled).clip(0)[0])
    margin = max(rul * 0.15, 5.0)
    return rul, max(rul - margin, 0.0), rul + margin


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the PrognosticAI RUL model")
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    p.add_argument("--register", action="store_true", help="register the model in MLflow")
    p.add_argument("--run-name", default=None)
    p.add_argument("--metrics-out", default=None, help="write metrics as JSON to this path")
    p.add_argument("--fail-above-rmse", type=float, default=None,
                   help=f"exit 1 if test RMSE is above this (CI gate, default config: {RMSE_GATE})")
    args = p.parse_args(argv)

    metrics = train(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate,
        device=args.device, log_to_mlflow=not args.no_mlflow,
        register_model=args.register, run_name=args.run_name,
    )
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
    if args.fail_above_rmse is not None and metrics["test_rmse"] > args.fail_above_rmse:
        print(f"\nQUALITY GATE FAILED: test RMSE {metrics['test_rmse']:.2f} "
              f"> {args.fail_above_rmse:.2f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
