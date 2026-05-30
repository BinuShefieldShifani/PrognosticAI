"""
data_processor.py — C-MAPSS FD001 Preprocessing Pipeline
==========================================================
What this does (plain English):
  1. Loads the raw sensor readings (26 cols, space-separated, no header)
  2. Computes the true RUL label for every row in the training set
     RUL = (max cycle for that engine) - (current cycle), capped at 130
  3. Drops sensors that don't change in FD001 (they add noise, not signal)
  4. Creates rolling features — mean and std over 5, 10, 15 cycle windows
     This captures "is the engine getting worse?" not just "what is it right now?"
  5. Normalises all features to [0, 1] so XGBoost isn't biased by scale

Why piecewise RUL (cap at 130)?
  Real engines don't degrade linearly from birth. They run "healthy" for most
  of their life, then degrade. Capping at 130 tells the model: "anything above
  130 cycles to failure — just predict 130. We only care when failure is close."
  This is the published standard (Saxena et al., 2008) and what GKN engineers expect.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_DIR, COL_NAMES, INFORMATIVE_SENSORS,
    MAX_RUL, ROLLING_WINDOWS, SCALER_PATH
)


# ── Load Raw Data ─────────────────────────────────────────────────────────

def load_raw(split: str) -> pd.DataFrame:
    """Load train or test file. split = 'train' | 'test'"""
    path = os.path.join(DATA_DIR, f"{split}_FD001.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COL_NAMES)
    df = df.dropna(axis=1, how="all")   # NASA files sometimes have trailing spaces → empty col
    return df


def load_rul_ground_truth() -> np.ndarray:
    """Load the ground-truth RUL for the test set (100 values, one per engine)."""
    path = os.path.join(DATA_DIR, "RUL_FD001.txt")
    return pd.read_csv(path, header=None).values.flatten()


# ── RUL Labelling ─────────────────────────────────────────────────────────

def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'rul' column to training data using piecewise-linear assumption.
    For each engine: rul = min(max_cycle - current_cycle, MAX_RUL)
    """
    max_cycles = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="unit")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=MAX_RUL)
    df = df.drop(columns=["max_cycle"])
    return df


# ── Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
    """
    Build the feature matrix used for model training and inference.

    Features created per engine:
      - Raw sensor values (informative sensors only)
      - Rolling mean  × 3 windows (5, 10, 15 cycles)
      - Rolling std   × 3 windows (5, 10, 15 cycles)
      = 14 base sensors × (1 + 2×3) = 14 × 7 = 98 features

    Plus: cycle_norm (normalized cycle position within engine life)
    """
    dfs = []
    for unit_id, group in df.groupby("unit"):
        g = group.sort_values("cycle").copy()

        # Normalize cycle by a fixed constant (max observed in FD001 training set).
        # CRITICAL: do NOT use g["cycle"].max() here — for test engines, the last
        # observed cycle is NOT the failure cycle, so cycle/max would always be 1.0
        # for every engine's last row, making the model think all engines are dying.
        g["cycle_norm"] = g["cycle"] / 400.0

        for sensor in INFORMATIVE_SENSORS:
            for w in ROLLING_WINDOWS:
                g[f"{sensor}_mean{w}"] = (
                    g[sensor].rolling(w, min_periods=1).mean()
                )
                g[f"{sensor}_std{w}"] = (
                    g[sensor].rolling(w, min_periods=1).std().fillna(0)
                )
        dfs.append(g)

    result = pd.concat(dfs, ignore_index=True)
    return result


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names used for training."""
    base = INFORMATIVE_SENSORS + ["cycle_norm"]
    rolling = []
    for s in INFORMATIVE_SENSORS:
        for w in ROLLING_WINDOWS:
            rolling += [f"{s}_mean{w}", f"{s}_std{w}"]
    return base + rolling


def scale_features(
    df: pd.DataFrame,
    fit: bool = True,
    scaler: MinMaxScaler | None = None
) -> tuple[np.ndarray, MinMaxScaler]:
    """
    MinMax-scale features to [0, 1].
    If fit=True, fits a new scaler (training). Otherwise uses provided scaler (inference).
    """
    feat_cols = get_feature_columns()
    X = df[feat_cols].values

    if fit:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Scaler saved → {SCALER_PATH}")
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


# ── Convenience: Build Train Dataset in One Call ──────────────────────────

def build_train_dataset() -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Full pipeline for training:
      raw → RUL labels → features → scaled X, y
    Returns: X_train, y_train, fitted_scaler
    """
    print("Loading train_FD001.txt ...")
    df = load_raw("train")
    print(f"  {len(df):,} rows, {df['unit'].nunique()} engines")

    print("Adding piecewise RUL labels (cap={MAX_RUL}) ...")
    df = add_rul_labels(df)

    print("Engineering features ...")
    df = engineer_features(df, fit_scaler=True)

    X, scaler = scale_features(df, fit=True)
    y = df["rul"].values
    print(f"  Feature matrix: {X.shape}  |  RUL range: [{y.min():.0f}, {y.max():.0f}]")
    return X, y, scaler


def build_test_dataset(scaler: MinMaxScaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline for test evaluation:
    Uses only the LAST cycle per engine (that's what RUL_FD001.txt labels).
    Returns: X_test (100 rows, one per engine), y_test (ground truth RUL)
    """
    print("Loading test_FD001.txt ...")
    df = load_raw("test")

    print("Engineering features ...")
    df = engineer_features(df, fit_scaler=False)

    # For test: one row per engine = the last observed cycle
    last_cycles = df.loc[df.groupby("unit")["cycle"].idxmax()]
    last_cycles = last_cycles.sort_values("unit")

    X_test, _ = scale_features(last_cycles, fit=False, scaler=scaler)
    y_test = load_rul_ground_truth()

    print(f"  Test matrix: {X_test.shape}  |  Ground truth RUL range: [{y_test.min():.0f}, {y_test.max():.0f}]")
    return X_test, y_test


# ── Single Engine Inference Data ──────────────────────────────────────────

def get_engine_data(engine_id: int, split: str = "train") -> pd.DataFrame:
    """Return all cycles for a single engine, with features engineered."""
    df = load_raw(split)
    if split == "train":
        df = add_rul_labels(df)
    engine_df = df[df["unit"] == engine_id].copy()
    engine_df = engineer_features(engine_df)
    return engine_df.sort_values("cycle")