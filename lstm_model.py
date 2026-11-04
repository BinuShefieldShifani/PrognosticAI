"""
lstm_model.py — LSTM network and sequence windows
===================================================
The challenger to the XGBoost model. Instead of 183 hand-built features, the
LSTM reads the last 30 cycles of the 14 informative sensors (plus the
normalised cycle number) and learns the degradation pattern itself.

Kept separate from lstm_trainer.py so the packaged MLflow model can rebuild the
network at load time without importing any training code.

Windows
  Every cycle of an engine becomes one sample: the window ending at that cycle.
  Engines with fewer cycles than the window are padded at the front by
  repeating their first reading (a common choice for C-MAPSS). These are the
  same "short history" engines the XGBoost package flags.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from config import (
    INFORMATIVE_SENSORS, LSTM_DROPOUT, LSTM_HIDDEN, LSTM_LAYERS, MAX_RUL, SEQUENCE_LENGTH,
)
from data_processor import CYCLE_NORM_DIVISOR


def lstm_spec() -> dict:
    """Everything needed to rebuild inputs and network. Stored with the packaged model."""
    return {
        "model_type": "lstm",
        "sensors": list(INFORMATIVE_SENSORS),
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": LSTM_HIDDEN,
        "num_layers": LSTM_LAYERS,
        "dropout": LSTM_DROPOUT,
        "cycle_norm_divisor": CYCLE_NORM_DIVISOR,
        "min_history": SEQUENCE_LENGTH,
        "max_rul": MAX_RUL,              # network predicts RUL / max_rul (0..1)
    }


def input_columns(spec: dict) -> list[str]:
    return list(spec["sensors"]) + ["cycle_norm"]


def add_cycle_norm(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    df = df.copy()
    df["cycle_norm"] = df["cycle"] / spec["cycle_norm_divisor"]
    return df


def build_windows(df: pd.DataFrame, X_scaled: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """
    One window per row. `df` must be sorted by unit then cycle, and `X_scaled`
    must be its scaled input matrix in the same row order.
    Returns windows (n_rows, seq_len, n_features) and the row index of each window's last cycle.
    """
    units = df["unit"].values
    windows, order = [], []
    start = 0
    for end in np.flatnonzero(np.r_[units[1:] != units[:-1], True]) + 1:
        block = X_scaled[start:end]
        pad = np.repeat(block[:1], seq_len - 1, axis=0)
        padded = np.vstack([pad, block])
        # sliding_window_view gives (n, n_features, seq_len); move time to axis 1
        windows.append(sliding_window_view(padded, seq_len, axis=0).transpose(0, 2, 1))
        order.append(np.arange(start, end))
        start = end
    return np.ascontiguousarray(np.concatenate(windows), dtype=np.float32), np.concatenate(order)


def build_network(spec: dict):
    """Create the PyTorch module. torch is imported here so the rest of the repo works without it."""
    import torch
    from torch import nn

    class RULLSTM(nn.Module):
        def __init__(self, n_features: int, hidden: int, layers: int, dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                                dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)   # last time step → RUL (scaled 0..1)

    return RULLSTM(len(input_columns(spec)), spec["hidden_size"], spec["num_layers"], spec["dropout"])
