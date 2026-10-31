"""
Test setup: every test runs against small synthetic C-MAPSS-format files in a
temp folder, so the suite is fast, offline and never touches ./data or ./models.
Environment variables are set here, before any project module imports config.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="prognosticai-tests-"))
os.environ["PROGNOSTIC_DATA_DIR"] = str(_TMP / "data")
os.environ["PROGNOSTIC_MODEL_DIR"] = str(_TMP / "models")
os.environ["PROGNOSTIC_DEVICE"] = "cpu"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP / 'mlflow.db'}"
os.environ["MLFLOW_ARTIFACT_LOCATION"] = str(_TMP / "mlruns")
os.environ["MLFLOW_EXPERIMENT_NAME"] = "prognosticai-tests"

N_SENSORS = 21
DEGRADING = {2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21}


def _engine_rows(unit: int, n_cycles: int, life: int, rng: np.random.Generator) -> np.ndarray:
    """One engine in C-MAPSS layout: unit, cycle, 3 settings, 21 sensors."""
    cycles = np.arange(1, n_cycles + 1)
    wear = (cycles / life) ** 2
    rows = np.zeros((n_cycles, 26))
    rows[:, 0] = unit
    rows[:, 1] = cycles
    rows[:, 2:5] = rng.normal(0, 0.001, (n_cycles, 3))
    for s in range(1, N_SENSORS + 1):
        base = 100 + 10 * s
        drift = 5 * wear if s in DEGRADING else 0
        rows[:, 4 + s] = base + drift + rng.normal(0, 0.05, n_cycles)
    return rows


def write_synthetic_cmapss(data_dir: Path, n_train: int = 20, n_test: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    data_dir.mkdir(parents=True, exist_ok=True)
    train = [_engine_rows(u, life, life, rng)
             for u, life in enumerate(rng.integers(120, 200, n_train), start=1)]
    test, rul = [], []
    for u, life in enumerate(rng.integers(120, 200, n_test), start=1):
        observed = int(rng.integers(40, life - 5))
        test.append(_engine_rows(u, observed, life, rng))
        rul.append(life - observed)
    fmt = ["%d", "%d"] + ["%.4f"] * 24
    np.savetxt(data_dir / "train_FD001.txt", np.vstack(train), fmt=fmt)
    np.savetxt(data_dir / "test_FD001.txt", np.vstack(test), fmt=fmt)
    np.savetxt(data_dir / "RUL_FD001.txt", np.array(rul), fmt="%d")


@pytest.fixture(scope="session", autouse=True)
def synthetic_data():
    write_synthetic_cmapss(Path(os.environ["PROGNOSTIC_DATA_DIR"]))
    return Path(os.environ["PROGNOSTIC_DATA_DIR"])
