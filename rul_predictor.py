"""
rul_predictor.py — Packaged RUL models (MLflow pyfunc)
=======================================================
Every model, whatever its architecture, is packaged with the same contract,
so the agent, the dashboard and (later) the API never need to know which one
is the champion.

  input : raw C-MAPSS cycles, one row per cycle
          columns: unit, cycle, and every sensor in the model's spec
          (extra columns such as os1-os3 are ignored)
  params: all_cycles (bool, default False)
          False → one row per engine, predicted at its last observed cycle
          True  → one row per cycle (each prediction only uses history up to that cycle)
  output: unit, last_cycle, history_cycles, short_history, rul, rul_lower, rul_upper
          (last_cycle = the last cycle the prediction could see)

Two implementations:
  XGBoostRULPredictor : feature_spec.json + scaler.pkl + xgboost_model.json
  LSTMRULPredictor    : lstm_spec.json    + scaler.pkl + lstm_state.pt
Each stores its own spec, so a config change after training never changes
what an existing model computes.
"""

import json
import os
import shutil
import tempfile
from importlib.metadata import version

import joblib
import mlflow
import numpy as np
import pandas as pd

from config import MIN_HISTORY_CYCLES

OUTPUT_COLUMNS = ["unit", "last_cycle", "history_cycles", "short_history",
                  "rul", "rul_lower", "rul_upper"]


# ── Shared helpers ────────────────────────────────────────────────────────

def validate_input(df: pd.DataFrame, sensors: list[str]) -> list[str]:
    """Return a list of problems with an inference request (empty = usable)."""
    if not isinstance(df, pd.DataFrame):
        return [f"expected a pandas DataFrame, got {type(df).__name__}"]
    required = ["unit", "cycle"] + list(sensors)
    missing = [c for c in required if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df.empty:
        return ["no rows"]

    problems = []
    non_numeric = [c for c in required if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")
        return problems
    nulls = [c for c in required if df[c].isna().any()]
    if nulls:
        problems.append(f"null values in: {nulls}")
    for unit, g in df.groupby("unit"):
        cycles = g["cycle"].values
        if not (np.diff(cycles) == 1).all():
            problems.append(f"engine {unit}: cycles must be consecutive and in order")
    return problems


def request_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Just the columns a packaged model reads, from any frame that has them (raw or engineered)."""
    from config import INFORMATIVE_SENSORS

    return df[["unit", "cycle"] + list(INFORMATIVE_SENSORS)]


def heuristic_band(rul: float) -> tuple[float, float]:
    """±15% band (minimum ±5 cycles). A heuristic, not a statistical interval."""
    margin = max(rul * 0.15, 5.0)
    return max(rul - margin, 0.0), rul + margin


def _prepare(model_input: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    problems = validate_input(model_input, sensors)
    if problems:
        raise ValueError("Invalid RUL request: " + "; ".join(problems))
    cols = ["unit", "cycle"] + list(sensors)
    return model_input[cols].sort_values(["unit", "cycle"], kind="stable").reset_index(drop=True)


def _format_output(rows: pd.DataFrame, rul: np.ndarray, all_cycles: bool, min_history: int) -> pd.DataFrame:
    """rows: the sorted input rows (unit, cycle) aligned with `rul`, one per cycle."""
    out = pd.DataFrame({
        "unit": rows["unit"].astype("int64").values,
        "last_cycle": rows["cycle"].astype("int64").values,
        "history_cycles": (rows.groupby("unit").cumcount() + 1).astype("int64").values,
        "rul": np.asarray(rul, dtype="float64").clip(0),
    })
    if not all_cycles:
        out = out.groupby("unit", sort=True).tail(1)
    out["short_history"] = out["history_cycles"] < min_history
    bands = [heuristic_band(float(r)) for r in out["rul"]]
    out["rul_lower"] = [b[0] for b in bands]
    out["rul_upper"] = [b[1] for b in bands]
    return out[OUTPUT_COLUMNS].reset_index(drop=True)


def _all_cycles(params) -> bool:
    return bool((params or {}).get("all_cycles", False))


# ── XGBoost package ───────────────────────────────────────────────────────

class XGBoostRULPredictor(mlflow.pyfunc.PythonModel):
    """183 engineered features → XGBoost."""

    def load_context(self, context):
        import xgboost as xgb

        with open(context.artifacts["feature_spec"]) as f:
            self.spec = json.load(f)
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.model = xgb.XGBRegressor()
        self.model.load_model(context.artifacts["model"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from data_processor import engineer_features, get_feature_columns

        rows = _prepare(model_input, self.spec["sensors"])
        all_cycles = _all_cycles(params)
        feats = engineer_features(rows, fit_scaler=False, spec=self.spec)   # same row order as `rows`
        idx = (np.arange(len(rows)) if all_cycles
               else rows.groupby("unit", sort=True).tail(1).index.values)
        X = self.scaler.transform(feats.iloc[idx][get_feature_columns(self.spec)].values)
        rul = np.full(len(rows), np.nan)
        rul[idx] = self.model.predict(X)
        return _format_output(rows, rul, all_cycles, self.spec.get("min_history", MIN_HISTORY_CYCLES))


# ── LSTM package ──────────────────────────────────────────────────────────

class LSTMRULPredictor(mlflow.pyfunc.PythonModel):
    """Last N cycles of raw (scaled) sensors → LSTM."""

    def load_context(self, context):
        import torch

        from lstm_model import build_network

        with open(context.artifacts["lstm_spec"]) as f:
            self.spec = json.load(f)
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.net = build_network(self.spec)
        self.net.load_state_dict(torch.load(context.artifacts["model"], map_location="cpu"))
        self.net.eval()

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        import torch

        from lstm_model import add_cycle_norm, build_windows, input_columns

        rows = _prepare(model_input, self.spec["sensors"])
        all_cycles = _all_cycles(params)
        X = self.scaler.transform(add_cycle_norm(rows, self.spec)[input_columns(self.spec)].values)
        windows, _ = build_windows(rows, X, self.spec["sequence_length"])   # one per row, same order
        idx = (np.arange(len(rows)) if all_cycles
               else rows.groupby("unit", sort=True).tail(1).index.values)
        rul = np.full(len(rows), np.nan)
        with torch.no_grad():
            rul[idx] = self.net(torch.from_numpy(windows[idx])).numpy() * self.spec["max_rul"]
        return _format_output(rows, rul, all_cycles, self.spec["min_history"])


# ── Building, logging, saving ─────────────────────────────────────────────

def _pip_requirements(extra: list[str]) -> list[str]:
    pkgs = ["mlflow", "scikit-learn", "pandas", "numpy", "joblib"] + extra
    return [f"{p}=={version(p)}" for p in pkgs]


def _code_paths() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    files = ("rul_predictor.py", "data_processor.py", "config.py", "lstm_model.py")
    return [os.path.join(here, f) for f in files]


def _xgb_artifacts(model, scaler, spec: dict, tmp: str) -> dict:
    paths = {
        "model": os.path.join(tmp, "xgboost_model.json"),
        "scaler": os.path.join(tmp, "scaler.pkl"),
        "feature_spec": os.path.join(tmp, "feature_spec.json"),
    }
    model.save_model(paths["model"])
    joblib.dump(scaler, paths["scaler"])
    with open(paths["feature_spec"], "w") as f:
        json.dump(spec, f, indent=2)
    return paths


def _lstm_artifacts(net, scaler, spec: dict, tmp: str) -> dict:
    import torch

    paths = {
        "model": os.path.join(tmp, "lstm_state.pt"),
        "scaler": os.path.join(tmp, "scaler.pkl"),
        "lstm_spec": os.path.join(tmp, "lstm_spec.json"),
    }
    torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, paths["model"])
    joblib.dump(scaler, paths["scaler"])
    with open(paths["lstm_spec"], "w") as f:
        json.dump(spec, f, indent=2)
    return paths


_KINDS = {
    "xgboost": (XGBoostRULPredictor, _xgb_artifacts, ["xgboost"]),
    "lstm": (LSTMRULPredictor, _lstm_artifacts, ["torch"]),
}


def _package(kind: str, model, scaler, spec: dict, input_example: pd.DataFrame, tmp: str) -> dict:
    """Write artifacts and build every keyword MLflow needs to log or save the package."""
    from mlflow.models import infer_signature

    cls, write, extra = _KINDS[kind]
    artifacts = write(model, scaler, spec, tmp)
    probe = cls()
    probe.load_context(type("Ctx", (), {"artifacts": artifacts})())
    output = probe.predict(None, input_example)
    return dict(
        python_model=cls(),
        artifacts=artifacts,
        code_paths=_code_paths(),
        pip_requirements=_pip_requirements(extra),
        signature=infer_signature(input_example, output, params={"all_cycles": False}),
        input_example=input_example,
        metadata={"model_type": kind},
    )


def log_package(kind: str, model, scaler, spec: dict, input_example: pd.DataFrame):
    """Log the packaged model to the active MLflow run under the name 'model'."""
    with tempfile.TemporaryDirectory() as tmp:
        return mlflow.pyfunc.log_model(name="model", **_package(kind, model, scaler, spec, input_example, tmp))


def save_package(kind: str, model, scaler, spec: dict, input_example: pd.DataFrame, path: str) -> str:
    """Save the packaged model to a local folder, for offline use without MLflow tracking."""
    if os.path.exists(path):
        shutil.rmtree(path)
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.pyfunc.save_model(path=path, **_package(kind, model, scaler, spec, input_example, tmp))
    return path


# Step-1 names, kept so existing callers keep working
def log_predictor(model, scaler, spec, input_example, registered_model_name=None):
    return log_package("xgboost", model, scaler, spec, input_example)


def save_predictor(model, scaler, spec, input_example, path):
    return save_package("xgboost", model, scaler, spec, input_example, path)


# ── Loading ───────────────────────────────────────────────────────────────

def model_type(predictor) -> str:
    return (predictor.metadata.metadata or {}).get("model_type", "unknown")


def load_predictor(uri: str | None = None, fallback: bool = True):
    """
    Load a packaged model. `uri` is a local folder or any MLflow URI
    (runs:/<id>/model, models:/<name>@<alias>). By default this is the registry
    champion; if the registry has no champion yet (fresh clone, offline), it
    falls back to the latest locally trained XGBoost package.
    The returned model has a `.source` attribute saying where it came from.
    """
    from config import MLFLOW_TRACKING_URI, PREDICTOR_DIR, PREDICTOR_URI

    uri = uri or PREDICTOR_URI
    if "://" in uri or uri.startswith("models:/") or uri.startswith("runs:/"):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            predictor = mlflow.pyfunc.load_model(uri)
            predictor.source = uri
            return predictor
        except Exception as exc:
            if not fallback:
                raise
            print(f"Could not load '{uri}' ({type(exc).__name__}); using local model at {PREDICTOR_DIR}")
            uri = PREDICTOR_DIR
    if not os.path.exists(uri):
        raise FileNotFoundError(f"No packaged model at '{uri}'. Run:  python model_trainer.py")
    predictor = mlflow.pyfunc.load_model(uri)
    predictor.source = uri
    return predictor
