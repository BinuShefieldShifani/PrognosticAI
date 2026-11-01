"""
rul_predictor.py — Packaged RUL model (MLflow pyfunc)
======================================================
Before this module the trained model and the scaler lived in two separate
files, and every caller had to rebuild the features itself. Nothing stopped a
retrained model from being loaded next to an old scaler.

RULPredictor bundles three things into one MLflow model:
  - feature_spec.json : the sensors and windows the model was trained with
  - scaler.pkl        : the fitted MinMaxScaler
  - xgboost_model.json: the XGBoost model in its native, version-portable format

Contract
  input : raw C-MAPSS cycles, one row per cycle
          columns: unit, cycle, and every sensor in the feature spec
          (extra columns such as os1-os3 are ignored)
  output: one row per engine, predicted at its last observed cycle
          unit, last_cycle, history_cycles, short_history, rul, rul_lower, rul_upper

The same contract will be used by the LSTM challenger and the FastAPI service,
so callers never need to know which model is behind it.
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


# ── Input validation ──────────────────────────────────────────────────────

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


def heuristic_band(rul: float) -> tuple[float, float]:
    """±15% band (minimum ±5 cycles). A heuristic, not a statistical interval."""
    margin = max(rul * 0.15, 5.0)
    return max(rul - margin, 0.0), rul + margin


# ── The packaged model ────────────────────────────────────────────────────

class RULPredictor(mlflow.pyfunc.PythonModel):
    """Raw sensor cycles in, RUL per engine out."""

    def load_context(self, context):
        import xgboost as xgb

        with open(context.artifacts["feature_spec"]) as f:
            self.spec = json.load(f)
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.model = xgb.XGBRegressor()
        self.model.load_model(context.artifacts["model"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from data_processor import engineer_features, get_feature_columns

        problems = validate_input(model_input, self.spec["sensors"])
        if problems:
            raise ValueError("Invalid RUL request: " + "; ".join(problems))

        cols = ["unit", "cycle"] + self.spec["sensors"]
        feats = engineer_features(model_input[cols].copy(), fit_scaler=False, spec=self.spec)
        last = feats.loc[feats.groupby("unit")["cycle"].idxmax()].sort_values("unit")

        X = self.scaler.transform(last[get_feature_columns(self.spec)].values)
        rul = self.model.predict(X).clip(0)

        history = model_input.groupby("unit")["cycle"].size().reindex(last["unit"]).values
        bands = [heuristic_band(float(r)) for r in rul]
        return pd.DataFrame({
            "unit": last["unit"].astype("int64").values,
            "last_cycle": last["cycle"].astype("int64").values,
            "history_cycles": history.astype("int64"),
            "short_history": history < MIN_HISTORY_CYCLES,
            "rul": rul.astype("float64"),
            "rul_lower": [b[0] for b in bands],
            "rul_upper": [b[1] for b in bands],
        })[OUTPUT_COLUMNS]


# ── Building, logging and loading ─────────────────────────────────────────

def _pip_requirements() -> list[str]:
    pkgs = ["mlflow", "xgboost", "scikit-learn", "pandas", "numpy", "joblib"]
    return [f"{p}=={version(p)}" for p in pkgs]


def _code_paths() -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    return [os.path.join(here, f) for f in ("rul_predictor.py", "data_processor.py", "config.py")]


def _write_artifacts(model, scaler, spec: dict, tmp: str) -> dict:
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


def _signature(input_example: pd.DataFrame, model, scaler, spec: dict):
    """Infer the signature by running the real predictor on the example."""
    from mlflow.models import infer_signature

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = _write_artifacts(model, scaler, spec, tmp)
        predictor = RULPredictor()
        predictor.load_context(type("Ctx", (), {"artifacts": artifacts})())
        output = predictor.predict(None, input_example)
    return infer_signature(input_example, output)


def log_predictor(model, scaler, spec: dict, input_example: pd.DataFrame,
                  registered_model_name: str | None = None):
    """Log the packaged model to the active MLflow run under the name 'model'."""
    signature = _signature(input_example, model, scaler, spec)
    with tempfile.TemporaryDirectory() as tmp:
        return mlflow.pyfunc.log_model(
            name="model",
            python_model=RULPredictor(),
            artifacts=_write_artifacts(model, scaler, spec, tmp),
            code_paths=_code_paths(),
            pip_requirements=_pip_requirements(),
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )


def save_predictor(model, scaler, spec: dict, input_example: pd.DataFrame, path: str) -> str:
    """Save the packaged model to a local folder, for offline use without MLflow tracking."""
    signature = _signature(input_example, model, scaler, spec)
    if os.path.exists(path):
        shutil.rmtree(path)
    with tempfile.TemporaryDirectory() as tmp:
        mlflow.pyfunc.save_model(
            path=path,
            python_model=RULPredictor(),
            artifacts=_write_artifacts(model, scaler, spec, tmp),
            code_paths=_code_paths(),
            pip_requirements=_pip_requirements(),
            signature=signature,
            input_example=input_example,
        )
    return path


def load_predictor(uri: str | None = None):
    """
    Load a packaged model. `uri` can be a local folder or any MLflow URI
    (runs:/<id>/model, models:/<name>@<alias>). Defaults to the local copy.
    """
    from config import PREDICTOR_DIR

    uri = uri or PREDICTOR_DIR
    if "://" not in uri and not os.path.exists(uri):
        raise FileNotFoundError(f"No packaged model at '{uri}'. Run:  python model_trainer.py")
    return mlflow.pyfunc.load_model(uri)
