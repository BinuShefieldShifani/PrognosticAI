"""
serve.py — REST API for the champion RUL model
===============================================
Serves whichever packaged model PROGNOSTIC_MODEL_URI points to. By default that is
the registry champion (models:/prognosticai-rul@champion), with the same local
fallback as the agent. In the Docker image it is a folder mounted at /models/champion,
written by export_model.py.

  GET  /health    200 when a model is loaded, 503 when not (readiness probe)
  GET  /model     which model is serving: type, version, source, its metrics
  POST /predict   raw sensor cycles in, remaining useful life out
  POST /reload    load the model again, e.g. after a new champion is promoted

The request/response contract is the packaged model's own contract, so XGBoost and
LSTM champions are served the same way.

    uvicorn serve:app --port 8000
"""

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Response
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, Field

from config import (
    CHAMPION_ALIAS, MLFLOW_REGISTERED_MODEL, MLFLOW_TRACKING_URI, PREDICTOR_URI,
)
from rul_predictor import OUTPUT_COLUMNS, load_predictor, model_type

MAX_ROWS = int(os.getenv("PROGNOSTIC_MAX_ROWS", "50000"))
ADMIN_TOKEN = os.getenv("PROGNOSTIC_ADMIN_TOKEN")   # if set, /reload needs X-Admin-Token

log = logging.getLogger("prognosticai.serve")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


# ── Schemas ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    cycles: list[dict[str, float]] = Field(
        ..., min_length=1,
        description="One object per cycle: unit, cycle and the model's sensors (s2, s3, ...). "
                    "Extra keys are ignored.",
    )
    all_cycles: bool = Field(False, description="True: one prediction per cycle instead of per engine")


class Prediction(BaseModel):
    unit: int
    last_cycle: int
    history_cycles: int
    short_history: bool
    rul: float
    rul_lower: float
    rul_upper: float


class ModelInfo(BaseModel):
    model_type: str
    source: str
    version: str | None = None
    val_rmse: float | None = None
    test_rmse: float | None = None
    nasa_score: float | None = None
    loaded_at: str


class PredictResponse(BaseModel):
    model: ModelInfo
    predictions: list[Prediction]


# ── Model holder ──────────────────────────────────────────────────────────

def _registry_info(uri: str) -> dict:
    """Version and metrics for a models:/<name>@<alias> URI (best effort)."""
    if not uri.startswith("models:/") or "@" not in uri:
        return {}
    try:
        import mlflow
        name, alias = uri[len("models:/"):].split("@", 1)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mv = mlflow.MlflowClient().get_model_version_by_alias(name, alias)
        info = {"version": str(mv.version)}
        for k in ("val_rmse", "test_rmse", "nasa_score"):
            if k in mv.tags:
                info[k] = float(mv.tags[k])
        return info
    except Exception:
        return {}


def _folder_info(path: str) -> dict:
    """export_model.py writes serving_info.json next to an exported package."""
    p = os.path.join(path, "serving_info.json")
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        meta = json.load(f)
    return {k: meta[k] for k in ("version", "val_rmse", "test_rmse", "nasa_score", "source")
            if meta.get(k) is not None}


def _integer_columns(schema) -> list[str]:
    """Columns the model's signature declares as integers."""
    if schema is None:
        return ["unit", "cycle"]
    from mlflow.types import DataType
    return [spec.name for spec in schema.inputs
            if getattr(spec, "type", None) in (DataType.long, DataType.integer)]


class ModelHolder:
    def __init__(self, uri: str):
        self.uri = uri
        self.predictor = None
        self.info: ModelInfo | None = None
        self.error: str | None = None
        self.required_columns: list[str] = []
        self.int_columns: list[str] = ["unit", "cycle"]
        self._lock = threading.Lock()

    def load(self):
        with self._lock:
            try:
                predictor = load_predictor(self.uri)
            except Exception as exc:                       # noqa: BLE001 - reported on /health
                self.error = f"{type(exc).__name__}: {exc}"
                log.error("model load failed: %s", self.error)
                return False
            source = getattr(predictor, "source", self.uri)
            extra = _registry_info(source) if source.startswith("models:/") else _folder_info(source)
            schema = predictor.metadata.get_input_schema()
            self.required_columns = schema.input_names() if schema else []
            self.int_columns = _integer_columns(schema)
            self.predictor = predictor
            self.info = ModelInfo(
                model_type=model_type(predictor),
                source=extra.pop("source", source),
                loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **extra,
            )
            self.error = None
            log.info("serving %s model from %s (version %s)",
                     self.info.model_type, self.info.source, self.info.version)
            return True


MODEL = ModelHolder(PREDICTOR_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    MODEL.load()        # a failed load does not stop the server; /health reports it
    yield


app = FastAPI(
    title="PrognosticAI RUL API",
    description="Remaining useful life for turbofan engines, served by the registry champion "
                f"({MLFLOW_REGISTERED_MODEL}@{CHAMPION_ALIAS}).",
    version="1.0",
    lifespan=lifespan,
)


def _require_model():
    if MODEL.predictor is None:
        raise HTTPException(status_code=503, detail=f"No model loaded: {MODEL.error}")


def _to_frame(cycles: list[dict[str, float]], int_columns: list[str]) -> pd.DataFrame:
    """
    JSON numbers arrive as floats. Columns the model's signature stores as integers
    (unit, cycle, and in FD001 also s17) are cast back, if they hold whole numbers.
    """
    frame = pd.DataFrame(cycles)
    for col in ("unit", "cycle"):
        if col not in frame.columns:
            raise HTTPException(status_code=422, detail=f"Invalid RUL request: missing columns: ['{col}']")
    for col in int_columns:
        if col not in frame.columns:
            continue                      # reported as missing by the caller
        values = frame[col]
        if values.isna().any() or not (values == values.round()).all():
            raise HTTPException(status_code=422, detail=f"Invalid RUL request: '{col}' must be whole numbers")
        frame[col] = values.astype("int64")
    return frame


def _short(message: str) -> str:
    """MLflow puts the whole input frame in schema errors; keep the reason, which follows 'Error:'."""
    if "Error:" in message:
        message = message.rsplit("Error:", 1)[1]
    return " ".join(message.split())[:500]


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health(response: Response):
    if MODEL.predictor is None:
        response.status_code = 503
        return {"status": "unavailable", "error": MODEL.error}
    return {"status": "ok", "model_type": MODEL.info.model_type, "version": MODEL.info.version}


@app.get("/model", response_model=ModelInfo)
def model_info():
    _require_model()
    return MODEL.info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, response: Response):
    _require_model()
    if len(req.cycles) > MAX_ROWS:
        raise HTTPException(status_code=413, detail=f"Too many rows ({len(req.cycles)}); limit is {MAX_ROWS}")

    started = time.perf_counter()
    frame = _to_frame(req.cycles, MODEL.int_columns)
    missing = [c for c in MODEL.required_columns if c not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Invalid RUL request: missing columns: {missing}")
    try:
        out = MODEL.predictor.predict(frame, params={"all_cycles": req.all_cycles})
    except (ValueError, MlflowException) as exc:
        # The packaged model checks its own input: signature, missing sensors, gaps, nulls
        raise HTTPException(status_code=422, detail=_short(str(exc))) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    info = MODEL.info
    response.headers["X-Model-Type"] = info.model_type
    if info.version:
        response.headers["X-Model-Version"] = info.version
    log.info("predict rows=%d engines=%d all_cycles=%s model=%s v=%s ms=%.1f",
             len(frame), out["unit"].nunique(), req.all_cycles, info.model_type, info.version, elapsed_ms)

    records = out[OUTPUT_COLUMNS].to_dict(orient="records")
    return PredictResponse(model=info, predictions=[Prediction(**r) for r in records])


@app.post("/reload")
def reload(x_admin_token: str | None = Header(default=None)):
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Missing or wrong X-Admin-Token")
    previous = MODEL.info.version if MODEL.info else None
    if not MODEL.load():
        raise HTTPException(status_code=503, detail=f"Reload failed, keeping the previous model: {MODEL.error}")
    return {"status": "reloaded", "previous_version": previous,
            "model_type": MODEL.info.model_type, "version": MODEL.info.version}
