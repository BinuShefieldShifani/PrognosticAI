"""
PrognosticAI — Central Configuration
=====================================
Dataset: NASA C-MAPSS FD001 (turbofan engine degradation)

Paths and runtime settings can be overridden with environment variables,
so the same code runs locally, in CI (CPU) and on Kaggle (GPU).
"""

import os

# ── LLM (Ollama in Docker) ─────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"   # change if Docker is on a different host
OLLAMA_MODEL = "gptoss:20b"              # the model you have available

# ── Dataset ────────────────────────────────────────────────────────────────
# We use FD001 only: single operating condition, single fault mode (HPC Degradation)
# This gives the cleanest signal and the most cited benchmark in PHM research
DATA_DIR = os.getenv("PROGNOSTIC_DATA_DIR", "data")
DATASET = "FD001"

# Columns exactly as NASA defines them (26 total, space-separated, no header)
COL_NAMES = (
    ["unit", "cycle", "os1", "os2", "os3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors that actually carry degradation signal in FD001
# (s1, s5, s6, s10, s16, s18, s19 are near-constant in FD001 — confirmed by variance analysis)
INFORMATIVE_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"
]

# ── RUL Labelling ──────────────────────────────────────────────────────────
# Piecewise-linear degradation assumption (standard for C-MAPSS)
# Engines are "healthy" (flat RUL=130) for their early cycles, then degrade linearly.
# Capping at 130 is the published standard — do NOT increase this.
MAX_RUL = 130

# ── Feature Engineering ───────────────────────────────────────────────────
ROLLING_WINDOWS = [5, 10, 20, 30]   # cycles — rolling mean/std: short to long smoothing
# Rolling linear-regression slope of each sensor over these windows (cycles).
# Long windows measure how fast a sensor is drifting, which is what separates an
# engine that is just starting to degrade from one that is close to failure.
SLOPE_WINDOWS = [10, 30, 50, 80]
SEQUENCE_LENGTH = 30                # look-back window for feature computation

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_DIR = os.getenv("PROGNOSTIC_MODEL_DIR", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_rul.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# "auto" picks cuda when an NVIDIA GPU is visible, otherwise cpu.
XGB_DEVICE = os.getenv("PROGNOSTIC_DEVICE", "auto")
N_ESTIMATORS = int(os.getenv("PROGNOSTIC_N_ESTIMATORS", "5000"))
LEARNING_RATE = float(os.getenv("PROGNOSTIC_LEARNING_RATE", "0.015"))

# Validation split is done by ENGINE, not by row. A row-level shuffle puts
# neighbouring cycles of the same engine in both train and val, which makes
# the validation score look better than it is and skews early stopping.
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# ── Experiment tracking (MLflow) ──────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "prognosticai-fd001")
MLFLOW_ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION")  # None = MLflow default
MLFLOW_REGISTERED_MODEL = "prognosticai-rul-xgb"

# Quality gate used by CI and the Dagster asset check (test-set RMSE, cycles)
RMSE_GATE = float(os.getenv("PROGNOSTIC_RMSE_GATE", "15.0"))

# ── Alert Thresholds ──────────────────────────────────────────────────────
THRESHOLD_CRITICAL = 30    # cycles — ground immediately
THRESHOLD_HIGH     = 60    # cycles — plan maintenance this week
THRESHOLD_MEDIUM   = 100   # cycles — schedule next maintenance window
# > 100 cycles → LOW: monitor normally

# ── What counts as a good result on C-MAPSS FD001 ─────────────────────────
# Published XGBoost baselines:      RMSE ≈ 13–16
# Published LSTM baselines:         RMSE ≈ 10–13
# Published SOTA (2024 ensembles):  RMSE ≈ 6–8
# Our target with good features:    RMSE < 16  → "good"
#                                   RMSE < 13  → "excellent"
RMSE_TARGET = 16.0
