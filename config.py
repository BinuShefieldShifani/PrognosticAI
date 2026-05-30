"""
PrognosticAI — Central Configuration
=====================================
Targets: GKN Aerospace / Volvo CE / Volvo Trucks
Dataset: NASA C-MAPSS FD001 (turbofan engine degradation)
"""

# ── LLM (Ollama in Docker) ─────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"   # change if Docker is on a different host
OLLAMA_MODEL = "gptoss:20b"              # the model you have available

# ── Dataset ────────────────────────────────────────────────────────────────
# We use FD001 only: single operating condition, single fault mode (HPC Degradation)
# This gives the cleanest signal and the benchmark most cited in GKN/aerospace research
DATA_DIR = "data"
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
ROLLING_WINDOWS = [5, 10, 15]       # cycles — captures short, medium, long trends
SEQUENCE_LENGTH = 30                # look-back window for feature computation

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/xgboost_rul.pkl"
SCALER_PATH = "models/scaler.pkl"

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
