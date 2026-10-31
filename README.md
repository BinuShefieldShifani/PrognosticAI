# PrognosticAI — Turbofan Engine Remaining Useful Life Prediction

[![CI](https://github.com/BinuShefieldShifani/PrognosticAI/actions/workflows/ci.yml/badge.svg)](https://github.com/BinuShefieldShifani/PrognosticAI/actions/workflows/ci.yml)

**Predictive maintenance pipeline for turbofan engines, from raw sensor data to maintenance work orders.**  
Built on NASA C-MAPSS FD001 · XGBoost (GPU/CPU) · MLflow · Dagster · GitHub Actions · LangGraph · Local LLM (Ollama)

---

## What It Does 

A turbofan engine has 21 sensors reporting data every flight cycle (takeoff → cruise → landing). Over time, components like the High Pressure Compressor degrade and blade erosion increases tip clearance, efficiency drops, and eventually the engine fails.

**PrognosticAI answers one question: how many cycles until this engine needs maintenance?**

It runs a four-node AI agent pipeline:

```
[Sensor Analyst]        ← detects which sensors are degrading (z-score anomaly detection)
       ↓
[RUL Estimator]         ← XGBoost model predicts Remaining Useful Life in cycles
       ↓
    (conditional)
       ├── degraded → [Failure Diagnosis]  ← LLM classifies failure mode from sensor patterns
       │                     ↓
       │              [Work Order Generator] ← LLM generates structured MRO work order
       │
       └── healthy  → [Routine Advisory]  ← LLM generates monitoring recommendation
```

---

## Dataset: NASA C-MAPSS FD001

| Property | Value |
|---|---|
| Source | NASA Prognostics Repository (public, free) |
| Sub-dataset | FD001 — single operating condition, single fault (HPC Degradation) |
| Training engines | 100 |
| Test engines | 100 |
| Sensors | 21 per cycle (14 informative, 7 near-constant in FD001) |
| Format | Space-separated, 26 columns, no header |

**Download:**
```bash
python download_data.py
```
Or manually: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

---

## Architecture

```
prognosticai/
├── config.py               ← thresholds, sensor lists, env-overridable paths and settings
├── download_data.py        ← fetches NASA C-MAPSS FD001
├── data_processor.py       ← loading, validation, RUL labelling, feature engineering
├── model_trainer.py        ← XGBoost training + evaluation, MLflow tracking, CLI
├── pipelines/definitions.py← Dagster assets, data-quality checks, weekly schedule
├── knowledge_base.py       ← ATA chapter maintenance procedures (MRO-001 → MRO-006)
├── agent_graph.py          ← LangGraph 4-node agent pipeline
├── app.py                  ← Streamlit dashboard
├── tests/                  ← pytest suite on synthetic C-MAPSS-format data
├── notebooks/kaggle_train.ipynb ← GPU training run on Kaggle
└── .github/workflows/ci.yml     ← lint, tests, real-data training + RMSE gate
```

---

## Model Performance on FD001

Measured on the official test set (100 engines, last observed cycle), seed 42, CPU run:

| Metric | This model | Published XGBoost | Published LSTM | SOTA |
|---|---|---|---|---|
| Test RMSE (cycles) | **18.4** | 13–16 | 10–13 | 6–8 |
| Test MAE (cycles) | **13.4** | | | |
| NASA Score S | **690** | ~300–500 | ~200–300 | <100 |

Across five different validation splits, test RMSE stayed between 18.0 and 19.4. Every run is
tracked in MLflow, so new feature or parameter experiments can be compared against this baseline.

**RMSE interpretation:** RMSE = 18 means predictions are off by about 18 flight cycles on average
for an engine whose history has been cut off at a random point.

**NASA Score S** penalises late predictions (predicting more remaining life than the engine
actually has, so maintenance comes too late) more than early ones. Lower = better.

**Validation split by engine.** An earlier version shuffled rows before the train/validation split.
Neighbouring cycles of the same engine then ended up on both sides, and validation RMSE came out at
6.4 while test RMSE was 18.5. Splitting by engine (`GroupShuffleSplit`) gives an honest validation
score (16–20), stops early at a few hundred rounds instead of 5000, and trains about 20x faster
with the same test result.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (~2 MB)
python download_data.py

# 3. Train model (about 30 s on CPU; logs the run to MLflow)
python model_trainer.py                # --device cpu|cuda|auto, --register, --no-mlflow

# 4. Launch UI
streamlit run app.py
```

## Experiment Tracking (MLflow)

Each training run logs its parameters, metrics (val/test RMSE, MAE, NASA score, best iteration,
training time), the validation curve, feature importance, per-engine test predictions, the scaler
and the model itself. `--register` adds the model to the registry as `prognosticai-rul-xgb`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000
```

## Pipeline Orchestration (Dagster)

```
cmapss_raw_files ──(raw_data_is_valid, blocking)──▶ rul_model ──(rmse_within_gate)
```

The data-quality check (schema, nulls, cycle continuity, label count) blocks training if the raw
data is bad. The RMSE gate fails if the test RMSE goes above 20 cycles. The whole graph is
scheduled weekly (Monday 03:00, Europe/Stockholm).

```bash
pip install -r requirements-dev.txt
dagster dev                                           # http://localhost:3000
# or one-off from the command line:
dagster asset materialize -m pipelines.definitions --select '*'
```

## Testing and CI

`pytest` runs the suite on small synthetic files in C-MAPSS format, so it is fast and offline.
It covers RUL labelling, feature engineering, data validation, the engine-level split, the NASA
score, MLflow logging and model reload, and the full Dagster graph including a failing gate.

GitHub Actions runs on every push and pull request:
1. `ruff`, Dagster definition validation and `pytest`
2. Training on the real FD001 data (CPU), failing the build if test RMSE > 20, with the MLflow
   run uploaded as a build artifact and metrics in the job summary

## GPU Training on Kaggle

`notebooks/kaggle_train.ipynb` clones the repo, runs the tests, then runs the Dagster pipeline on
a Kaggle GPU and shows the MLflow runs. Set the accelerator to GPU and turn Internet on first.

### Ollama (LLM backend)

The agent uses a local LLM via [Ollama](https://ollama.ai) running in Docker:

```bash
# Pull and run Ollama
docker run -d --gpus all -p 11434:11434 ollama/ollama
docker exec -it <container> ollama pull gptoss:20b
```

If Ollama is unavailable, the sensor analysis and RUL prediction still work — only the LLM-powered failure diagnosis and work order generation are affected.

---

## Key Engineering Decisions

**Why FD001?** Single operating condition means no need to cluster or normalise by operating point. The signal-to-noise ratio is highest, and published benchmarks are most consistent. Easier to validate.

**Why Dagster over Airflow?** It installs with pip, runs locally with `dagster dev`, models the data and the model as assets with built-in quality checks, and the whole graph can be run inside pytest, which is what lets CI test the pipeline itself.

**Why piecewise RUL (cap at 130)?** Engines don't degrade linearly from new. They run healthy for most of their life. Capping tells the model "don't bother learning the healthy regime — focus on the last 130 cycles." Published standard since Saxena et al. (2008).

**Why these 14 sensors?** In FD001, sensors s1, s5, s6, s10, s16, s18, s19 have near-zero variance — they carry no degradation signal. Including them adds noise. This is validated by both variance analysis and feature importance from XGBoost.

**Why rolling features (5, 10, 15 cycles)?** A single cycle's sensor reading is noisy. Rolling mean/std captures *trend* — is sensor s7 increasing over the last 10 cycles? That's more informative than any single reading.

---

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM'08 Conference.
- NASA C-MAPSS benchmark: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
- LangGraph documentation: https://langchain-ai.github.io/langgraph/
