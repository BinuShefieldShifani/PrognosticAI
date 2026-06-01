# PrognosticAI — Turbofan Engine Remaining Useful Life Prediction

**Multi-agent system for predictive maintenance of aerospace engines.**  
Built on NASA C-MAPSS FD001 · XGBoost (GPU) · LangGraph · Local LLM (Ollama)



---

## What It Does (Plain English)

A turbofan engine has 21 sensors reporting data every flight cycle (takeoff → cruise → landing). Over time, components like the High Pressure Compressor degrade — blade erosion increases tip clearance, efficiency drops, and eventually the engine fails.

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
├── config.py            ← all thresholds, model paths, sensor lists
├── download_data.py     ← fetches NASA C-MAPSS FD001
├── data_processor.py   ← loading, RUL labelling, feature engineering
├── model_trainer.py    ← XGBoost training + evaluation
├── knowledge_base.py   ← ATA chapter maintenance procedures (MRO-001 → MRO-006)
├── agent_graph.py      ← LangGraph 4-node agent pipeline
└── app.py              ← Streamlit dashboard
```

---

## Model Performance on FD001

| Metric | Our Result | XGBoost Baseline | LSTM Baseline | SOTA |
|---|---|---|---|---|
| RMSE (cycles) | **~13–16** | 13–16 | 10–13 | 6–8 |
| NASA Score S | **< 400** | ~300–500 | ~200–300 | <100 |

**RMSE interpretation:** RMSE=15 means predictions are off by ~15 flight cycles on average. Industries schedules maintenance windows of 50–200 cycles — so RMSE < 20 is operationally useful.

**NASA Score S** penalises late predictions (under-estimating RUL) more than early ones — a late maintenance call grounds an aircraft, early just costs money. Lower = better.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (~2 MB)
python download_data.py

# 3. Train model (~30 sec on GPU, ~3 min on CPU)
python model_trainer.py

# 4. Launch UI
streamlit run app.py
```

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

**Why piecewise RUL (cap at 130)?** Engines don't degrade linearly from new. They run healthy for most of their life. Capping tells the model "don't bother learning the healthy regime — focus on the last 130 cycles." Published standard since Saxena et al. (2008).

**Why these 14 sensors?** In FD001, sensors s1, s5, s6, s10, s16, s18, s19 have near-zero variance — they carry no degradation signal. Including them adds noise. This is validated by both variance analysis and feature importance from XGBoost.

**Why rolling features (5, 10, 15 cycles)?** A single cycle's sensor reading is noisy. Rolling mean/std captures *trend* — is sensor s7 increasing over the last 10 cycles? That's more informative than any single reading.

---

## References

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM'08 Conference.
- NASA C-MAPSS benchmark: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6
- LangGraph documentation: https://langchain-ai.github.io/langgraph/
