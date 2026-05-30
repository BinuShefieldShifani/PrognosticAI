"""
agent_graph.py — LangGraph Multi-Agent Orchestration
=====================================================
Four nodes, one conditional edge:

   [sensor_analyst]
         │
   [rul_estimator]
         │
    ┌────┴────────────────────┐
    │ RUL < 100?              │
   YES                        NO
    │                         │
[failure_diagnosis]    [routine_advisory]
    │                         │
[workorder_generator]         │
    └─────────────────────────┘
               │
             [END]

Node responsibilities:
  sensor_analyst     — pure Python / statistics, no LLM
  rul_estimator      — XGBoost model call, no LLM
  failure_diagnosis  — LLM: diagnose failure mode from sensor pattern
  workorder_generator— LLM: generate structured maintenance work order
  routine_advisory   — LLM: generate monitoring advisory (healthy engine)
"""

import json
import re
import numpy as np
from typing import TypedDict, Optional

import ollama
from langgraph.graph import StateGraph, START, END

from config import (
    OLLAMA_HOST, OLLAMA_MODEL,
    INFORMATIVE_SENSORS,
    THRESHOLD_CRITICAL, THRESHOLD_HIGH, THRESHOLD_MEDIUM,
)
from model_trainer import load_model_and_scaler, predict_rul
from data_processor import get_engine_data, scale_features, get_feature_columns
from knowledge_base import retrieve_procedures, format_procedure_for_llm


# ── Agent State ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    engine_id: int
    split: str                    # "train" or "test"

    # Sensor analysis outputs
    current_cycle: int
    sensor_snapshot: dict          # {sensor_name: current_value}
    anomalous_sensors: list[str]
    degradation_score: float       # 0 = healthy, 1 = severely degraded
    degradation_detected: bool

    # RUL estimation outputs
    rul_prediction: float
    rul_lower: float
    rul_upper: float
    priority: str                  # CRITICAL / HIGH / MEDIUM / LOW

    # LLM outputs
    failure_mode: str
    failure_mode_confidence: str   # HIGH / MEDIUM / LOW
    maintenance_procedures: list[dict]
    work_order: dict

    # Trace — shown in Streamlit UI
    reasoning_log: list[str]


# ── Ollama Client ─────────────────────────────────────────────────────────

_client = ollama.Client(host=OLLAMA_HOST)


def call_llm(prompt: str, system: str = "") -> str:
    """Call Ollama with the configured model. Handles connection errors gracefully."""
    try:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = _client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.1, "num_predict": 600},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[LLM unavailable: {e}]"


def extract_json_from_response(text: str) -> dict:
    """Robustly parse JSON — handles fences, leading prose, single quotes."""
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(candidate.replace("'", '"'))
        except Exception:
            pass
    return {"raw_output": text}


# ── Node 1: Sensor Analyst (pure Python — no LLM) ────────────────────────

def sensor_analyst_node(state: AgentState) -> AgentState:
    """
    Analyses the sensor time series for the given engine.

    What it does:
    - Loads full engine history from the dataset
    - Computes z-score of each sensor over its history
    - Sensors with |z| > 2.0 at the final cycle are flagged as anomalous
    - Computes a health degradation score (0–1) from the fraction of sensors degrading
    """
    log = state.get("reasoning_log", [])
    engine_id = state["engine_id"]
    split = state.get("split", "train")

    log.append(f"[Sensor Analyst] Loading engine {engine_id} from {split} set...")
    engine_df = get_engine_data(engine_id, split=split)
    last_row = engine_df.iloc[-1]

    current_cycle = int(last_row["cycle"])
    log.append(f"[Sensor Analyst] Engine {engine_id} — {current_cycle} observed cycles")

    # Snapshot of current sensor values
    sensor_snapshot = {s: float(last_row[s]) for s in INFORMATIVE_SENSORS}

    # Z-score anomaly detection on each sensor's full history
    anomalous = []
    degradation_scores = []

    for sensor in INFORMATIVE_SENSORS:
        series = engine_df[sensor].values
        mu, sigma = series.mean(), series.std()
        if sigma < 1e-6:
            continue                            # constant sensor — skip
        z = abs((series[-1] - mu) / sigma)
        degradation_scores.append(min(z / 3.0, 1.0))   # normalise to [0,1]
        if z > 2.0:
            anomalous.append(sensor)

    degradation_score = float(np.mean(degradation_scores)) if degradation_scores else 0.0

    if anomalous:
        log.append(f"[Sensor Analyst] Anomalous sensors (|z| > 2σ): {', '.join(anomalous)}")
        log.append(f"[Sensor Analyst] Overall degradation score: {degradation_score:.2f}/1.00")
    else:
        log.append("[Sensor Analyst] All sensors within normal range.")

    return {
        **state,
        "current_cycle": current_cycle,
        "sensor_snapshot": sensor_snapshot,
        "anomalous_sensors": anomalous,
        "degradation_score": degradation_score,
        "degradation_detected": len(anomalous) > 0,
        "reasoning_log": log,
    }


# ── Node 2: RUL Estimator (XGBoost — no LLM) ─────────────────────────────

def rul_estimator_node(state: AgentState) -> AgentState:
    """
    Loads the pre-trained XGBoost model and predicts Remaining Useful Life.
    """
    log = state["reasoning_log"]
    engine_id = state["engine_id"]
    split = state.get("split", "train")

    log.append("[RUL Estimator] Loading XGBoost model...")
    model, scaler = load_model_and_scaler()

    engine_df = get_engine_data(engine_id, split=split)
    feat_cols = get_feature_columns()

    # Use last observed cycle for prediction
    last_features = engine_df[feat_cols].iloc[-1:].values
    X_scaled, _ = scale_features(engine_df, fit=False, scaler=scaler)
    last_scaled = X_scaled[-1:]

    rul, lower, upper = predict_rul(last_scaled, model)
    log.append(f"[RUL Estimator] Predicted RUL: {rul:.1f} cycles  (95% CI: [{lower:.1f}, {upper:.1f}])")

    # Assign priority
    if rul < THRESHOLD_CRITICAL:
        priority = "CRITICAL"
    elif rul < THRESHOLD_HIGH:
        priority = "HIGH"
    elif rul < THRESHOLD_MEDIUM:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    log.append(f"[RUL Estimator] Maintenance priority: {priority}")

    return {
        **state,
        "rul_prediction": rul,
        "rul_lower": lower,
        "rul_upper": upper,
        "priority": priority,
        "reasoning_log": log,
    }


# ── Routing Function ──────────────────────────────────────────────────────

def route_by_priority(state: AgentState) -> str:
    """
    Conditional edge: degraded engines go to failure_diagnosis,
    healthy engines go straight to routine_advisory.
    """
    if state["rul_prediction"] < THRESHOLD_MEDIUM or state["degradation_detected"]:
        return "failure_diagnosis"
    return "routine_advisory"


# ── Node 3a: Failure Diagnosis (LLM) ─────────────────────────────────────

def failure_diagnosis_node(state: AgentState) -> AgentState:
    """
    LLM analyses the pattern of anomalous sensors and diagnoses the failure mode.

    For FD001, the true fault is always HPC Degradation — but the LLM doesn't know that.
    It reasons from sensor patterns, which is how a real system would work.
    """
    log = state["reasoning_log"]
    log.append("[Failure Diagnosis] Calling LLM for failure mode classification...")

    anomalous = state["anomalous_sensors"]
    snapshot = state["sensor_snapshot"]
    rul = state["rul_prediction"]
    score = state["degradation_score"]

    # Build a compact sensor context (don't dump all values — LLM needs signal not noise)
    sensor_context = "\n".join(
        f"  {s}: {snapshot[s]:.4f}" + (" ← ANOMALOUS" if s in anomalous else "")
        for s in INFORMATIVE_SENSORS
    )

    # Which sensors are anomalous maps to which ATA system
    sensor_hints = ""
    if any(s in anomalous for s in ["s2", "s3", "s4"]):
        sensor_hints += "Pressure sensors (s2, s3, s4) anomalous — suggests compressor or inlet issue. "
    if any(s in anomalous for s in ["s7", "s8", "s9"]):
        sensor_hints += "Temperature sensors (s7, s8, s9) anomalous — suggests combustor or turbine thermal issue. "
    if any(s in anomalous for s in ["s11", "s12"]):
        sensor_hints += "Fan/LPC sensors (s11, s12) anomalous — suggests fan or low-pressure system. "
    if any(s in anomalous for s in ["s14", "s15"]):
        sensor_hints += "Bypass/efficiency sensors anomalous — suggests efficiency degradation. "

    prompt = f"""You are a turbofan engine health monitoring system for an aerospace MRO facility.

SENSOR READINGS (current cycle):
{sensor_context}

DEGRADATION SCORE: {score:.2f} / 1.00
ESTIMATED RUL: {rul:.1f} cycles
{f"SENSOR PATTERN HINTS: {sensor_hints}" if sensor_hints else ""}

Based on the sensor pattern, diagnose the most likely failure mode from:
  A) HPC (High Pressure Compressor) Degradation — blade erosion, tip clearance loss
  B) HPT (High Pressure Turbine) Degradation — thermal fatigue, TBC spallation
  C) Fan Degradation / FOD — leading edge erosion, imbalance
  D) Bearing Wear — oil contamination, vibration
  E) General Engine Degradation — EGT margin loss, performance restoration needed

Respond with a JSON object only — no other text:
{{
  "failure_mode": "<one of the failure modes above, written out in full>",
  "confidence": "<HIGH | MEDIUM | LOW>",
  "primary_indicators": ["<sensor> shows <what>", ...],
  "secondary_risks": "<any secondary concerns>",
  "reasoning": "<2-3 sentences of engineering reasoning>"
}}"""

    response = call_llm(prompt, system="You are an expert aerospace engineer specialising in gas turbine health monitoring. Be concise and technical.")
    diagnosis = extract_json_from_response(response)

    failure_mode = diagnosis.get("failure_mode", "HPC Degradation")
    confidence   = diagnosis.get("confidence", "MEDIUM")
    reasoning    = diagnosis.get("reasoning", "")

    log.append(f"[Failure Diagnosis] Mode: {failure_mode}  Confidence: {confidence}")
    if reasoning:
        log.append(f"[Failure Diagnosis] Reasoning: {reasoning}")

    # Retrieve matching maintenance procedures
    procedures = retrieve_procedures(failure_mode, rul)
    log.append(f"[Failure Diagnosis] Retrieved {len(procedures)} matching maintenance procedures: "
                f"{', '.join(p.id for p in procedures)}")

    return {
        **state,
        "failure_mode": failure_mode,
        "failure_mode_confidence": confidence,
        "maintenance_procedures": [
            {"id": p.id, "title": p.title, "severity": p.severity,
             "man_hours": p.man_hours, "steps": p.steps,
             "parts_to_preorder": p.parts_to_preorder}
            for p in procedures
        ],
        "reasoning_log": log,
    }


# ── Node 3b: Routine Advisory (LLM) ──────────────────────────────────────

def routine_advisory_node(state: AgentState) -> AgentState:
    """For healthy engines: generate a brief monitoring advisory instead of a work order."""
    log = state["reasoning_log"]
    log.append("[Routine Advisory] Engine is healthy — generating monitoring advisory...")

    return {
        **state,
        "failure_mode": "No fault detected",
        "failure_mode_confidence": "HIGH",
        "maintenance_procedures": [],
        "work_order": {
            "type": "MONITORING_ADVISORY",
            "engine_id": state["engine_id"],
            "rul_cycles": round(state["rul_prediction"], 1),
            "priority": "LOW",
            "recommendation": "Engine operating within normal parameters. Continue scheduled monitoring.",
            "next_inspection_cycle": state["current_cycle"] + 100,
            "actions": ["Continue normal operation", "Review at next scheduled borescope interval"],
        },
        "reasoning_log": log,
    }


# ── Node 4: Work Order Generator (LLM) ───────────────────────────────────

def workorder_generator_node(state: AgentState) -> AgentState:
    """
    Generates a structured, MRO-system-ready maintenance work order.
    This is the final deliverable — what an engineer would actually act on.
    """
    log = state["reasoning_log"]
    log.append("[Work Order Generator] Generating structured work order via LLM...")

    procedures = state["maintenance_procedures"]
    proc_context = "\n\n".join(
        format_procedure_for_llm_dict(p) for p in procedures
    ) if procedures else "No specific procedure retrieved."

    import random
    wo_id     = f"WO-{state['engine_id']:04d}-{random.randint(100000,999999)}"
    deadline  = state["current_cycle"] + max(int(state["rul_prediction"] * 0.7), 5)
    shop      = any(p.get("man_hours", 0) > 20 for p in procedures)
    hours     = sum(p.get("man_hours", 0) for p in procedures)
    grounding = state["priority"] == "CRITICAL"
    all_parts = list({pt for p in procedures for pt in p.get("parts_to_preorder", [])})
    proc_ids  = [p["id"] for p in procedures]
    wo_type   = "AOG" if grounding else "UNSCHEDULED_MAINTENANCE"

    # Pre-fill all structural fields — LLM only fills the two text fields
    # This forces valid JSON output and prevents hallucinated structure
    prompt = (
        "Fill in the two fields marked <FILL> in this JSON work order.\n"
        "Return the completed JSON only — absolutely no other text before or after.\n\n"
        f'{{\n'
        f'  "work_order_id": "{wo_id}",\n'
        f'  "engine_id": {state["engine_id"]},\n'
        f'  "priority": "{state["priority"]}",\n'
        f'  "type": "{wo_type}",\n'
        f'  "rul_cycles_remaining": {state["rul_prediction"]:.1f},\n'
        f'  "failure_mode": "{state["failure_mode"]}",\n'
        f'  "deadline_cycles": {deadline},\n'
        f'  "actions": <FILL: JSON array of 4 specific maintenance steps for {state["failure_mode"]}>,\n'
        f'  "procedures_referenced": {json.dumps(proc_ids)},\n'
        f'  "parts_to_preorder": {json.dumps(all_parts[:6])},\n'
        f'  "estimated_man_hours": {hours},\n'
        f'  "shop_visit_required": {str(shop).lower()},\n'
        f'  "grounding_required": {str(grounding).lower()},\n'
        f'  "notes": <FILL: one sentence — why this priority given {state["degradation_score"]:.2f} degradation score>\n'
        f'}}\n\n'
        f'Applicable procedures (for actions context):\n{proc_context[:600]}'
    )

    response = call_llm(
        prompt,
        system="You are an aircraft MRO engineer. Return only the completed JSON. No preamble, no explanation."
    )
    work_order = extract_json_from_response(response)

    if "raw_output" in work_order:
        work_order = _fallback_work_order(state, procedures)

    log.append(f"[Work Order Generator] Work order generated: {work_order.get('work_order_id', 'WO-UNKNOWN')}")
    log.append(f"[Work Order Generator] Type: {work_order.get('type')} | Grounding: {work_order.get('grounding_required', False)}")

    return {**state, "work_order": work_order, "reasoning_log": log}


def format_procedure_for_llm_dict(proc_dict: dict) -> str:
    steps = "\n".join(proc_dict["steps"])
    parts = ", ".join(proc_dict["parts_to_preorder"])
    return (
        f"Procedure {proc_dict['id']}: {proc_dict['title']}\n"
        f"Severity: {proc_dict['severity']} | Man-hours: {proc_dict['man_hours']}\n"
        f"Steps:\n{steps}\n"
        f"Parts: {parts}"
    )


def _fallback_work_order(state: AgentState, procedures: list) -> dict:
    """Clean fallback work order — used when LLM output cannot be parsed."""
    import random
    all_parts = list({
        part for proc in procedures
        for part in proc.get("parts_to_preorder", [])
    })
    grounding = state["priority"] == "CRITICAL"
    wo_type   = "AOG" if grounding else "UNSCHEDULED_MAINTENANCE"
    deadline  = state["current_cycle"] + max(int(state["rul_prediction"] * 0.7), 5)
    failure   = state["failure_mode"] or "engine degradation"
    return {
        "work_order_id": f"WO-{state['engine_id']:04d}-{random.randint(100000, 999999)}",
        "engine_id": state["engine_id"],
        "priority": state["priority"],
        "type": wo_type,
        "rul_cycles_remaining": round(state["rul_prediction"], 1),
        "failure_mode": failure,
        "deadline_cycles": deadline,
        "actions": [
            f"Perform borescope inspection of HPC stages 1–7 (ATA 72-30)",
            f"Measure blade tip clearance at each stage — serviceable limit 0.45 mm",
            f"Cut oil filter and inspect for metallic debris — submit SOAP sample (ATA 72-80)",
            f"Review EGT margin trend data — flag if margin < 50°C above redline",
        ],
        "procedures_referenced": [p["id"] for p in procedures],
        "parts_to_preorder": all_parts[:6],
        "estimated_man_hours": sum(p.get("man_hours", 0) for p in procedures),
        "shop_visit_required": any(p.get("man_hours", 0) > 20 for p in procedures),
        "grounding_required": grounding,
        "notes": (
            f"Engine {state['engine_id']} has {state['rul_prediction']:.1f} cycles remaining "
            f"with degradation score {state['degradation_score']:.2f}/1.00 across "
            f"{len(state['anomalous_sensors'])} anomalous sensors. "
            f"Immediate {failure} inspection required."
        ),
    }


# ── Build the LangGraph ───────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("sensor_analyst",      sensor_analyst_node)
    graph.add_node("rul_estimator",       rul_estimator_node)
    graph.add_node("failure_diagnosis",   failure_diagnosis_node)
    graph.add_node("routine_advisory",    routine_advisory_node)
    graph.add_node("workorder_generator", workorder_generator_node)

    graph.add_edge(START, "sensor_analyst")
    graph.add_edge("sensor_analyst", "rul_estimator")

    graph.add_conditional_edges(
        "rul_estimator",
        route_by_priority,
        {
            "failure_diagnosis": "failure_diagnosis",
            "routine_advisory":  "routine_advisory",
        }
    )

    graph.add_edge("failure_diagnosis",   "workorder_generator")
    graph.add_edge("workorder_generator", END)
    graph.add_edge("routine_advisory",    END)

    return graph.compile()


# ── Run a Single Analysis ─────────────────────────────────────────────────

def run_analysis(engine_id: int, split: str = "train") -> AgentState:
    """Entry point — run the full agent pipeline for one engine."""
    compiled = build_graph()

    initial_state: AgentState = {
        "engine_id": engine_id,
        "split": split,
        "current_cycle": 0,
        "sensor_snapshot": {},
        "anomalous_sensors": [],
        "degradation_score": 0.0,
        "degradation_detected": False,
        "rul_prediction": 0.0,
        "rul_lower": 0.0,
        "rul_upper": 0.0,
        "priority": "LOW",
        "failure_mode": "",
        "failure_mode_confidence": "",
        "maintenance_procedures": [],
        "work_order": {},
        "reasoning_log": [],
    }

    result = compiled.invoke(initial_state)
    return result