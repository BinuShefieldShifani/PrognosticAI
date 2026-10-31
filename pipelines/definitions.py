"""
Dagster pipeline for PrognosticAI
=================================
Asset graph (run with `dagster dev` from the repo root):

    cmapss_raw_files ──(check: raw_data_is_valid, blocking)──▶ rul_model
                                                                  │
                                                     (check: rmse_within_gate)

- cmapss_raw_files : downloads C-MAPSS FD001 if missing, records row/engine counts
- raw_data_is_valid: blocking data-quality check, stops training on bad data
- rul_model        : trains XGBoost, logs the run to MLflow, returns the metrics
- rmse_within_gate: fails if test RMSE regresses past the configured gate

A weekly schedule re-runs the whole graph (Monday 03:00, Stockholm time).
"""

import os

import pandas as pd
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetSelection,
    Config,
    Definitions,
    MetadataValue,
    ScheduleDefinition,
    asset,
    asset_check,
    define_asset_job,
)

import download_data
from config import COL_NAMES, DATA_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, RMSE_GATE
from data_processor import load_raw, validate_raw
from model_trainer import train

RAW_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]


@asset(group_name="prognosticai", description="NASA C-MAPSS FD001 raw sensor files")
def cmapss_raw_files(context: AssetExecutionContext) -> dict:
    missing = [f for f in RAW_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        context.log.info(f"Downloading missing files: {missing}")
        download_data.download()
    train_df, test_df = load_raw("train"), load_raw("test")
    summary = {
        "data_dir": DATA_DIR,
        "train_rows": int(len(train_df)),
        "train_engines": int(train_df["unit"].nunique()),
        "test_rows": int(len(test_df)),
        "test_engines": int(test_df["unit"].nunique()),
    }
    context.add_output_metadata(summary)
    return summary


@asset_check(asset=cmapss_raw_files, blocking=True,
             description="Schema, nulls and cycle continuity for train and test files")
def raw_data_is_valid() -> AssetCheckResult:
    problems = {}
    for split in ("train", "test"):
        issues = validate_raw(load_raw(split))
        if issues:
            problems[split] = issues
    rul_path = os.path.join(DATA_DIR, "RUL_FD001.txt")
    n_rul = len(pd.read_csv(rul_path, header=None))
    n_test = load_raw("test")["unit"].nunique()
    if n_rul != n_test:
        problems["rul"] = [f"{n_rul} RUL labels for {n_test} test engines"]
    return AssetCheckResult(
        passed=not problems,
        metadata={"problems": MetadataValue.json(problems or {}),
                  "expected_columns": len(COL_NAMES)},
    )


class TrainConfig(Config):
    n_estimators: int | None = None
    learning_rate: float | None = None
    device: str | None = None          # auto / cuda / cpu
    register_model: bool = True
    rmse_gate: float = RMSE_GATE


@asset(group_name="prognosticai", deps=[cmapss_raw_files],
       description="XGBoost RUL model, tracked in MLflow")
def rul_model(context: AssetExecutionContext, config: TrainConfig) -> dict:
    metrics = train(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        device=config.device,
        log_to_mlflow=True,
        register_model=config.register_model,
        run_name=f"dagster-{context.run_id[:8]}",
    )
    metrics["rmse_gate"] = config.rmse_gate
    context.add_output_metadata({
        "test_rmse": round(metrics["test_rmse"], 2),
        "val_rmse": round(metrics["val_rmse"], 2),
        "nasa_score": round(metrics["nasa_score"], 1),
        "best_iteration": metrics["best_iteration"],
        "mlflow_run_id": metrics["mlflow_run_id"],
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "mlflow_experiment": MLFLOW_EXPERIMENT,
    })
    return metrics


@asset_check(asset=rul_model, description="Test-set RMSE must stay under the gate")
def rmse_within_gate(rul_model: dict) -> AssetCheckResult:
    rmse, gate = rul_model["test_rmse"], rul_model["rmse_gate"]
    return AssetCheckResult(
        passed=rmse <= gate,
        severity=AssetCheckSeverity.ERROR,
        metadata={"test_rmse": round(rmse, 2), "gate": gate},
    )


retrain_job = define_asset_job("retrain_rul_model", selection=AssetSelection.all())

weekly_retrain = ScheduleDefinition(
    job=retrain_job,
    cron_schedule="0 3 * * 1",
    execution_timezone="Europe/Stockholm",
)

defs = Definitions(
    assets=[cmapss_raw_files, rul_model],
    asset_checks=[raw_data_is_valid, rmse_within_gate],
    jobs=[retrain_job],
    schedules=[weekly_retrain],
)
