"""
Dagster pipeline for PrognosticAI
=================================
Asset graph (run with `dagster dev` from the repo root):

    cmapss_raw_files ──(check: raw_data_is_valid, blocking)──┬──▶ xgboost_rul_model ──┐
                                                              │   (check: xgboost_rmse_within_gate)
                                                              └──▶ lstm_rul_model ─────┤
                                                                  (check: lstm_rmse_within_gate)
                                                                                       ▼
                                                                                champion_model

- cmapss_raw_files  : downloads C-MAPSS FD001 if missing, records row/engine counts
- raw_data_is_valid : blocking data-quality check, stops training on bad data
- xgboost_rul_model : trains XGBoost, logs to MLflow, registers it as challenger-xgboost
- lstm_rul_model    : trains the PyTorch LSTM on the same split, registers it as challenger-lstm
- champion_model    : compares challengers with the champion on validation RMSE and
                      moves the champion alias if one is clearly better (registry.py)

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
from config import (
    COL_NAMES, DATA_DIR, LSTM_RMSE_GATE, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI,
    PROMOTION_MIN_GAIN, RMSE_GATE,
)
from data_processor import load_raw, validate_raw
from model_trainer import train
from registry import format_decision, promote

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


def _metrics_metadata(metrics: dict) -> dict:
    keep = ("test_rmse", "val_rmse", "nasa_score", "test_mae")
    meta = {k: round(metrics[k], 2) for k in keep}
    meta.update({"mlflow_run_id": metrics.get("mlflow_run_id", "-"),
                 "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
                 "mlflow_experiment": MLFLOW_EXPERIMENT})
    return meta


class XGBoostConfig(Config):
    n_estimators: int | None = None
    learning_rate: float | None = None
    device: str | None = None          # auto / cuda / cpu
    register_model: bool = True
    rmse_gate: float = RMSE_GATE


@asset(group_name="prognosticai", deps=[cmapss_raw_files],
       description="XGBoost RUL model, tracked in MLflow, registered as challenger-xgboost")
def xgboost_rul_model(context: AssetExecutionContext, config: XGBoostConfig) -> dict:
    metrics = train(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        device=config.device,
        log_to_mlflow=True,
        register_model=config.register_model,
        run_name=f"dagster-xgb-{context.run_id[:8]}",
        rmse_gate=config.rmse_gate,
    )
    metrics["rmse_gate"] = config.rmse_gate
    context.add_output_metadata({**_metrics_metadata(metrics), "best_iteration": metrics["best_iteration"]})
    return metrics


class LSTMConfig(Config):
    epochs: int | None = None
    device: str | None = None          # auto / cuda / cpu / mps
    register_model: bool = True
    rmse_gate: float = LSTM_RMSE_GATE
    hidden_size: int | None = None     # smaller values only for quick tests


@asset(group_name="prognosticai", deps=[cmapss_raw_files],
       description="PyTorch LSTM RUL model on the same split, registered as challenger-lstm")
def lstm_rul_model(context: AssetExecutionContext, config: LSTMConfig) -> dict:
    from lstm_trainer import train_lstm

    metrics = train_lstm(
        epochs=config.epochs,
        device=config.device,
        log_to_mlflow=True,
        register_model=config.register_model,
        run_name=f"dagster-lstm-{context.run_id[:8]}",
        rmse_gate=config.rmse_gate,
        hidden_size=config.hidden_size,
    )
    metrics["rmse_gate"] = config.rmse_gate
    context.add_output_metadata({**_metrics_metadata(metrics), "best_epoch": metrics["best_epoch"]})
    return metrics


def _gate_result(metrics: dict) -> AssetCheckResult:
    rmse, gate = metrics["test_rmse"], metrics["rmse_gate"]
    return AssetCheckResult(
        passed=rmse <= gate,
        severity=AssetCheckSeverity.ERROR,
        metadata={"test_rmse": round(rmse, 2), "gate": gate},
    )


@asset_check(asset=xgboost_rul_model, description="XGBoost test RMSE must stay under its gate")
def xgboost_rmse_within_gate(xgboost_rul_model: dict) -> AssetCheckResult:
    return _gate_result(xgboost_rul_model)


@asset_check(asset=lstm_rul_model, description="LSTM test RMSE must stay under its gate")
def lstm_rmse_within_gate(lstm_rul_model: dict) -> AssetCheckResult:
    return _gate_result(lstm_rul_model)


class PromotionConfig(Config):
    min_gain: float = PROMOTION_MIN_GAIN
    dry_run: bool = False


@asset(group_name="prognosticai", deps=[xgboost_rul_model, lstm_rul_model],
       description="Champion/challenger decision on validation RMSE; moves the champion alias")
def champion_model(context: AssetExecutionContext, config: PromotionConfig) -> dict:
    decision = promote(min_gain=config.min_gain, dry_run=config.dry_run)
    after = decision["champion_after"] or {}
    context.add_output_metadata({
        "promoted": decision["promoted"],
        "reason": decision["reason"],
        "champion": f"{after.get('model_type', '-')} v{after.get('version', '-')}",
        "decision": MetadataValue.md(format_decision(decision)),
    })
    return decision


ALL_ASSETS = [cmapss_raw_files, xgboost_rul_model, lstm_rul_model, champion_model]
ALL_CHECKS = [raw_data_is_valid, xgboost_rmse_within_gate, lstm_rmse_within_gate]

retrain_job = define_asset_job("retrain_rul_model", selection=AssetSelection.all())

weekly_retrain = ScheduleDefinition(
    job=retrain_job,
    cron_schedule="0 3 * * 1",
    execution_timezone="Europe/Stockholm",
)

defs = Definitions(
    assets=ALL_ASSETS,
    asset_checks=ALL_CHECKS,
    jobs=[retrain_job],
    schedules=[weekly_retrain],
)
