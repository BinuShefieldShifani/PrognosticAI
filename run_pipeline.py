"""
run_pipeline.py — The whole pipeline in one command
====================================================
Runs the Dagster asset graph end to end (download → data check → XGBoost and
LSTM training → RMSE gates → champion/challenger promotion), then writes the
comparison report. This is the only thing the Kaggle notebook and CI call.

  python run_pipeline.py --device cuda                     # Kaggle GPU
  python run_pipeline.py --device cpu --lstm-epochs 15     # quick CPU run
  python run_pipeline.py --zip-out mlflow_run.zip          # also bundle the MLflow store

Exit code is 1 if any asset fails or any quality check fails.
"""

import argparse
import os
import shutil
import sys
import zipfile

from dagster import materialize

from config import LSTM_RMSE_GATE, PROMOTION_MIN_GAIN, RMSE_GATE


def run(device: str = "auto", xgb_estimators: int | None = None, lstm_epochs: int | None = None,
        xgb_rmse_gate: float = RMSE_GATE, lstm_rmse_gate: float = LSTM_RMSE_GATE,
        min_gain: float = PROMOTION_MIN_GAIN, lstm_hidden: int | None = None,
        report_dir: str = "reports"):
    from pipelines.definitions import ALL_ASSETS, ALL_CHECKS
    from report import write_report

    result = materialize(
        ALL_ASSETS + ALL_CHECKS,
        run_config={"ops": {
            "xgboost_rul_model": {"config": {"device": device, "n_estimators": xgb_estimators,
                                             "rmse_gate": xgb_rmse_gate}},
            "lstm_rul_model": {"config": {"device": device, "epochs": lstm_epochs,
                                          "rmse_gate": lstm_rmse_gate, "hidden_size": lstm_hidden}},
            "champion_model": {"config": {"min_gain": min_gain}},
        }},
        raise_on_error=False,
    )
    checks = result.get_asset_check_evaluations()
    decision = result.output_for_node("champion_model") if result.success else None
    report = write_report(report_dir, decision)
    return result, checks, report


def zip_tracking_store(path: str):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for item in ("mlflow.db", "mlruns", "reports"):
            if os.path.isfile(item):
                z.write(item)
            elif os.path.isdir(item):
                for root, _, files in os.walk(item):
                    for f in files:
                        z.write(os.path.join(root, f))
    print(f"Tracking store zipped → {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run the full PrognosticAI pipeline")
    p.add_argument("--device", default="auto", help="auto / cuda / cpu (mps also works for the LSTM)")
    p.add_argument("--xgb-estimators", type=int, default=None)
    p.add_argument("--lstm-epochs", type=int, default=None)
    p.add_argument("--lstm-hidden", type=int, default=None, help="only for quick tests")
    p.add_argument("--xgb-rmse-gate", type=float, default=RMSE_GATE)
    p.add_argument("--lstm-rmse-gate", type=float, default=LSTM_RMSE_GATE)
    p.add_argument("--min-gain", type=float, default=PROMOTION_MIN_GAIN)
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--zip-out", default=None)
    args = p.parse_args(argv)

    result, checks, report = run(args.device, args.xgb_estimators, args.lstm_epochs,
                                 args.xgb_rmse_gate, args.lstm_rmse_gate, args.min_gain,
                                 args.lstm_hidden, args.report_dir)

    print("\n" + "=" * 58)
    print(f"  Pipeline run: {'SUCCESS' if result.success else 'FAILED'}")
    for c in checks:
        print(f"  check {c.check_name:<26} {'PASSED' if c.passed else 'FAILED'}")
    print("=" * 58 + "\n")
    print(open(report).read())
    if args.zip_out:
        zip_tracking_store(args.zip_out)
    if shutil.which("nvidia-smi") is None and args.device == "cuda":
        print("Note: --device cuda was requested but no NVIDIA GPU is visible.")
    return 0 if result.success and all(c.passed for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
