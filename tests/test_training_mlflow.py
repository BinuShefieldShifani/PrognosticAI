import mlflow
import numpy as np

from config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from data_processor import load_raw
from model_trainer import load_model_and_scaler, train


def test_training_run_is_logged_to_mlflow():
    metrics = train(n_estimators=30, learning_rate=0.1, device="cpu",
                    log_to_mlflow=True, register_model=True, run_name="pytest")

    for key in ("val_rmse", "test_rmse", "test_mae", "nasa_score"):
        assert np.isfinite(metrics[key])

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run = mlflow.get_run(metrics["mlflow_run_id"])
    assert run.data.params["max_depth"] == "5"
    assert run.data.params["slope_windows"] == "10,30,50,80"
    assert run.data.tags["split_strategy"].startswith("engine")
    assert abs(run.data.metrics["test_rmse"] - metrics["test_rmse"]) < 1e-6

    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id, "evaluation")}
    assert {"evaluation/feature_importance.csv", "evaluation/test_predictions.csv"} <= artifacts

    # Logged model is the packaged predictor: raw cycles in, one row per engine out
    model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
    raw = load_raw("test")
    out = model.predict(raw[raw["unit"].isin([1, 2])])
    assert list(out["unit"]) == [1, 2]
    assert (out["rul"] >= 0).all()

    # The joblib copy used by the Streamlit app is still written
    xgb_model, scaler = load_model_and_scaler()
    assert xgb_model is not None and scaler is not None
    assert mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is not None
