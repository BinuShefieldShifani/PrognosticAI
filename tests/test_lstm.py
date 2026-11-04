import numpy as np
import pandas as pd
import pytest

from config import LSTM_PREDICTOR_DIR
from data_processor import build_train_dataset, load_raw
from lstm_model import build_windows, lstm_spec
from lstm_trainer import prepare_data, train_lstm
from model_trainer import engine_split
from rul_predictor import OUTPUT_COLUMNS, load_predictor, model_type

torch = pytest.importorskip("torch")


def test_windows_are_padded_at_the_front_and_end_on_each_cycle():
    df = pd.DataFrame({"unit": [1, 1, 1, 2, 2], "cycle": [1, 2, 3, 1, 2]})
    X = np.arange(10, dtype=float).reshape(5, 2)
    windows, order = build_windows(df, X, seq_len=3)
    assert windows.shape == (5, 3, 2)
    np.testing.assert_array_equal(windows[0], [X[0], X[0], X[0]])     # first cycle: all padding
    np.testing.assert_array_equal(windows[2], [X[0], X[1], X[2]])     # full window
    np.testing.assert_array_equal(windows[4], [X[3], X[3], X[4]])     # engine 2 never sees engine 1
    assert list(order) == [0, 1, 2, 3, 4]


def test_lstm_uses_the_same_validation_engines_as_xgboost():
    X, y, _, groups_xgb = build_train_dataset(return_groups=True)
    _, val_xgb = engine_split(X, y, groups_xgb)
    _, y_l, groups_l, *_ = prepare_data(lstm_spec())
    _, val_lstm = engine_split(np.zeros(len(y_l)), y_l, groups_l)
    assert set(groups_xgb[val_xgb]) == set(groups_l[val_lstm])


@pytest.fixture(scope="module")
def lstm_trained():
    metrics = train_lstm(epochs=2, device="cpu", log_to_mlflow=True, register_model=False,
                         run_name="pytest-lstm", hidden_size=8)
    return metrics, load_predictor(LSTM_PREDICTOR_DIR)


def test_lstm_training_logs_metrics(lstm_trained):
    metrics, _ = lstm_trained
    for key in ("val_rmse", "test_rmse", "test_mae", "nasa_score"):
        assert np.isfinite(metrics[key])
    assert metrics["epochs_run"] == 2 and metrics["mlflow_run_id"]


def test_lstm_package_has_the_same_contract(lstm_trained):
    _, predictor = lstm_trained
    raw = load_raw("test")
    out = predictor.predict(raw[raw["unit"].isin([2, 1])])
    assert model_type(predictor) == "lstm"
    assert list(out.columns) == OUTPUT_COLUMNS
    assert list(out["unit"]) == [1, 2] and (out["rul"] >= 0).all()


def test_lstm_all_cycles_ends_on_the_default_prediction(lstm_trained):
    _, predictor = lstm_trained
    one = load_raw("test").query("unit == 3")
    every = predictor.predict(one, params={"all_cycles": True})
    assert len(every) == len(one)
    assert every["rul"].iloc[-1] == pytest.approx(predictor.predict(one)["rul"].iloc[0], rel=1e-5)
