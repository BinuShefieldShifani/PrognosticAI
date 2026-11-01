import json
import os

import numpy as np
import pandas as pd
import pytest

import data_processor
from config import INFORMATIVE_SENSORS, PREDICTOR_DIR
from data_processor import build_test_dataset, load_raw
from model_trainer import load_model_and_scaler, train
from rul_predictor import OUTPUT_COLUMNS, load_predictor, validate_input


@pytest.fixture(scope="module")
def trained():
    train(n_estimators=30, learning_rate=0.1, device="cpu", log_to_mlflow=False)
    return load_predictor()


def test_packaged_model_matches_training_pipeline(trained):
    """Raw cycles through the package must give the same RUL as the training code path."""
    model, scaler = load_model_and_scaler()
    X_test, _ = build_test_dataset(scaler)
    expected = model.predict(X_test).clip(0)

    out = trained.predict(load_raw("test"))
    assert list(out.columns) == OUTPUT_COLUMNS
    np.testing.assert_allclose(out["rul"].values, expected, rtol=1e-5, atol=1e-4)


def test_one_row_per_engine_in_unit_order(trained):
    raw = load_raw("test")
    shuffled = raw[raw["unit"].isin([3, 1, 2])].sort_values(["unit", "cycle"], ascending=[False, True])
    out = trained.predict(shuffled)
    assert list(out["unit"]) == [1, 2, 3]
    last = raw[raw["unit"].isin([1, 2, 3])].groupby("unit")["cycle"].max().values
    assert list(out["last_cycle"]) == list(last)
    assert (out["rul_lower"] <= out["rul"]).all() and (out["rul"] <= out["rul_upper"]).all()


def test_extra_columns_are_ignored(trained):
    raw = load_raw("test")
    one = raw[raw["unit"] == 1]
    trimmed = one[["unit", "cycle"] + INFORMATIVE_SENSORS]
    assert trained.predict(one)["rul"].iloc[0] == pytest.approx(trained.predict(trimmed)["rul"].iloc[0])


def test_short_history_is_flagged(trained):
    one = load_raw("test").query("unit == 1")
    out = trained.predict(one.tail(20))
    assert bool(out["short_history"].iloc[0]) is True
    assert out["history_cycles"].iloc[0] == 20


def test_missing_sensor_is_rejected_with_a_clear_message(trained):
    one = load_raw("test").query("unit == 1").drop(columns=["s4"])
    with pytest.raises(Exception, match="s4"):
        trained.predict(one)


def test_validate_input_catches_gaps_and_nulls():
    one = load_raw("test").query("unit == 1").reset_index(drop=True)
    gap = one.drop(index=5)
    assert any("consecutive" in p for p in validate_input(gap, INFORMATIVE_SENSORS))
    nulls = one.copy()
    nulls.loc[3, "s7"] = np.nan
    assert any("null" in p for p in validate_input(nulls, INFORMATIVE_SENSORS))
    assert validate_input(one, INFORMATIVE_SENSORS) == []


def test_package_uses_its_own_feature_spec_not_current_config(trained, monkeypatch):
    """A config change after training must not change what an existing model computes."""
    one = load_raw("test").query("unit == 1")
    before = trained.predict(one)["rul"].iloc[0]

    monkeypatch.setattr(data_processor, "ROLLING_WINDOWS", [3])
    monkeypatch.setattr(data_processor, "SLOPE_WINDOWS", [5])
    after = load_predictor().predict(one)["rul"].iloc[0]
    assert after == pytest.approx(before)

    with open(os.path.join(PREDICTOR_DIR, "artifacts", "feature_spec.json")) as f:
        spec = json.load(f)
    assert spec["slope_windows"] == [10, 30, 50, 80]


def test_signature_describes_raw_sensor_input(trained):
    names = trained.metadata.get_input_schema().input_names()
    assert names[:2] == ["unit", "cycle"]
    assert set(INFORMATIVE_SENSORS) <= set(names)
    assert isinstance(trained.predict(load_raw("test").query("unit == 2")), pd.DataFrame)
