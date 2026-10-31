import numpy as np
import pandas as pd

from config import COL_NAMES, INFORMATIVE_SENSORS, MAX_RUL, ROLLING_WINDOWS, SLOPE_WINDOWS
from data_processor import (
    add_rul_labels, engineer_features, get_feature_columns, load_raw, validate_raw,
)


def test_rul_labels_are_capped_and_reach_zero_at_failure():
    df = add_rul_labels(load_raw("train"))
    assert df["rul"].max() == MAX_RUL
    last_rows = df.loc[df.groupby("unit")["cycle"].idxmax()]
    assert (last_rows["rul"] == 0).all()


def test_feature_matrix_shape_and_no_nans():
    feats = get_feature_columns()
    per_sensor = 1 + 2 * len(ROLLING_WINDOWS) + len(SLOPE_WINDOWS)
    expected = len(INFORMATIVE_SENSORS) * per_sensor + 1  # + cycle_norm
    assert len(feats) == expected == 183
    df = engineer_features(load_raw("train"))
    assert not df[feats].isna().any().any()


def test_cycle_norm_does_not_leak_engine_lifetime():
    # The last observed cycle of a partial (test) engine must not look like end of life
    df = engineer_features(load_raw("test"))
    last = df.loc[df.groupby("unit")["cycle"].idxmax()]
    assert (last["cycle_norm"] < 1.0).all()


def test_validate_raw_accepts_clean_data():
    assert validate_raw(load_raw("train")) == []


def test_validate_raw_flags_nulls_and_gaps():
    df = load_raw("train").copy()
    df.loc[3, "s4"] = np.nan
    df = df.drop(index=10)  # breaks the 1..N cycle sequence for engine 1
    problems = validate_raw(df)
    assert any("null" in p for p in problems)
    assert any("not a 1..N sequence" in p for p in problems)


def test_validate_raw_flags_missing_columns():
    df = pd.DataFrame(columns=COL_NAMES[:-1])
    assert any("missing columns" in p for p in validate_raw(df))


def test_slope_matches_a_known_linear_trend():
    from data_processor import _rolling_slope
    t = pd.Series(np.arange(1, 101, dtype=float))
    x = pd.Series(3.0 + 0.5 * t)           # slope exactly 0.5 per cycle
    s = _rolling_slope(x, t, 30)
    assert s[0] == 0.0                     # one point: no slope yet
    assert np.allclose(s[1:], 0.5)


def test_slope_only_uses_past_cycles():
    from data_processor import _rolling_slope
    t = pd.Series(np.arange(1, 61, dtype=float))
    flat = pd.Series(np.ones(60))
    jump = flat.copy()
    jump.iloc[40:] = 10.0                  # change happens at cycle 41
    a, b = _rolling_slope(flat, t, 30), _rolling_slope(jump, t, 30)
    assert np.allclose(a[:40], b[:40])     # earlier rows must not see the future
