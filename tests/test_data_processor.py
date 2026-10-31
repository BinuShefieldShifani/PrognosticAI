import numpy as np
import pandas as pd

from config import COL_NAMES, INFORMATIVE_SENSORS, MAX_RUL, ROLLING_WINDOWS
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
    expected = len(INFORMATIVE_SENSORS) * (1 + 2 * len(ROLLING_WINDOWS)) + 1  # + cycle_norm
    assert len(feats) == expected == 99
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
