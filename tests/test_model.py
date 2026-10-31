import numpy as np

from data_processor import build_train_dataset
from model_trainer import engine_split, nasa_score, resolve_device


def test_nasa_score_is_zero_for_perfect_predictions():
    y = np.array([10.0, 50.0, 100.0])
    assert nasa_score(y, y) == 0.0


def test_nasa_score_penalises_late_more_than_early():
    y = np.array([50.0])
    late = nasa_score(y, y + 20)    # predicted more life than the engine has
    early = nasa_score(y, y - 20)   # predicted failure too soon
    assert late > early > 0


def test_validation_split_never_shares_an_engine():
    X, y, _, groups = build_train_dataset(return_groups=True)
    tr, val = engine_split(X, y, groups)
    assert set(groups[tr]).isdisjoint(set(groups[val]))
    assert len(tr) + len(val) == len(y)


def test_resolve_device_respects_explicit_choice():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
