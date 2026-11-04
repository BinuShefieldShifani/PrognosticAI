"""Promotion rules, tested on a throwaway registered model with controlled metrics."""

import uuid

import mlflow
import pytest

from config import CHAMPION_ALIAS, MLFLOW_TRACKING_URI
from registry import candidates, promote, register_run


class _Const(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        return model_input


def _version(name, kind, val, test, gate=15.0):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("registry-tests")
    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=_Const())
    metrics = {"val_rmse": val, "test_rmse": test, "nasa_score": 300.0}
    return register_run(run.info.run_id, kind, metrics, gate, name=name)


@pytest.fixture
def name():
    return f"rul-test-{uuid.uuid4().hex[:8]}"


def _champion(name):
    return mlflow.MlflowClient().get_registered_model(name).aliases.get(CHAMPION_ALIAS)


def test_first_eligible_model_becomes_champion(name):
    v = _version(name, "xgboost", val=13.0, test=12.6)
    decision = promote(min_gain=0.5, name=name)
    assert decision["promoted"] and _champion(name) == v


def test_challenger_needs_to_win_by_the_margin(name):
    v1 = _version(name, "xgboost", val=13.0, test=12.6)
    promote(min_gain=0.5, name=name)
    _version(name, "lstm", val=12.8, test=11.9)          # better, but only by 0.2
    decision = promote(min_gain=0.5, name=name)
    assert not decision["promoted"] and _champion(name) == v1
    assert "margin" in decision["reason"] and "only 0.20" in decision["reason"]


def test_clear_winner_is_promoted_and_ranked_on_validation_not_test(name):
    _version(name, "xgboost", val=13.0, test=12.6)
    promote(min_gain=0.5, name=name)
    v_lstm = _version(name, "lstm", val=11.5, test=13.4)   # worse on test, better on validation
    decision = promote(min_gain=0.5, name=name)
    assert decision["promoted"] and _champion(name) == v_lstm


def test_model_over_its_gate_is_never_promoted(name):
    _version(name, "lstm", val=5.0, test=40.0, gate=18.0)
    decision = promote(min_gain=0.5, name=name)
    assert not decision["promoted"] and _champion(name) is None


def test_dry_run_does_not_move_the_alias(name):
    _version(name, "xgboost", val=13.0, test=12.6)
    decision = promote(min_gain=0.5, dry_run=True, name=name)
    assert decision["promoted"] and decision["dry_run"] and _champion(name) is None


def test_candidates_lists_one_challenger_per_type(name):
    _version(name, "xgboost", val=13.0, test=12.6)
    _version(name, "xgboost", val=12.9, test=12.5)
    _version(name, "lstm", val=12.0, test=12.0)
    champion, challengers = candidates(name)
    assert champion is None
    assert sorted(c["model_type"] for c in challengers) == ["lstm", "xgboost"]


def test_champion_is_not_listed_again_as_a_challenger(name):
    _version(name, "xgboost", val=13.0, test=12.6)
    promote(min_gain=0.5, name=name)
    champion, challengers = candidates(name)
    assert champion is not None and challengers == []
    assert not promote(min_gain=0.5, name=name)["promoted"]
