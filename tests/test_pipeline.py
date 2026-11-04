from dagster import materialize

from pipelines.definitions import ALL_ASSETS, ALL_CHECKS, defs

QUICK = {
    "xgboost_rul_model": {"config": {"n_estimators": 30, "learning_rate": 0.1, "device": "cpu",
                                     "rmse_gate": 1000.0}},
    "lstm_rul_model": {"config": {"epochs": 2, "device": "cpu", "hidden_size": 8, "rmse_gate": 1000.0}},
    "champion_model": {"config": {"min_gain": 0.5}},
}


def test_definitions_load():
    assert defs.get_job_def("retrain_rul_model") is not None
    assert defs.get_schedule_def("retrain_rul_model_schedule").cron_schedule == "0 3 * * 1"


def test_full_pipeline_trains_both_models_and_picks_a_champion():
    result = materialize(ALL_ASSETS + ALL_CHECKS, run_config={"ops": QUICK})
    assert result.success
    checks = result.get_asset_check_evaluations()
    assert len(checks) == 3 and all(c.passed for c in checks)
    decision = result.output_for_node("champion_model")
    assert decision["champion_after"]["model_type"] in {"xgboost", "lstm"}


def test_gate_fails_when_rmse_is_too_high():
    ops = {**QUICK, "xgboost_rul_model": {"config": {**QUICK["xgboost_rul_model"]["config"],
                                                     "n_estimators": 5, "rmse_gate": 0.0001}}}
    result = materialize(ALL_ASSETS + ALL_CHECKS, run_config={"ops": ops}, raise_on_error=False)
    gate = [c for c in result.get_asset_check_evaluations() if c.check_name == "xgboost_rmse_within_gate"]
    assert gate and not gate[0].passed
