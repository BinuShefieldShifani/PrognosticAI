from dagster import materialize

from pipelines.definitions import (
    cmapss_raw_files, defs, raw_data_is_valid, rul_model, rmse_within_gate,
)


def test_definitions_load():
    assert defs.get_job_def("retrain_rul_model") is not None
    assert defs.get_schedule_def("retrain_rul_model_schedule").cron_schedule == "0 3 * * 1"


def test_full_pipeline_materializes_and_checks_pass():
    result = materialize(
        [cmapss_raw_files, rul_model, raw_data_is_valid, rmse_within_gate],
        run_config={"ops": {"rul_model": {"config": {
            "n_estimators": 30, "learning_rate": 0.1, "device": "cpu",
            "register_model": False, "rmse_gate": 1000.0,
        }}}},
    )
    assert result.success
    checks = result.get_asset_check_evaluations()
    assert len(checks) == 2 and all(c.passed for c in checks)
    assert result.output_for_node("rul_model")["test_rmse"] > 0


def test_gate_fails_when_rmse_is_too_high():
    result = materialize(
        [cmapss_raw_files, rul_model, raw_data_is_valid, rmse_within_gate],
        run_config={"ops": {"rul_model": {"config": {
            "n_estimators": 5, "learning_rate": 0.1, "device": "cpu",
            "register_model": False, "rmse_gate": 0.0001,
        }}}},
        raise_on_error=False,
    )
    gate = [c for c in result.get_asset_check_evaluations()
            if c.check_name == "rmse_within_gate"]
    assert gate and not gate[0].passed
