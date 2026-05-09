# tests/unit/test_mlflow_setup.py

from unittest.mock import patch


@patch("mlflow.start_run")
@patch("mlflow.log_param")
@patch("mlflow.log_metric")
def test_dummy_run_executes(
    mock_metric,
    mock_param,
    mock_run,
):

    from src.utils.mlflow_setup import (
        setup_mlflow
    )

    setup_mlflow()

    assert True