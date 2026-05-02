# tests/unit/test_mlflow_setup.py

from src.mlflow_setup.dummy_run import run_dummy_experiment


def test_dummy_run_executes():
    run_dummy_experiment()
