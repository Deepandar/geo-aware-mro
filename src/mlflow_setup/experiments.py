# src/mlflow_setup/experiments.py

import mlflow


def setup_experiment(experiment_name: str) -> None:
    """Create or set MLflow experiment."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is None:
        mlflow.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)
