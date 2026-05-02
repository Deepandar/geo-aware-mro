# src/mlflow_setup/dummy_run.py

from src.mlflow_setup.experiments import setup_experiment
from src.mlflow_setup.run_tracker import start_run, log_params, log_metrics, end_run
import random


def run_dummy_experiment():
    setup_experiment("mro_experiment")

    start_run("dummy_run_v1")

    params = {"alpha": 0.1, "model": "baseline"}
    metrics = {"accuracy": random.uniform(0.7, 0.9)}

    log_params(params)
    log_metrics(metrics)

    end_run()


if __name__ == "__main__":
    run_dummy_experiment()
