# src/mlflow_setup/run_tracker.py

import mlflow


def start_run(run_name: str):
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict):
    mlflow.log_params(params)


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def end_run():
    mlflow.end_run()
