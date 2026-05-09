# src/utils/mlflow_setup.py

"""
MLflow experiment bootstrap for GEO-AWARE MRO.

Features:
    - safe local file tracking
    - experiment auto-creation
    - Windows-safe configuration
    - lightweight smoke test
    - pytest-friendly

Run:
    python -m src.utils.mlflow_setup
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow


logger = logging.getLogger(__name__)


EXPERIMENT_NAME = "geo-aware-mro"


def setup_mlflow() -> str:
    """
    Initialise MLflow tracking + experiment.

    Returns
    -------
    str
        MLflow tracking URI
    """

    # ---------------------------------------------------------
    # Local tracking directory
    # ---------------------------------------------------------

    tracking_dir = (
        Path("mlruns")
        .absolute()
    )

    tracking_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Windows-safe file URI
    # ---------------------------------------------------------

    tracking_uri = (
        tracking_dir.as_uri()
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    logger.info(
        (
            "MLflow tracking URI set | "
            "%s"
        ),
        tracking_uri,
    )

    # ---------------------------------------------------------
    # Create / load experiment
    # ---------------------------------------------------------

    experiment = (
        mlflow.get_experiment_by_name(
            EXPERIMENT_NAME
        )
    )

    if experiment is None:

        experiment_id = (
            mlflow.create_experiment(
                EXPERIMENT_NAME
            )
        )

        logger.info(
            (
                "Created MLflow experiment | "
                "name=%s | id=%s"
            ),
            EXPERIMENT_NAME,
            experiment_id,
        )

    else:

        experiment_id = (
            experiment.experiment_id
        )

        logger.info(
            (
                "Using existing experiment | "
                "name=%s | id=%s"
            ),
            EXPERIMENT_NAME,
            experiment_id,
        )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # ---------------------------------------------------------
    # Lightweight smoke test
    # ---------------------------------------------------------

    try:

        logger.info(
            "Running MLflow smoke test..."
        )

        with mlflow.start_run(
            run_name=(
                "mlflow_setup_smoke_test"
            )
        ):

            mlflow.log_param(
                "environment",
                "local",
            )

            mlflow.log_metric(
                "smoke_metric",
                1.0,
            )

        logger.info(
            (
                "MLflow smoke test "
                "completed successfully"
            )
        )

    except Exception as exc:

        logger.warning(
            (
                "MLflow smoke test failed | "
                "%s"
            ),
            str(exc),
        )

    logger.info(
        "MLflow setup complete"
    )

    return tracking_uri


if __name__ == "__main__":

    uri = setup_mlflow()

    print("\nMLflow setup complete.")
    print(f"Tracking URI: {uri}\n")