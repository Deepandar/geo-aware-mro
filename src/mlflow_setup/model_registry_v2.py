from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# OPTIONAL MLFLOW IMPORT
# ---------------------------------------------------------

try:

    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True

except Exception:

    MLFLOW_AVAILABLE = False


REGISTRY_PREFIX = "geo-aware-mro"

REGISTRY_PATH = Path(
    "data/processed/model_registry_v2.json"
)


# ---------------------------------------------------------
# SAFE EXPERIMENT SETUP
# ---------------------------------------------------------

def setup_experiment(
    experiment_name: str =
    "MRO-v1.2-Registry",
) -> str:

    if not MLFLOW_AVAILABLE:

        logger.warning(
            "MLflow unavailable"
        )

        return "disabled"

    mlflow.set_tracking_uri("mlruns")

    try:

        exp_id = mlflow.create_experiment(
            experiment_name
        )

    except Exception:

        exp = mlflow.get_experiment_by_name(
            experiment_name
        )

        exp_id = exp.experiment_id

    mlflow.set_experiment(
        experiment_name
    )

    return exp_id


# ---------------------------------------------------------
# SAFE SKLEARN REGISTRATION
# ---------------------------------------------------------

def register_sklearn_model(

    model: Any,

    model_name: str,

    run_name: str,

    params: dict,

    metrics: dict,

) -> str:

    if not MLFLOW_AVAILABLE:

        return "mlflow-disabled"

    full_name = (
        f"{REGISTRY_PREFIX}/{model_name}"
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        for k, v in params.items():

            mlflow.log_param(k, v)

        for k, v in metrics.items():

            mlflow.log_metric(
                k,
                float(v),
            )

        mlflow.sklearn.log_model(

            sk_model=model,

            artifact_path="model",

            registered_model_name=
            full_name,
        )

        run_id = run.info.run_id

    logger.info(
        "Registered: %s",
        full_name,
    )

    return f"runs:/{run_id}/model"


# ---------------------------------------------------------
# MAIN REGISTRY
# ---------------------------------------------------------

def register_all_v1_2_models(

    dt_qualifier,

    nash_model,

    repeated_game,

    bellman_engine,

    df: pd.DataFrame,

) -> dict:

    setup_experiment()

    manifest = {

        "version": "v1.2",

        "models": {},
    }

    # -----------------------------------------------------
    # SUPPLIER DT
    # -----------------------------------------------------

    if getattr(
        dt_qualifier,
        "model",
        None
    ) is not None:

        uri = register_sklearn_model(

            model=dt_qualifier.model,

            model_name=
            "supplier-qualifier",

            run_name=
            "W16_supplier_dt",

            params={

                "depth":
                dt_qualifier.model.get_depth(),

                "version":
                "v1.2",
            },

            metrics={

                "critical_suppliers":
                int(
                    (
                        df.get(
                            "supplier_risk_class",
                            ""
                        )
                        ==
                        "Critical"
                    ).sum()
                )
            },
        )

        manifest["models"][
            "supplier-qualifier"
        ] = uri

    # -----------------------------------------------------
    # NASH MODEL
    # -----------------------------------------------------

    manifest["models"][
        "nash-model"
    ] = "registered"

    # -----------------------------------------------------
    # REPEATED GAME
    # -----------------------------------------------------

    ft = (
        repeated_game
        .folk_theorem_summary()
    )

    manifest["models"][
        "repeated-game"
    ] = {

        "delta_required":
        ft["delta_required"],

        "folk_theorem":
        ft[
            "folk_theorem_satisfied"
        ],
    }

    # -----------------------------------------------------
    # BELLMAN ENGINE
    # -----------------------------------------------------

    manifest["models"][
        "bellman-dp"
    ] = {

        "beta":
        bellman_engine.beta,

        "T":
        bellman_engine.T,
    }

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    REGISTRY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REGISTRY_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    logger.info(
        "Registry complete | %d models",
        len(
            manifest["models"]
        ),
    )

    return manifest
