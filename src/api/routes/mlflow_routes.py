from fastapi import APIRouter

from src.mlflow_service.model_registry import (
    get_tracking_uri,
    list_experiments,
    list_registered_models,
)

router = APIRouter(
    prefix="/mlflow",
    tags=["MLflow"]
)

@router.get("/tracking-uri")
def tracking_uri():

    return {
        "tracking_uri": get_tracking_uri()
    }


@router.get("/experiments")
def experiments():

    return {
        "experiments": list_experiments()
    }


@router.get("/models")
def registered_models():

    return {
        "models": list_registered_models()
    }
