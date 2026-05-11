from fastapi import APIRouter

from src.orchestration.pipeline import execute_pipeline

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/execute")
def execute():

    task = execute_pipeline()

    return {"pipeline_id": task.id, "status": "started"}
