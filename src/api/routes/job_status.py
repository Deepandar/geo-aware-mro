from fastapi import APIRouter

from celery.result import AsyncResult

from src.orchestration.celery_app import celery_app

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("/{task_id}")
def get_job_status(task_id: str):

    result = AsyncResult(
        task_id,
        app=celery_app,
    )

    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result,
    }
