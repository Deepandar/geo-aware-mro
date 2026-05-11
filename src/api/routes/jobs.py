from fastapi import APIRouter

from src.orchestration.tasks import run_monte_carlo

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.post("/simulation")
def submit_simulation():

    task = run_monte_carlo.delay()

    return {"task_id": task.id, "status": "submitted"}
