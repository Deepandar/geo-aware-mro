from fastapi import APIRouter

from src.orchestration.rl_tasks import train_rl_agent

router = APIRouter(prefix="/rl", tags=["RL"])


@router.post("/train")
def trigger_training():

    task = train_rl_agent.delay()

    return {"task_id": task.id, "status": "submitted"}
