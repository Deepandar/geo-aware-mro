import time

from src.orchestration.celery_app import (
    celery_app
)

@celery_app.task
def train_rl_agent():

    time.sleep(10)

    return {
        "model": "PPO",
        "timesteps": 50000,
        "status": "trained"
    }
