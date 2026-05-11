import time
import random

from src.orchestration.celery_app import (
    celery_app
)

@celery_app.task
def run_monte_carlo():

    time.sleep(5)

    return {
        "simulation": "monte_carlo",
        "fill_rate": round(
            random.uniform(0.85, 0.99),
            3,
        ),
        "status": "completed"
    }
