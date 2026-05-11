from src.orchestration.tasks import (
    run_monte_carlo
)

task = run_monte_carlo.delay()

print("TASK ID:", task.id)
print("TASK STATE:", task.status)
