from celery import chain

from src.orchestration.tasks import (
    run_monte_carlo
)

from src.orchestration.rl_tasks import (
    train_rl_agent
)

def execute_supply_chain_pipeline():

    workflow = chain(
        run_monte_carlo.s(),
        train_rl_agent.s(),
    )

    return workflow.delay()
