from fastapi import (
    APIRouter,
    Depends,
)

import asyncio

from src.security.auth import (
    verify_api_key
)

router = APIRouter(
    prefix="/simulation",
    tags=["Simulation"]
)

@router.get("/run")
async def run_simulation(
    authorized: bool = Depends(
        verify_api_key
    ),
):

    await asyncio.sleep(1)

    return {
        "status": "completed",
        "simulation": "monte_carlo"
    }
