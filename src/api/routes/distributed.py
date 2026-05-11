from fastapi import APIRouter

from src.distributed.ray_runner import (
    ray_status
)

router = APIRouter(
    prefix="/distributed",
    tags=["Distributed"]
)

@router.get("/status")
def distributed_status():

    return ray_status()
