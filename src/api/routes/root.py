from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def root():
    return {"project": "geo-aware-mro", "phase": "Phase B", "status": "operational"}
