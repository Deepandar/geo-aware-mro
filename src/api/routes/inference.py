from fastapi import (
    APIRouter,
    Depends,
)

from src.api.schemas.inference import (
    RiskRequest,
    RiskResponse,
)

from src.security.auth import (
    verify_api_key
)

router = APIRouter(
    prefix="/inference",
    tags=["Inference"]
)

@router.post(
    "/risk-score",
    response_model=RiskResponse,
)
def score_risk(
    payload: RiskRequest,
    authorized: bool = Depends(
        verify_api_key
    ),
):

    risk_score = (
        payload.supplier_score * 0.8
    )

    return {
        "risk_score": risk_score
    }
