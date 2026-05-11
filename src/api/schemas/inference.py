from pydantic import BaseModel


class RiskRequest(BaseModel):
    country: str
    supplier_score: float


class RiskResponse(BaseModel):
    risk_score: float
