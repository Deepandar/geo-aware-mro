from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="geo-aware-mro",
    version="1.3"
)

class HealthResponse(BaseModel):
    status: str

@app.get("/")
def root():
    return {
        "project": "geo-aware-mro",
        "status": "stable",
        "version": "1.3"
    }

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}
