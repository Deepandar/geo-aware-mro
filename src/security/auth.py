from fastapi import (
    Header,
    HTTPException,
)

from src.config.settings import settings


def verify_api_key(x_api_key: str = Header(...)):

    if x_api_key != settings.API_KEY:

        raise HTTPException(status_code=401, detail="Invalid API key")

    return True
