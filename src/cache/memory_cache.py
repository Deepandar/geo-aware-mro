from functools import lru_cache


@lru_cache(maxsize=128)
def cached_risk_score(
    country: str,
    supplier_score: float,
):

    return supplier_score * 0.8
