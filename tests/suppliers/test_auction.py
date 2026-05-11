import numpy as np
import pandas as pd

from src.suppliers.auction_module import (
    ProcurementAuctionEngine,
    AuctionResult,
)


def test_sealed_bid():

    engine = ProcurementAuctionEngine(seed=42)

    row = {
        "item_id": "SKU001",
        "unit_cost": 10000.0,
        "ci_score": 0.8,
    }

    r = engine._sim_sealed(row, 5)

    assert isinstance(r, AuctionResult)

    assert r.auction_type == "sealed_bid"


def test_english():

    engine = ProcurementAuctionEngine(seed=42)

    row = {
        "item_id": "SKU001",
        "unit_cost": 10000.0,
        "ci_score": 0.9,
    }

    r = engine._sim_english(row, 5)

    assert isinstance(r, AuctionResult)

    assert r.auction_type == "english"


def test_portfolio():

    np.random.seed(42)

    df = pd.DataFrame(
        {
            "item_id": [f"SKU{i:03d}" for i in range(20)],
            "ci_tier": np.random.choice(
                ["High", "Medium", "Low"],
                20,
            ),
            "ci_score": np.random.uniform(
                0.3,
                0.95,
                20,
            ),
            "ved_class": np.random.choice(
                ["V", "E", "D"],
                20,
            ),
            "unit_cost": np.random.uniform(
                500,
                20000,
                20,
            ),
            "geo_risk_score": np.random.uniform(
                0,
                0.8,
                20,
            ),
        }
    )

    engine = ProcurementAuctionEngine(seed=42)

    p = engine.run_portfolio_auctions(df)

    assert p.n_auctions >= 0
