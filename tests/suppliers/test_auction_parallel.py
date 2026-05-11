import numpy as np
import pandas as pd

from src.suppliers.auction_module import (
    ProcurementAuctionEngine,
)


def test_parallel_portfolio():

    np.random.seed(42)

    df = pd.DataFrame({

        "item_id": [
            f"SKU{i:03d}"
            for i in range(50)
        ],

        "ci_tier": np.random.choice(
            ["High", "Medium", "Low"],
            50,
        ),

        "ci_score": np.random.uniform(
            0.3,
            0.95,
            50,
        ),

        "ved_class": np.random.choice(
            ["V", "E", "D"],
            50,
        ),

        "unit_cost": np.random.uniform(
            500,
            20000,
            50,
        ),

        "geo_risk_score": np.random.uniform(
            0,
            0.8,
            50,
        ),

    })

    engine = ProcurementAuctionEngine()

    p = engine.run_portfolio_auctions(df)

    assert p.n_auctions >= 0

    assert p.total_baseline_cost >= 0

    assert p.total_auction_cost >= 0
