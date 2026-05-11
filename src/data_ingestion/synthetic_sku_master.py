"""
Synthetic SKU master generator.

Provides deterministic synthetic inventory data
for classifier and pipeline experimentation.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42


def generate_sku_master(n_skus: int = 500) -> pd.DataFrame:
    """
    Generate synthetic SKU master dataset.
    """

    rng = np.random.default_rng(RNG_SEED)

    depot_choices = ["Forward", "Border", "Rear"]

    country_choices = [
        "CN",
        "US",
        "DE",
        "IN",
        "RU",
        "UA",
        "IR",
        "AE",
    ]

    df = pd.DataFrame(
        {
            "item_id": range(1, n_skus + 1),
            "unit_cost": rng.uniform(10, 500, n_skus),
            "demand": rng.integers(1, 250, n_skus),
            "equipment_density_score": rng.uniform(0, 1, n_skus),
            "adi": rng.uniform(0.5, 2.0, n_skus),
            "cv_squared": rng.uniform(0.1, 2.5, n_skus),
            "depot_tier": rng.choice(depot_choices, n_skus),
            "environment_multiplier": rng.uniform(0.8, 1.4, n_skus),
            # ---------------------------------------------------------
            # IMPORTANT:
            # Convert to float immediately for stochastic simulation
            # compatibility (DES / Monte Carlo / resilience decay)
            # ---------------------------------------------------------
            "lead_time_days": rng.integers(5, 180, n_skus).astype(float),
            "supply_origin_country": rng.choice(country_choices, n_skus),
            # ---------------------------------------------------------
            # Placeholder prior geo-risk
            # Later overwritten by BayesianRiskScorer
            # ---------------------------------------------------------
            "geo_risk_score": rng.uniform(0, 1, n_skus),
            # ---------------------------------------------------------
            # Supplier concentration proxy
            # Required for compound LTR + scenario activation
            # ---------------------------------------------------------
            "hhi_score": rng.uniform(0.1, 1.0, n_skus),
            "mean_demand": rng.uniform(5, 200, n_skus),
            "std_demand": rng.uniform(1, 50, n_skus),
            "mean_lead_time": rng.uniform(5, 180, n_skus),
            "std_lead_time": rng.uniform(1, 30, n_skus),
            "stockout_cost_usd": rng.uniform(100, 50000, n_skus),
        }
    )

    # -------------------------------------------------------------
    # Additional validation for simulation stability
    # -------------------------------------------------------------

    assert df["lead_time_days"].dtype == float

    assert df["hhi_score"].between(0, 1).all()

    assert df["geo_risk_score"].between(0, 1).all()

    return df


if __name__ == "__main__":

    output_path = Path("data/raw/sku_master_raw.parquet")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = generate_sku_master(n_skus=500)

    df.to_parquet(
        output_path,
        index=False,
    )

    print(df.head())

    print(("\nSaved synthetic SKU master " f"→ {output_path}"))
