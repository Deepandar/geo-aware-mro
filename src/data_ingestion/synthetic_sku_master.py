# src/data_ingestion/synthetic_sku_master.py

import pandas as pd
import numpy as np


def generate_sku_master(n=100, seed=42):
    np.random.seed(seed)

    df = pd.DataFrame({
        "unit_cost": np.random.uniform(10, 500, n),
        "demand": np.random.randint(1, 200, n),
        "stockout_cost_usd": np.random.uniform(100, 20000, n),
        "depot_tier": np.random.choice(["Forward", "Border", "Rear"], n),
        "lead_time_days": np.random.randint(5, 180, n),
    })

    return df


def save(path="data/processed/sku_master.csv"):
    df = generate_sku_master()
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    save()
