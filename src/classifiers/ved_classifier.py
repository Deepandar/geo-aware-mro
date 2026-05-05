# src/classifiers/ved_classifier.py

import pandas as pd


def classify_ved(df: pd.DataFrame) -> pd.DataFrame:
    if "stockout_cost_usd" not in df.columns:
        raise ValueError("Missing required column: stockout_cost_usd")

    df = df.copy()

    def assign(row):
        if row["stockout_cost_usd"] > 10000:
            return "V"
        elif row["stockout_cost_usd"] > 1000:
            return "E"
        return "D"

    df["ved_class"] = df.apply(assign, axis=1)

    return df
