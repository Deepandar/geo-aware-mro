# src/classifiers/fns_classifier.py

import pandas as pd


def classify_fns(df: pd.DataFrame) -> pd.DataFrame:
    if "demand" not in df.columns:
        raise ValueError("Missing required column: demand")

    df = df.copy()

    df["demand_rank"] = df["demand"].rank(pct=True)

    def assign(p):
        if p >= 0.66:
            return "F"
        elif p >= 0.33:
            return "N"
        return "S"

    df["fns_class"] = df["demand_rank"].apply(assign)

    return df
