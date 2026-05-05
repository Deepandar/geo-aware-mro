# src/classifiers/abc_classifier.py

import pandas as pd


def classify_abc(df: pd.DataFrame) -> pd.DataFrame:
    if "unit_cost" not in df.columns or "demand" not in df.columns:
        raise ValueError("Missing required columns: unit_cost, demand")

    df = df.copy()

    if "annual_consumption_value" not in df.columns:
        df["annual_consumption_value"] = df["unit_cost"] * df["demand"]

    df = df.sort_values(by="annual_consumption_value", ascending=False)

    total_acv = df["annual_consumption_value"].sum()
    if total_acv == 0:
        raise ValueError("Total ACV is zero")

    df["cum_acv"] = df["annual_consumption_value"].cumsum()
    df["cum_pct"] = df["cum_acv"] / total_acv

    def assign_class(p):
        if p <= 0.8:
            return "A"
        elif p <= 0.95:
            return "B"
        return "C"

    df["abc_class"] = df["cum_pct"].apply(assign_class)

    return df.reset_index(drop=True)
