# src/classifiers/dominance_check.py

import pandas as pd
import numpy as np


class DominanceChecker:

    def __init__(self, top_pct=0.01, threshold=0.5):
        self.top_pct = top_pct
        self.threshold = threshold

    def check_and_remediate(self, df: pd.DataFrame):
        if "annual_consumption_value" not in df.columns:
            raise ValueError("Missing column: annual_consumption_value")

        df = df.copy()

        total = float(df["annual_consumption_value"].sum())
        n_top = max(1, int(len(df) * self.top_pct))

        top = df.nlargest(n_top, "annual_consumption_value")
        ratio = float(top["annual_consumption_value"].sum() / total)

        bias = bool(ratio > self.threshold)

        if bias:
            print(f"⚠ DOMINANCE DETECTED: {ratio:.2%}")
            df["acv_for_abc"] = np.log1p(df["annual_consumption_value"])
        else:
            df["acv_for_abc"] = df["annual_consumption_value"]

        return df, {
            "bias_detected": bias,
            "concentration_ratio": ratio,
            "top_n": int(n_top)
        }
