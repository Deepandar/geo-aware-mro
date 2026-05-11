"""
Dominance bias detection and remediation.

Protects ABC classification from heavy-tail SKU concentration.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DominanceChecker:
    """
    Detect and remediate concentration dominance.
    """

    top_pct: float = 0.01
    dominance_threshold: float = 0.50

    def check_and_remediate(
        self,
        df: pd.DataFrame,
        acv_col: str = "annual_consumption_value",
    ):
        """
        Detect concentration dominance and optionally remediate.

        Returns:
            (df, result_dict)
        """

        df = df.copy()

        if acv_col not in df.columns:
            raise ValueError(f"Missing column: {acv_col}")

        total = df[acv_col].sum()

        sorted_df = df.sort_values(
            by=acv_col,
            ascending=False,
        )

        top_n = max(1, int(len(df) * self.top_pct))

        concentration_ratio = sorted_df.head(top_n)[acv_col].sum() / total

        bias_detected = concentration_ratio >= self.dominance_threshold

        if bias_detected:
            print(f"⚠ DOMINANCE DETECTED: " f"{concentration_ratio:.2%}")

            df["acv_for_abc"] = np.log1p(df[acv_col])

        else:
            df["acv_for_abc"] = df[acv_col]

        result = {
            "bias_detected": bool(bias_detected),
            "concentration_ratio": float(concentration_ratio),
            "top_n": int(top_n),
        }

        return df, result
