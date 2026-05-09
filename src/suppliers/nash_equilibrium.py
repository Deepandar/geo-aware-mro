from __future__ import annotations

import numpy as np
import pandas as pd


class NashEquilibriumEngine:

    """
    Strategic sourcing equilibrium engine.

    Simulates:
    - supplier competition
    - diversification
    - pricing pressure
    - resilience allocation
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):

        self.alpha = alpha
        self.beta = beta

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out = df.copy()

        resilience = (
            1.0
            - out[
                "supplier_risk_score"
            ]
        )

        criticality = (
            out.get(
                "ci_score",
                0.5,
            )
        )

        out[
            "strategic_risk_score"
        ] = (
            self.alpha
            * (
                1.0 - resilience
            )
            +
            self.beta
            * criticality
        ).clip(0,1)

        out[
            "supplier_strategy"
        ] = np.where(
            out[
                "strategic_risk_score"
            ] > 0.75,
            "Diversify",
            np.where(
                out[
                    "strategic_risk_score"
                ] > 0.50,
                "Dual Source",
                "Maintain",
            )
        )

        return out
