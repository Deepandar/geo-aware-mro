# src/classifiers/ltr_scorer.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LTRScorer:
    """
    Compound Lead-Time Risk Scorer (v1.5)

    Features:
        - non-zero systemic risk floor
        - stochastic resilience amplification
        - robust percentile normalization
        - Black Swan stability
        - NaN elimination
        - backward compatibility
    """

    def __init__(
        self,
        geo_weight: float = 0.35,
        hhi_weight: float = 0.20,
        location_weight: float = 0.20,
        leadtime_weight: float = 0.25,
    ):

        self.geo_weight = geo_weight

        self.hhi_weight = hhi_weight

        self.location_weight = location_weight

        self.leadtime_weight = leadtime_weight

        logger.info(
            ("LTRScorer initialised | " "geo=%.2f | " "hhi=%.2f"),
            self.geo_weight,
            self.hhi_weight,
        )

    # ---------------------------------------------------------
    # Main compute
    # ---------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        self._validate(df)

        df = df.copy()

        # -----------------------------------------------------
        # Backward compatibility defaults
        # -----------------------------------------------------

        if "geo_risk_score" not in df.columns:

            df["geo_risk_score"] = 0.0

        if "hhi_score" not in df.columns:

            df["hhi_score"] = 0.0

        if "location_score" not in df.columns:

            df["location_score"] = 0.5

        if "resilience_multiplier" not in df.columns:

            df["resilience_multiplier"] = 1.0

        # -----------------------------------------------------
        # Safe lead-time handling
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Input sanitation BEFORE normalization
        # -----------------------------------------------------

        lead_time = (
            df["lead_time_days"].fillna(df["lead_time_days"].median()).astype(float)
        )

        lead_time = np.nan_to_num(
            lead_time,
            nan=30.0,
            posinf=365.0,
            neginf=1.0,
        )

        lead_time = np.clip(
            lead_time,
            1.0,
            365.0,
        )
        # -----------------------------------------------------
        # Structural components
        # -----------------------------------------------------

        geo_component = df["geo_risk_score"].astype(float).fillna(0.0)

        hhi_component = df["hhi_score"].astype(float).fillna(0.0)

        location_component = df["location_score"].astype(float).fillna(0.5)

        # -----------------------------------------------------
        # Robust lead-time normalization
        # -----------------------------------------------------

        # -----------------------------------------------------
        # Robust percentile estimation
        # -----------------------------------------------------

        clean_lt = lead_time[np.isfinite(lead_time)]

        p5 = float(
            np.percentile(
                clean_lt,
                5,
            )
        )

        p95 = float(
            np.percentile(
                clean_lt,
                95,
            )
        )

        spread = p95 - p5

        denom = spread if spread > 1e-6 else 1.0

        lead_component = (
            np.clip(
                lead_time,
                p5,
                p95,
            )
            - p5
        ) / denom

        # -----------------------------------------------------
        # Base structural LTR
        # -----------------------------------------------------

        base_ltr = (
            self.geo_weight * geo_component
            + self.hhi_weight * hhi_component
            + self.location_weight * location_component
            + self.leadtime_weight * lead_component
        )

        base_ltr = np.clip(
            base_ltr,
            0.01,
            1.0,
        )

        # -----------------------------------------------------
        # Dynamic disruption amplification
        # -----------------------------------------------------

        resilience = df["resilience_multiplier"].astype(float).fillna(1.0)

        dynamic_multiplier = 1.0 + (resilience - 1.0) * 0.35

        # -----------------------------------------------------
        # Final compound LTR
        # -----------------------------------------------------

        df["ltr_score"] = base_ltr * dynamic_multiplier

        # -----------------------------------------------------
        # Final sanitation
        # -----------------------------------------------------

        df["ltr_score"] = np.nan_to_num(
            df["ltr_score"],
            nan=0.05,
            posinf=1.0,
            neginf=0.05,
        )

        df["ltr_score"] = np.clip(
            df["ltr_score"],
            0.01,
            1.0,
        )

        # -----------------------------------------------------
        # Risk bands
        # -----------------------------------------------------

        df["ltr_risk_band"] = pd.cut(
            df["ltr_score"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=[
                "LOW",
                "MEDIUM",
                "HIGH",
            ],
            include_lowest=True,
        )

        logger.info(
            ("LTR scoring complete | " "mean=%.3f | " "std=%.3f | " "nan_count=%d"),
            float(df["ltr_score"].mean()),
            float(df["ltr_score"].std()),
            int(df["ltr_score"].isna().sum()),
        )

        return df

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate(
        self,
        df: pd.DataFrame,
    ) -> None:

        required = {
            "lead_time_days",
        }

        missing = required - set(df.columns)

        if missing:

            raise ValueError(("LTRScorer missing " f"columns: {missing}"))

        if df.empty:

            raise ValueError(("LTRScorer received " "empty dataframe"))
