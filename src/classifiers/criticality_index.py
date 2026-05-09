# src/classifiers/criticality_index.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


EPSILON = 1e-6


class CriticalityIndexer:

    """
    Robust Criticality Indexer (v1.5)

    Features:
        - robust scaling
        - Laplacian smoothing
        - Black Swan stability
        - existential sourcing override
        - NaN firewall
        - heavy-tail resilience
    """

    def __init__(self):

        logger.info(
            "CriticalityIndexer initialised"
        )

    # -------------------------------------------------------------
    # Main compute
    # -------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        self._validate(df)

        out = df.copy()

        # ---------------------------------------------------------
        # Encode categorical dimensions
        # ---------------------------------------------------------

        abc_map = {
            "A": 1.0,
            "B": 0.6,
            "C": 0.3,
        }

        ved_map = {
            "Vital": 1.0,
            "Essential": 0.6,
            "Desirable": 0.3,
            "V": 1.0,
            "E": 0.6,
            "D": 0.3,
        }

        fns_map = {
            "Fast": 1.0,
            "Normal": 0.7,
            "Slow": 0.4,
            "Smooth": 1.0,
            "Erratic": 0.8,
            "Intermittent": 0.5,
            "Lumpy": 0.3,
        }

        out["abc_score"] = (
            out["abc_class"]
            .map(abc_map)
            .fillna(0.5)
        )

        out["ved_score"] = (
            out["ved_class"]
            .map(ved_map)
            .fillna(0.5)
        )

        out["fns_score"] = (
            out["fns_class"]
            .map(fns_map)
            .fillna(0.5)
        )

        # ---------------------------------------------------------
        # Robust log-LTR scaling
        # ---------------------------------------------------------

        ltr = np.log1p(
            out["ltr_score"]
            .astype(float)
            .fillna(0.5)
        )

        median_ltr = float(
            ltr.median()
        )

        iqr_ltr = float(
            ltr.quantile(0.75)
            - ltr.quantile(0.25)
        )

        iqr_ltr = max(
            iqr_ltr,
            EPSILON,
        )

        robust_ltr = (
            ltr - median_ltr
        ) / iqr_ltr

        robust_ltr = np.clip(
            robust_ltr,
            -3,
            3,
        )

        robust_ltr = (
            robust_ltr + 3
        ) / 6

        robust_ltr = robust_ltr.fillna(
            0.5
        )

        # ---------------------------------------------------------
        # Geo-risk scaling
        # ---------------------------------------------------------

        geo = np.log1p(
            out["geo_risk_score"]
            .astype(float)
            .fillna(0.5)
        )

        geo = np.clip(
            geo,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # Base weighted CI
        # ---------------------------------------------------------

        out["ci_score"] = (

            0.25 * out["abc_score"]

            + 0.25 * out["ved_score"]

            + 0.15 * out["fns_score"]

            + 0.20 * robust_ltr

            + 0.15 * geo
        )

        # ---------------------------------------------------------
        # EXISTENTIAL OVERRIDE:
        # Alternative sourcing required
        # automatically becomes CRITICAL
        # ---------------------------------------------------------

        if (
            "alternative_sourcing_required"
            in out.columns
        ):

            alt_mask = (
                out[
                    "alternative_sourcing_required"
                ]
                == True
            )

            out.loc[
                alt_mask,
                "ci_score"
            ] = 1.0

        # ---------------------------------------------------------
        # NaN firewall
        # ---------------------------------------------------------

        nan_mask = (
            out["ci_score"]
            .isna()
        )

        if nan_mask.any():

            logger.warning(
                (
                    "NaN CI detected | "
                    "fallback=%d"
                ),
                int(
                    nan_mask.sum()
                ),
            )

            out.loc[
                nan_mask,
                "ci_score"
            ] = 0.95

        # ---------------------------------------------------------
        # Final clipping
        # ---------------------------------------------------------

        out["ci_score"] = np.clip(
            out["ci_score"],
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # Operational bands
        # ---------------------------------------------------------

        out["ci_band"] = pd.cut(
            out["ci_score"],
            bins=[0, 0.4, 0.7, 1.0],
            labels=[
                "LOW",
                "MEDIUM",
                "HIGH",
            ],
            include_lowest=True,
        )

        logger.info(
            (
                "Criticality scoring complete | "
                "mean=%.3f | "
                "high=%d | "
                "critical=%d"
            ),
            float(
                out["ci_score"].mean()
            ),
            int(
                (
                    out["ci_score"] > 0.7
                ).sum()
            ),
            int(
                (
                    out["ci_score"] == 1.0
                ).sum()
            ),
        )

        return out

    # -------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------

    def _validate(
        self,
        df: pd.DataFrame,
    ) -> None:

        required = {
            "abc_class",
            "ved_class",
            "fns_class",
            "ltr_score",
            "geo_risk_score",
        }

        missing = required - set(df.columns)

        if missing:

            raise ValueError(
                (
                    "Missing columns: "
                    f"{missing}"
                )
            )

        if df.empty:

            raise ValueError(
                "Empty dataframe"
            )