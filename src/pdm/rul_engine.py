from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RULEngine:
    """
    CMAPSS-inspired Remaining Useful Life engine.

    Features:
        - degradation trajectories
        - impending failure prediction
        - PdM urgency propagation
        - hybrid push/pull replenishment

    This is NOT full CMAPSS yet.
    It creates Bellman-ready condition state.
    """

    def __init__(
        self,
        random_seed: int = 42,
        critical_rul_days: int = 30,
    ):

        self.rng = np.random.default_rng(random_seed)

        self.critical_rul_days = critical_rul_days

        logger.info(
            ("RULEngine initialised | " "critical_rul=%d"),
            self.critical_rul_days,
        )

    # -------------------------------------------------------------
    # Generate synthetic degradation state
    # -------------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out = df.copy()

        # ---------------------------------------------------------
        # Generate synthetic RUL distribution
        # ---------------------------------------------------------

        rul_days = self.rng.gamma(
            shape=5,
            scale=25,
            size=len(out),
        )

        rul_days = np.clip(
            rul_days,
            1,
            365,
        )

        out["rul_days"] = rul_days

        # ---------------------------------------------------------
        # Failure urgency
        # ---------------------------------------------------------

        out["failure_urgency"] = 1.0 - (out["rul_days"] / 365.0)

        out["failure_urgency"] = np.clip(
            out["failure_urgency"],
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # Critical failure trigger
        # ---------------------------------------------------------

        out["imminent_failure"] = out["rul_days"] <= self.critical_rul_days

        # ---------------------------------------------------------
        # Hybrid replenishment trigger
        # ---------------------------------------------------------

        if "rop" in out.columns:

            inventory_trigger = out["demand"] >= out["rop"]

        else:

            inventory_trigger = np.zeros(
                len(out),
                dtype=bool,
            )

        out["hybrid_replenishment_trigger"] = (
            inventory_trigger | out["imminent_failure"]
        )

        # ---------------------------------------------------------
        # Pull-forward logic
        # ---------------------------------------------------------

        out["rul_risk_multiplier"] = 1.0 + (out["failure_urgency"] * 0.75)

        # ---------------------------------------------------------
        # Escalate CI if impending failure
        # ---------------------------------------------------------

        if "ci_score" in out.columns:

            imminent_mask = out["imminent_failure"]

            out.loc[
                imminent_mask,
                "ci_score",
            ] = np.maximum(
                out.loc[
                    imminent_mask,
                    "ci_score",
                ],
                0.90,
            )

        logger.info(
            (
                "RUL scoring complete | "
                "mean_rul=%.2f | "
                "imminent=%d | "
                "hybrid_triggers=%d"
            ),
            float(out["rul_days"].mean()),
            int(out["imminent_failure"].sum()),
            int(out["hybrid_replenishment_trigger"].sum()),
        )

        return out
