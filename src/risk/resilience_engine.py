# src/risk/resilience_engine.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class ResilienceEngine:

    """
    Coupled resilience recovery engine (v1.4)

    Features:
        - persistent disruption states
        - exponential recovery
        - disruption fatigue coupling
        - dynamic recovery degradation
        - temporal resilience persistence

    Recovery:

        lambda_eff = lambda_0 / (1 + k*D)

        M(t)=1+(M0-1)e^(-lambda_eff*t)
    """

    def __init__(
        self,
        decay_lambda: float = 0.025,
        fatigue_beta: float = 0.02,
        recovery_threshold: float = 1.01,
    ):

        self.decay_lambda = decay_lambda

        self.fatigue_beta = fatigue_beta

        self.recovery_threshold = (
            recovery_threshold
        )

        logger.info(
            (
                "ResilienceEngine initialised | "
                "lambda=%.3f | "
                "fatigue_beta=%.3f | "
                "threshold=%.3f"
            ),
            self.decay_lambda,
            self.fatigue_beta,
            self.recovery_threshold,
        )

    # -------------------------------------------------------------
    # Apply temporal recovery
    # -------------------------------------------------------------

    def apply_decay(
        self,
        df: pd.DataFrame,
        sim_time: int,
    ) -> pd.DataFrame:

        self._validate(df)

        out = df.copy()

        active_mask = (
            out["scenario_active"] == True
        )

        # ---------------------------------------------------------
        # No active disruptions
        # ---------------------------------------------------------

        if not active_mask.any():

            logger.info(
                "No active disruptions detected"
            )

            out["resilience_multiplier"] = 1.0

            out["lambda_eff"] = (
                self.decay_lambda
            )

            return out

        # ---------------------------------------------------------
        # Compute disruption fatigue
        # ---------------------------------------------------------

        disruption_load = int(
            active_mask.sum()
        )

        lambda_eff = (
            self.decay_lambda
            / (
                1.0
                + (
                    self.fatigue_beta
                    * disruption_load
                )
            )
        )

        # ---------------------------------------------------------
        # Preserve original disrupted state
        # ---------------------------------------------------------

        if (
            "initial_disrupted_lt"
            not in out.columns
        ):

            out[
                "initial_disrupted_lt"
            ] = out["lead_time_days"]

        initial_disrupted_lt = (
            out.loc[
                active_mask,
                "initial_disrupted_lt"
            ]
            .astype(float)
        )

        initial_mult = (
            out.loc[
                active_mask,
                "scenario_lt_multiplier"
            ]
            .astype(float)
        )

        # ---------------------------------------------------------
        # Coupled exponential decay
        # ---------------------------------------------------------

        decay_factor = np.exp(
            -lambda_eff * sim_time
        )

        recovered_mult = (
            1.0
            + (
                initial_mult - 1.0
            ) * decay_factor
        )

        # ---------------------------------------------------------
        # Persist metrics
        # ---------------------------------------------------------

        out.loc[
            active_mask,
            "lambda_eff"
        ] = lambda_eff

        out.loc[
            active_mask,
            "disruption_load"
        ] = disruption_load

        out.loc[
            active_mask,
            "resilience_multiplier"
        ] = recovered_mult

        # ---------------------------------------------------------
        # IMPORTANT:
        # Recover from ORIGINAL disrupted state
        # ---------------------------------------------------------

        out.loc[
            active_mask,
            "lead_time_days"
        ] = (
            initial_disrupted_lt
            * recovered_mult
        )

        # ---------------------------------------------------------
        # Persistent geo-risk coupling
        # ---------------------------------------------------------

        if "geo_risk_score" in out.columns:

            persistent_geo = (
                out.loc[
                    active_mask,
                    "geo_risk_score"
                ]
                * (
                    0.90
                    + 0.10 * decay_factor
                )
            )

            out.loc[
                active_mask,
                "geo_risk_score"
            ] = np.clip(
                persistent_geo,
                0.0,
                1.0,
            )

        # ---------------------------------------------------------
        # Recovery completion logic
        # ---------------------------------------------------------

        recovered_mask = (
            recovered_mult
            <= self.recovery_threshold
        )

        recovered_indices = (
            out.loc[
                active_mask
            ].index[recovered_mask]
        )

        out.loc[
            recovered_indices,
            "scenario_active"
        ] = False

        out.loc[
            recovered_indices,
            "active_scenario"
        ] = "RECOVERED"

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------

        logger.info(
            (
                "Resilience decay applied | "
                "active=%d | "
                "recovered=%d | "
                "lambda_eff=%.4f | "
                "mean_multiplier=%.3f"
            ),
            int(
                out["scenario_active"].sum()
            ),
            len(recovered_indices),
            lambda_eff,
            float(
                out.loc[
                    active_mask,
                    "resilience_multiplier"
                ].mean()
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
            "lead_time_days",
            "scenario_active",
            "scenario_lt_multiplier",
        }

        missing = required - set(df.columns)

        if missing:

            raise ValueError(
                (
                    "ResilienceEngine missing "
                    f"columns: {missing}"
                )
            )

        if df.empty:

            raise ValueError(
                (
                    "ResilienceEngine received "
                    "empty dataframe"
                )
            )

        if (
            df["lead_time_days"] < 0
        ).any():

            raise ValueError(
                (
                    "Negative lead times "
                    "detected"
                )
            )

        if (
            "scenario_lt_multiplier"
            in df.columns
        ):

            if (
                df[
                    "scenario_lt_multiplier"
                ] < 1.0
            ).any():

                raise ValueError(
                    (
                        "Scenario multipliers "
                        "must be >= 1.0"
                    )
                )