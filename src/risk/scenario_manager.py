# src/risk/scenario_manager.py

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------
# Geopolitical correlation structure
# -----------------------------------------------------------------

SCENARIO_CORRELATIONS = {
    ("RU", "CN"): 0.65,
    ("RU", "IR"): 0.55,
    ("CN", "TW"): 0.70,
}


class ScenarioManager:
    """
    Stochastic Black Swan Scenario Engine (v1.3)

    Features:
        - risk-conditioned activation
        - stochastic lead-time perturbation
        - geo-risk amplification
        - temporal compatibility
        - correlated disruption propagation
        - semantically constrained shocks
    """

    def __init__(
        self,
        scenario_path: str | Path = ("config/scenario_library.yaml"),
        random_seed: int = 42,
        geo_lambda: float = 0.8,
        hhi_lambda: float = 0.4,
    ):

        with open(scenario_path) as f:
            cfg = yaml.safe_load(f)

        self.scenarios = cfg["scenario_library"]

        self.geo_lambda = geo_lambda
        self.hhi_lambda = hhi_lambda

        self.rng = np.random.default_rng(random_seed)

        logger.info(
            ("ScenarioManager initialised | " "scenarios=%d"),
            len(self.scenarios),
        )

    # -------------------------------------------------------------
    # Main injection engine
    # -------------------------------------------------------------

    def inject(
        self,
        df: pd.DataFrame,
        sim_time: int | None = None,
    ) -> pd.DataFrame:

        self._validate(df)

        out = df.copy()

        out["active_scenario"] = "NONE"

        out["scenario_lt_multiplier"] = 1.0

        out["scenario_active"] = False

        # ---------------------------------------------------------
        # Iterate scenarios
        # ---------------------------------------------------------

        for (
            scenario_name,
            scenario_cfg,
        ) in self.scenarios.items():

            base_prob = scenario_cfg["activation_probability"]

            affected = scenario_cfg["affected_origins"]

            # -----------------------------------------------------
            # Build affected mask
            # -----------------------------------------------------

            if "ALL" in affected:

                mask = np.ones(
                    len(out),
                    dtype=bool,
                )

            else:

                mask = out["supply_origin_country"].isin(affected)

            if not mask.any():
                continue

            # -----------------------------------------------------
            # Risk-conditioned activation probability
            # -----------------------------------------------------

            geo_risk = out.loc[mask, "geo_risk_score"]

            hhi = out.loc[mask, "hhi_score"]

            activation_prob = base_prob * (
                1.0 + self.geo_lambda * geo_risk + self.hhi_lambda * hhi
            )

            activation_prob = np.clip(
                activation_prob,
                0.0,
                0.95,
            )

            # -----------------------------------------------------
            # Bernoulli activation
            # -----------------------------------------------------

            activated = self.rng.random(size=len(activation_prob)) < activation_prob

            if not activated.any():
                continue

            # -----------------------------------------------------
            # Temporal compatibility
            # -----------------------------------------------------

            # Backward-compatible duration handling
            duration = scenario_cfg.get(
                "duration_days",
                scenario_cfg.get(
                    "duration_periods",
                    30,
                ),
            )

            if sim_time is not None:

                active_window = sim_time <= duration

                if not active_window:
                    continue

            # -----------------------------------------------------
            # Distribution sampling
            # -----------------------------------------------------

            dist_cfg = scenario_cfg["lead_time_shift"]

            # -------------------------------------------------
            # Backward-compatible LT multiplier handling
            #
            # Legacy schema:
            #   lead_time_shift:
            #       mu_multiplier:
            #       sigma_multiplier:
            #
            # Week19 schema:
            #   lead_time_shift:
            #       HIGH_RISK:
            #           mu_mult:
            #           sigma_mult:
            # -------------------------------------------------

            if "mu_multiplier" in dist_cfg:

                mu_mult = dist_cfg.get(
                    "mu_multiplier",
                    1.0,
                )

                sigma_mult = dist_cfg.get(
                    "sigma_multiplier",
                    1.0,
                )

            else:

                country_cluster = "UNKNOWN"

                try:

                    from src.simulation.lead_time_fitter import (
                        COUNTRY_CLUSTERS,
                    )

                    country_cluster = COUNTRY_CLUSTERS.get(
                        str(out.loc[mask, "supply_origin_country"].iloc[0]).upper(),
                        "UNKNOWN",
                    )

                except Exception:
                    pass

                cluster_cfg = dist_cfg.get(
                    country_cluster,
                    {},
                )

                mu_mult = cluster_cfg.get(
                    "mu_mult",
                    1.0,
                )

                sigma_mult = cluster_cfg.get(
                    "sigma_mult",
                    1.0,
                )

            # ---------------------------------------------
            # Backward-compatible distribution handling
            #
            # Legacy schema:
            #   distribution: gamma/lognormal/etc
            #
            # Week19 schema:
            #   distribution omitted
            #   (Gamma handled centrally by Week18 engine)
            # ---------------------------------------------

            distribution = dist_cfg.get(
                "distribution",
                "gamma",
            )

            affected_idx = out.loc[mask].index[activated]

            # -----------------------------------------------------
            # Gamma distribution
            # -----------------------------------------------------

            if distribution == "gamma":

                shape = max(
                    (
                        mu_mult
                        / max(
                            sigma_mult,
                            0.1,
                        )
                    )
                    ** 2,
                    1.0,
                )

                scale = max(
                    sigma_mult**2 / mu_mult,
                    0.1,
                )

                sampled_mult = self.rng.gamma(
                    shape=shape,
                    scale=scale,
                    size=len(affected_idx),
                )

            # -----------------------------------------------------
            # Lognormal distribution
            # -----------------------------------------------------

            elif distribution == "lognormal":

                sampled_mult = self.rng.lognormal(
                    mean=np.log(
                        max(
                            mu_mult,
                            1.01,
                        )
                    ),
                    sigma=max(
                        sigma_mult / 5,
                        0.1,
                    ),
                    size=len(affected_idx),
                )

            # -----------------------------------------------------
            # Deterministic fallback
            # -----------------------------------------------------

            else:

                sampled_mult = np.full(
                    len(affected_idx),
                    mu_mult,
                )

            # -----------------------------------------------------
            # IMPORTANT:
            # Disruptions must NEVER reduce lead time
            # -----------------------------------------------------

            sampled_mult = np.clip(
                sampled_mult,
                1.0,
                None,
            )

            # -----------------------------------------------------
            # Preserve original shocked state
            # -----------------------------------------------------

            out.loc[affected_idx, "initial_disrupted_lt"] = (
                out.loc[affected_idx, "lead_time_days"] * sampled_mult
            )

            # -----------------------------------------------------
            # Apply disruption
            # -----------------------------------------------------

            geo_delta = scenario_cfg["geo_risk_delta"]

            out.loc[affected_idx, "lead_time_days"] *= sampled_mult

            out.loc[affected_idx, "geo_risk_score"] += (
                geo_delta * out.loc[affected_idx, "geo_risk_score"]
            )

            out.loc[affected_idx, "active_scenario"] = scenario_name

            out.loc[affected_idx, "scenario_lt_multiplier"] = sampled_mult

            out.loc[affected_idx, "scenario_active"] = True

            # -----------------------------------------------------
            # Correlated propagation
            # -----------------------------------------------------

            activated_origins = set(out.loc[affected_idx, "supply_origin_country"])

            for (
                src,
                dst,
            ), corr_strength in SCENARIO_CORRELATIONS.items():

                if src not in activated_origins:
                    continue

                corr_mask = out["supply_origin_country"] == dst

                if not corr_mask.any():
                    continue

                correlated_activation = (
                    self.rng.random(size=corr_mask.sum()) < corr_strength
                )

                corr_idx = out.loc[corr_mask].index[correlated_activation]

                if len(corr_idx) == 0:
                    continue

                correlated_mult = sampled_mult.mean() * corr_strength

                correlated_mult = max(
                    correlated_mult,
                    1.0,
                )

                out.loc[corr_idx, "lead_time_days"] *= correlated_mult

                out.loc[corr_idx, "geo_risk_score"] += geo_delta * corr_strength

                out.loc[corr_idx, "active_scenario"] = f"{scenario_name}_spillover"

                out.loc[corr_idx, "scenario_lt_multiplier"] = correlated_mult

                out.loc[corr_idx, "scenario_active"] = True

            logger.warning(
                ("Scenario activated | " "%s | affected=%d"),
                scenario_name,
                len(affected_idx),
            )

        # ---------------------------------------------------------
        # Final cleanup
        # ---------------------------------------------------------

        out["geo_risk_score"] = np.clip(
            out["geo_risk_score"],
            0.0,
            1.0,
        )

        logger.info(
            ("Scenario injection complete | " "active=%d"),
            int(out["scenario_active"].sum()),
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
            "supply_origin_country",
            "lead_time_days",
            "geo_risk_score",
            "hhi_score",
        }

        missing = required - set(df.columns)

        if missing:

            raise ValueError(("ScenarioManager missing " f"columns: {missing}"))

        if df.empty:

            raise ValueError(("ScenarioManager received " "empty dataframe"))

        if (df["lead_time_days"] < 0).any():

            raise ValueError(("Negative lead times " "detected"))

        if df["geo_risk_score"].between(0, 1).all() is False:

            raise ValueError(("Geo-risk scores " "outside [0,1]"))
