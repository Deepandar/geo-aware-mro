"""
Scenario Injector — Black Swan Runtime Disruption
=================================================
Injects stochastic Black Swan disruptions into SKU-level
simulation inputs for Monte Carlo + DES stress testing.

Author : Deepender
Version: 1.2.0
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from src.simulation.lead_time_fitter import (
    COUNTRY_CLUSTERS,
)

logger = logging.getLogger(__name__)


# =========================================================
# IMPACT DATACLASS
# =========================================================

@dataclass
class ScenarioImpact:

    scenario_name: str

    lt_mu_overrides: dict
    lt_sigma_overrides: dict

    modified_df: pd.DataFrame

    n_skus_affected: int
    pct_skus_affected: float

    mean_lt_multiplier: float

    geo_risk_delta: float

    stockout_cost_multiplier: float


# =========================================================
# SCENARIO INJECTOR
# =========================================================

class ScenarioInjector:

    def __init__(
        self,
        scenario_library_path: str,
        rng_seed: int = 42,
    ):

        self.path = Path(scenario_library_path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Scenario library not found: {self.path}"
            )

        with open(self.path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.library = raw.get("scenario_library", {})

        if "baseline" in raw:
            self.library["baseline"] = raw["baseline"]

        self.rng = np.random.default_rng(rng_seed)

        logger.info(
            "Loaded %d scenarios from %s",
            len(self.library),
            self.path,
        )

    # =====================================================
    # PUBLIC API
    # =====================================================

    def inject(
        self,
        sku_df: pd.DataFrame,
        scenario: str,
    ) -> ScenarioImpact:

        if scenario not in self.library:
            raise ValueError(
                f"Unknown scenario: {scenario}"
            )

        scenario_cfg = self.library[scenario]

        modified_df = sku_df.copy()

        mu_overrides = {}
        sigma_overrides = {}

        affected_count = 0
        lt_multipliers = []

        affected_origins = scenario_cfg.get(
            "affected_origins",
            [],
        )

        affected_clusters = scenario_cfg.get(
            "affected_clusters",
            [],
        )

        affected_pct = float(
            scenario_cfg.get(
                "affected_skus_pct",
                0.0,
            )
        )

        geo_delta = float(
            scenario_cfg.get(
                "geo_risk_delta",
                0.0,
            )
        )

        stockout_mult = float(
            scenario_cfg.get(
                "stockout_cost_multiplier",
                1.0,
            )
        )

        lead_time_shift = scenario_cfg.get(
            "lead_time_shift",
            {},
        )

        # =================================================
        # PROCESS EACH SKU
        # =================================================

        for idx, row in modified_df.iterrows():

            item_id = str(
                row.get(
                    "item_id",
                    f"SKU_{idx}",
                )
            )

            origin = str(
                row.get(
                    "supply_origin_country",
                    "UNKNOWN",
                )
            )

            cluster = COUNTRY_CLUSTERS.get(
                origin,
                "UNKNOWN",
            )

            base_mu = float(
                row.get(
                    "mean_lead_time",
                    30.0,
                )
            )

            base_sigma = float(
                row.get(
                    "std_lead_time",
                    max(base_mu * 0.20, 1.0),
                )
            )

            # =============================================
            # DETERMINE ELIGIBILITY
            # =============================================

            origin_match = (
                "ALL" in affected_origins
                or origin in affected_origins
            )

            cluster_match = (
                "ALL" in affected_clusters
                or cluster in affected_clusters
            )

            stochastic_trigger = (
                self.rng.random() < affected_pct
            )

            affected = (
                scenario == "baseline"
                or (
                    origin_match
                    and cluster_match
                    and stochastic_trigger
                )
            )

            # =============================================
            # BASELINE
            # =============================================

            if scenario == "baseline":

                new_mu = base_mu
                new_sigma = base_sigma
                multiplier = 1.0

            # =============================================
            # DISRUPTION
            # =============================================

            elif affected:

                cluster_shift = lead_time_shift.get(
                    cluster,
                    {
                        "mu_mult": 1.0,
                        "sigma_mult": 1.0,
                    },
                )

                mu_mult = float(
                    cluster_shift.get(
                        "mu_mult",
                        1.0,
                    )
                )

                sigma_mult = float(
                    cluster_shift.get(
                        "sigma_mult",
                        1.0,
                    )
                )

                noise = self.rng.normal(
                    loc=1.0,
                    scale=0.10,
                )

                noise = max(noise, 0.50)

                multiplier = mu_mult * noise

                new_mu = max(
                    1.0,
                    base_mu * multiplier,
                )

                new_sigma = max(
                    1.0,
                    base_sigma * sigma_mult,
                )

                affected_count += 1

                lt_multipliers.append(multiplier)

            else:

                new_mu = base_mu
                new_sigma = base_sigma
                multiplier = 1.0

            # =============================================
            # APPLY OVERRIDES
            # =============================================

            mu_overrides[item_id] = new_mu
            sigma_overrides[item_id] = new_sigma

            modified_df.loc[
                idx,
                "mean_lead_time",
            ] = new_mu

            modified_df.loc[
                idx,
                "std_lead_time",
            ] = new_sigma

            modified_df.loc[
                idx,
                "geo_risk_score",
            ] = min(
                1.0,
                float(
                    row.get(
                        "geo_risk_score",
                        0.0,
                    )
                )
                + geo_delta,
            )

        # =================================================
        # SUMMARY METRICS
        # =================================================

        n_total = len(modified_df)

        pct_affected = (
            affected_count / n_total
            if n_total > 0
            else 0.0
        )

        mean_mult = (
            float(np.mean(lt_multipliers))
            if len(lt_multipliers) > 0
            else 1.0
        )

        logger.info(
            "Scenario=%s | affected=%d/%d | mean_LT_mult=%.2f",
            scenario,
            affected_count,
            n_total,
            mean_mult,
        )

        return ScenarioImpact(

            scenario_name=scenario,

            lt_mu_overrides=mu_overrides,
            lt_sigma_overrides=sigma_overrides,

            modified_df=modified_df,

            n_skus_affected=affected_count,
            pct_skus_affected=pct_affected,

            mean_lt_multiplier=mean_mult,

            geo_risk_delta=geo_delta,

            stockout_cost_multiplier=stockout_mult,
        )

    # =====================================================
    # RUN ALL SCENARIOS
    # =====================================================

    def inject_all(
        self,
        sku_df: pd.DataFrame,
    ) -> list[ScenarioImpact]:

        impacts = []

        for scenario in self.library.keys():

            impacts.append(
                self.inject(
                    sku_df,
                    scenario,
                )
            )

        return impacts

    # =====================================================
    # IMPACT SUMMARY
    # =====================================================

    @staticmethod
    def impact_summary(
        impacts: list[ScenarioImpact],
    ) -> pd.DataFrame:

        rows = []

        for impact in impacts:

            rows.append({

                "scenario":
                    impact.scenario_name,

                "n_skus_affected":
                    impact.n_skus_affected,

                "pct_skus_affected":
                    impact.pct_skus_affected,

                "mean_lt_multiplier":
                    impact.mean_lt_multiplier,

                "geo_risk_delta":
                    impact.geo_risk_delta,

                "stockout_cost_multiplier":
                    impact.stockout_cost_multiplier,
            })

        return pd.DataFrame(rows)


# =========================================================
# SMOKE TEST
# =========================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    df = pd.DataFrame({
        "item_id": ["A", "B", "C"],
        "supply_origin_country": ["RU", "IN", "CN"],
        "mean_lead_time": [60, 20, 45],
        "std_lead_time": [15, 5, 10],
        "geo_risk_score": [0.8, 0.2, 0.5],
    })

    injector = ScenarioInjector(
        "config/scenario_library.yaml"
    )

    impacts = injector.inject_all(df)

    print(
        injector.impact_summary(impacts)
    )
