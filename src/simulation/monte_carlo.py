"""
Monte Carlo Orchestrator — Week 20
==================================
Runs stochastic Black Swan trials across all scenarios.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulation.depot_sim import (
    GeoAwareMROEnv,
)

from src.simulation.scenario_injector import (
    ScenarioInjector,
)

logger = logging.getLogger(__name__)

SCENARIOS = [
    "baseline",
    "sanctions",
    "conflict",
    "pandemic",
    "port_closure",
    "logistics_collapse",
]


@dataclass
class MonteCarloPipeline:

    n_trials: int = 10

    sim_periods: int = 52

    scenario_config_path: str = (
        "config/scenario_library.yaml"
    )

    seed: int = 42

    fast_mode: bool = True

    def __post_init__(self):

        self.rng = np.random.default_rng(
            self.seed
        )

        self.injector = ScenarioInjector(
            self.scenario_config_path
        )

    # =====================================================
    # SINGLE TRIAL
    # =====================================================

    def _run_single_trial(
        self,
        sku_df: pd.DataFrame,
        scenario: str,
        trial_id: int,
    ) -> dict:

        # ---------------------------------------------
        # Apply scenario disruption
        # ---------------------------------------------

        impact = self.injector.inject(
            sku_df,
            scenario,
        )

        sim_df = impact.modified_df.copy()

        # ---------------------------------------------
        # Build simulator
        # ---------------------------------------------

        sim = GeoAwareMROEnv(
            n_periods=self.sim_periods,
            n_trials=1,
            seed=int(
                self.rng.integers(
                    0,
                    1_000_000,
                )
            ),
            fast_mode=self.fast_mode,
        )

        # ---------------------------------------------
        # Execute trial
        # ---------------------------------------------

        sim_results = sim.run_trial(
            sku_df=sim_df,
            trial_id=trial_id,
        )

        # ---------------------------------------------
        # Aggregate KPIs
        # ---------------------------------------------

        mean_fill_rate = float(
            sim_results.mean_fill_rate
        )

        total_stockout_cost = float(
            sim_results.total_cost
        )

        tsl_compliance_rate = float(
            sim_results.tsl_compliance_rate
        )

        # ---------------------------------------------
        # Fast-mode compatible placeholders
        # ---------------------------------------------

        fill_rate_high = mean_fill_rate
        fill_rate_medium = mean_fill_rate
        fill_rate_low = mean_fill_rate

        cvs_fill_rate = mean_fill_rate
        cds_fill_rate = mean_fill_rate

        cvs_fix_holds = bool(
            mean_fill_rate >= 0.90
        )

        # ---------------------------------------------
        # Final record
        # ---------------------------------------------

        return {

            "trial_id":
                trial_id,

            "scenario":
                scenario,

            "mean_fill_rate":
                mean_fill_rate,

            "fill_rate_high":
                fill_rate_high,

            "fill_rate_medium":
                fill_rate_medium,

            "fill_rate_low":
                fill_rate_low,

            "total_stockout_cost":
                total_stockout_cost,

            "tsl_compliance_rate":
                tsl_compliance_rate,

            "cvs_fill_rate":
                cvs_fill_rate,

            "cds_fill_rate":
                cds_fill_rate,

            "cvs_fix_holds":
                cvs_fix_holds,

            "n_skus":
                len(sim_df),

            "scenario_lt_multiplier":
                impact.mean_lt_multiplier,

            "pct_skus_affected":
                impact.pct_skus_affected,
        }

    # =====================================================
    # SINGLE SCENARIO
    # =====================================================

    def run_scenario(
        self,
        sku_df: pd.DataFrame,
        scenario: str,
    ) -> pd.DataFrame:

        rows = []

        for trial_id in range(
            self.n_trials
        ):

            row = self._run_single_trial(
                sku_df=sku_df,
                scenario=scenario,
                trial_id=trial_id,
            )

            rows.append(row)

        return pd.DataFrame(rows)

    # =====================================================
    # ALL SCENARIOS
    # =====================================================

    def run_all(
        self,
        sku_df: pd.DataFrame,
    ) -> pd.DataFrame:

        results = []

        for scenario in SCENARIOS:

            try:

                sc_df = self.run_scenario(
                    sku_df,
                    scenario,
                )

                results.append(sc_df)

                logger.info(
                    "Completed scenario: %s",
                    scenario,
                )

            except Exception as exc:

                logger.exception(
                    "Scenario failed: %s | %s",
                    scenario,
                    str(exc),
                )

        if not results:

            raise RuntimeError(
                "Monte Carlo pipeline produced no scenario outputs."
            )

        return pd.concat(
            results,
            ignore_index=True,
        )


# =========================================================
# CONVENIENCE FUNCTION
# =========================================================

def run_monte_carlo(
    sku_df: pd.DataFrame,
    n_trials: int = 10,
    sim_periods: int = 52,
) -> pd.DataFrame:

    mc = MonteCarloPipeline(
        n_trials=n_trials,
        n_periods=sim_periods,
    )

    return mc.run_all(sku_df)
