from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulation.depot_sim import DepotSimulator
from src.simulation.scenario_injector import ScenarioInjector

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
class TrialResult:
    trial_id: int
    scenario: str

    mean_fill_rate: float

    fill_rate_high: float
    fill_rate_medium: float
    fill_rate_low: float

    total_stockout_cost: float

    tsl_compliance_rate: float

    cvs_fill_rate: float
    cds_fill_rate: float
    cvs_fix_holds: bool

    cost_high_tier: float
    cost_medium_tier: float
    cost_low_tier: float


class MonteCarloPipeline:

    def __init__(
        self,
        n_trials: int = 100,
        sim_periods: int = 26,
        fast_mode: bool = False,
        seed: int = 42,
    ):
        self.n_trials = n_trials
        self.sim_periods = sim_periods
        self.fast_mode = fast_mode

        self.rng = np.random.default_rng(seed)

        self.injector = ScenarioInjector(
            "config/scenario_library.yaml"
        )

        logger.info(
            "MonteCarloPipeline initialized | "
            "trials=%d periods=%d",
            n_trials,
            sim_periods,
        )

    def _run_single_trial(
        self,
        sku_df: pd.DataFrame,
        scenario: str,
        trial_id: int,
    ) -> dict:

        impact = self.injector.inject(
            sku_df,
            scenario,
        )

        sim_df = impact.modified_df.copy()

        sim = DepotSimulator(
            periods=self.sim_periods,
            seed=int(self.rng.integers(0, 1_000_000)),
        )

        sim_result = sim.run(sim_df)

        overall_fill = float(
            sim_result["fill_rate"].mean()
        )

        high_df = sim_result[
            sim_result["ci_tier"] == "High"
        ]

        med_df = sim_result[
            sim_result["ci_tier"] == "Medium"
        ]

        low_df = sim_result[
            sim_result["ci_tier"] == "Low"
        ]

        fill_high = (
            float(high_df["fill_rate"].mean())
            if len(high_df)
            else overall_fill
        )

        fill_med = (
            float(med_df["fill_rate"].mean())
            if len(med_df)
            else overall_fill
        )

        fill_low = (
            float(low_df["fill_rate"].mean())
            if len(low_df)
            else overall_fill
        )

        stockout_cost = float(
            sim_result["total_stockout_cost"].sum()
        )

        tsl_compliance = float(
            (
                sim_result["fill_rate"]
                >= sim_result["tsl"]
            ).mean()
        )

        cvs_fill = fill_high
        cds_fill = fill_low

        cvs_fix = bool(
            cvs_fill >= cds_fill
        )

        return TrialResult(
            trial_id=trial_id,
            scenario=scenario,

            mean_fill_rate=overall_fill,

            fill_rate_high=fill_high,
            fill_rate_medium=fill_med,
            fill_rate_low=fill_low,

            total_stockout_cost=stockout_cost,

            tsl_compliance_rate=tsl_compliance,

            cvs_fill_rate=cvs_fill,
            cds_fill_rate=cds_fill,
            cvs_fix_holds=cvs_fix,

            cost_high_tier=float(
                high_df["total_stockout_cost"].sum()
                if len(high_df) else 0.0
            ),

            cost_medium_tier=float(
                med_df["total_stockout_cost"].sum()
                if len(med_df) else 0.0
            ),

            cost_low_tier=float(
                low_df["total_stockout_cost"].sum()
                if len(low_df) else 0.0
            ),
        ).__dict__

    def run_scenario(
        self,
        sku_df: pd.DataFrame,
        scenario: str,
    ) -> pd.DataFrame:

        rows = []

        for trial in range(self.n_trials):

            rows.append(
                self._run_single_trial(
                    sku_df=sku_df,
                    scenario=scenario,
                    trial_id=trial,
                )
            )

        out = pd.DataFrame(rows)

        logger.info(
            "Scenario complete | %s | rows=%d",
            scenario,
            len(out),
        )

        return out

    def run_all(
        self,
        sku_df: pd.DataFrame,
        mlflow_parent_run: bool = False,
    ) -> pd.DataFrame:

        all_results = []

        for scenario in SCENARIOS:

            try:

                sc_df = self.run_scenario(
                    sku_df,
                    scenario,
                )

                all_results.append(sc_df)

            except Exception as e:

                logger.exception(
                    "Scenario failed: %s",
                    scenario,
                )

                if not self.fast_mode:
                    raise e

        combined = pd.concat(
            all_results,
            ignore_index=True,
        )

        logger.info(
            "Monte Carlo complete | rows=%d",
            len(combined),
        )

        return combined
