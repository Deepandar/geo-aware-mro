from __future__ import annotations

import logging

import pandas as pd

from src.simulation.distribution_fitting import (
    fit_global_distribution,
)

from src.simulation.monte_carlo_rollout import (
    MonteCarloEngine,
)

logger = logging.getLogger(__name__)


class SimulatorPipeline:

    def __init__(
        self,
        n_trials: int = 100,
        horizon: int = 26,
        seed: int = 42,
    ):

        self.n_trials = n_trials
        self.horizon = horizon
        self.seed = seed

        self.engine = MonteCarloEngine(
            n_trials=n_trials,
            horizon=horizon,
            seed=seed,
        )

    # -----------------------------------------------------
    # Run
    # -----------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        # -------------------------------------------------
        # Global lead-time fit
        # -------------------------------------------------

        fit_result = (
            fit_global_distribution(df)
        )

        fit_dist = fit_result.dist_name
        fit_params = fit_result.params

        logger.info(
            (
                "Using global LT distribution | "
                "%s"
            ),
            fit_dist,
        )

        outputs = []

        # -------------------------------------------------
        # Monte Carlo
        # -------------------------------------------------

        for _, row in df.iterrows():

            sku_id = row["item_id"]

            mean_demand = float(
                row["mean_demand"]
            )

            rop = float(
                row["rop"]
            )

            q_star = float(
                row["q_star"]
            )

            result = self.engine.run_policy_simulation(
                sku_id=sku_id,
                mean_demand=mean_demand,
                rop=rop,
                eoq=q_star,
                fit_dist=fit_dist,
                fit_params=fit_params,
            )

            outputs.append(result)

        out = pd.DataFrame(
            outputs
        )

        logger.info(
            (
                "Simulation complete | "
                "rows=%s"
            ),
            len(out),
        )

        return out
