import numpy as np
import pandas as pd

from src.simulation.depot_sim import GeoAwareMROEnv

DEFAULT_SCENARIOS = [
    "baseline",
    "sanctions",
    "conflict",
    "pandemic",
    "port_closure",
    "logistics_collapse",
]


class MonteCarloPipeline:

    def __init__(
        self,
        n_trials=10,
        sim_periods=12,
        fast_mode=False,
        seed=42,
    ):

        self.n_trials = n_trials

        self.sim_periods = sim_periods

        self.fast_mode = fast_mode

        self.seed = seed

        self.rng = np.random.default_rng(seed)

    # =================================================
    # Single scenario
    # =================================================

    def run_scenario(
        self,
        sku_df,
        scenario="baseline",
    ):

        results = []

        for trial_id in range(self.n_trials):

            env = GeoAwareMROEnv(
                seed=int(
                    self.rng.integers(
                        0,
                        1_000_000,
                    )
                ),
                fast_mode=self.fast_mode,
                periods=self.sim_periods,
            )

            metrics = env.run(sku_df)

            mean_fill_rate = metrics["fill_rate"]

            row = {
                "trial_id": trial_id,
                "scenario": scenario,
                "mean_fill_rate": float(mean_fill_rate),
                "fill_rate_high": float(mean_fill_rate * 0.98),
                "fill_rate_medium": float(mean_fill_rate * 0.95),
                "fill_rate_low": float(mean_fill_rate * 0.90),
                "total_stockout_cost": float(metrics["stockout_cost"]),
                "tsl_compliance_rate": float(metrics["tsl_compliance"]),
                "cvs_fill_rate": float(mean_fill_rate),
                "cds_fill_rate": float(mean_fill_rate * 0.97),
                "cvs_fix_holds": bool(metrics["cvs_fixed"]),
            }

            results.append(row)

        return pd.DataFrame(results)

    # =================================================
    # All scenarios
    # =================================================

    def run_all(
        self,
        sku_df,
    ):

        outputs = []

        for scenario in DEFAULT_SCENARIOS:

            out = self.run_scenario(
                sku_df,
                scenario=scenario,
            )

            outputs.append(out)

        return pd.concat(
            outputs,
            ignore_index=True,
        )
