import numpy as np
import pandas as pd


class GeoAwareMROEnv:

    def __init__(
        self,
        seed=None,
        fast_mode=False,
        periods=12,
        initial_inventory=1000,
        n_trials=5,
        **kwargs,
    ):
        self.seed = seed
        self.fast_mode = fast_mode
        self.periods = periods
        self.initial_inventory = initial_inventory
        self.n_trials = n_trials

        self.rng = np.random.default_rng(seed)

    # =================================================
    # Single simulation
    # =================================================

    def run(self, sku_df):

        inventory = self.initial_inventory

        total_demand = 0
        fulfilled_demand = 0

        stockout_cost = 0.0
        holding_cost = 0.0

        for _ in range(self.periods):

            demand = int(
                self.rng.normal(
                    loc=100,
                    scale=20,
                )
            )

            demand = max(demand, 0)

            total_demand += demand

            fulfilled = min(
                inventory,
                demand,
            )

            fulfilled_demand += fulfilled

            inventory -= fulfilled

            unmet = demand - fulfilled

            stockout_cost += unmet * 50.0

            holding_cost += inventory * 0.5

            inventory += 200

        fill_rate = fulfilled_demand / total_demand if total_demand > 0 else 1.0

        return {
            "fill_rate": float(fill_rate),
            "tsl_compliance": float(fill_rate),
            "stockout_cost": float(stockout_cost),
            "holding_cost": float(holding_cost),
            "total_cost": float(stockout_cost + holding_cost),
            "cvs_fixed": bool(fill_rate > 0.8),
        }

    # =================================================
    # TEST-COMPATIBLE MONTE CARLO API
    # =================================================

    def run_monte_carlo(
        self,
        sku_df,
    ):

        outputs = []

        for trial_id in range(self.n_trials):

            metrics = self.run(sku_df)

            outputs.append(
                {
                    "trial_id": trial_id,
                    "mean_fill_rate": metrics["fill_rate"],
                    "total_cost": metrics["total_cost"],
                    "stockout_cost": metrics["stockout_cost"],
                    "holding_cost": metrics["holding_cost"],
                    "tsl_compliance": metrics["tsl_compliance"],
                    "cvs_fixed": metrics["cvs_fixed"],
                }
            )

        return pd.DataFrame(outputs)
