"""
SimPy DES Depot Simulator
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import simpy

from src.simulation.demand_process import (
    build_demand_process,
)

from src.simulation.order_pipeline import (
    SKUInventoryState,
)


@dataclass
class TrialResult:

    trial_id: int

    mean_fill_rate: float

    total_cost: float

    tsl_compliance_rate: float


class DepotSimulator:

    def __init__(
        self,
        sim_periods: int = 52,
        n_trials: int = 100,
        seed: int = 42,
        fast_mode: bool = True,
    ):

        self.sim_periods = sim_periods
        self.n_trials = n_trials
        self.seed = seed
        self.fast_mode = fast_mode

    def _simulate_single_sku(
        self,
        row,
        rng,
    ):

        demand_proc = build_demand_process(
            row.to_dict(),
            rng,
        )

        inventory = float(
            row.get(
                "q_star",
                20.0,
            )
        )

        rop = float(
            row.get(
                "rop",
                10.0,
            )
        )

        total_demand = 0.0
        total_fill = 0.0

        total_holding = 0.0
        total_stockout = 0.0

        unit_cost = 100.0
        holding_rate = 0.20
        stockout_cost = 500.0

        pending_orders = []

        for t in range(self.sim_periods):

            arrivals = [
                q
                for at, q in pending_orders
                if at <= t
            ]

            inventory += sum(arrivals)

            pending_orders = [
                (at, q)
                for at, q in pending_orders
                if at > t
            ]

            demand = demand_proc.sample()

            total_demand += demand

            filled = min(
                inventory,
                demand,
            )

            total_fill += filled

            shortfall = demand - filled

            inventory = max(
                inventory - demand,
                0.0,
            )

            total_stockout += (
                shortfall * stockout_cost
            )

            total_holding += (
                inventory
                * unit_cost
                * holding_rate
            )

            if inventory <= rop:

                lead_time = max(
                    int(
                        rng.normal(
                            6,
                            2,
                        )
                    ),
                    1,
                )

                order_qty = float(
                    row.get(
                        "q_star",
                        20.0,
                    )
                )

                pending_orders.append(
                    (
                        t + lead_time,
                        order_qty,
                    )
                )

        fill_rate = (
            total_fill / total_demand
            if total_demand > 0
            else 1.0
        )

        total_cost = (
            total_holding
            + total_stockout
        )

        tsl_target = float(
            row.get(
                "tsl",
                0.90,
            )
        )

        tsl_met = (
            fill_rate >= tsl_target
        )

        return {
            "fill_rate": fill_rate,
            "total_cost": total_cost,
            "tsl_met": tsl_met,
        }

    def run_trial(
        self,
        sku_df: pd.DataFrame,
        trial_id: int,
    ):

        rng = np.random.default_rng(
            self.seed + trial_id
        )

        rows = []

        for _, row in sku_df.iterrows():

            rows.append(
                self._simulate_single_sku(
                    row,
                    rng,
                )
            )

        df = pd.DataFrame(rows)

        return TrialResult(
            trial_id=trial_id,

            mean_fill_rate=float(
                df["fill_rate"].mean()
            ),

            total_cost=float(
                df["total_cost"].mean()
            ),

            tsl_compliance_rate=float(
                df["tsl_met"].mean()
            ),
        )

    def run_monte_carlo(
        self,
        sku_df: pd.DataFrame,
    ):

        records = []

        for trial_id in range(
            self.n_trials
        ):

            r = self.run_trial(
                sku_df,
                trial_id,
            )

            records.append({

                "trial_id":
                    r.trial_id,

                "mean_fill_rate":
                    r.mean_fill_rate,

                "total_cost":
                    r.total_cost,

                "tsl_compliance_rate":
                    r.tsl_compliance_rate,
            })

        return pd.DataFrame(records)

