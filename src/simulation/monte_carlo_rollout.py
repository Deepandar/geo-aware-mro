from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from dataclasses import dataclass

from src.simulation.scenario_correlation import (
    ShockTimeline,
    CorrelationPropagator,
    SCENARIO_LIBRARY,
)

logger = logging.getLogger(__name__)


# =========================================================
# Simulation Result
# =========================================================


@dataclass
class SimulationResult:

    item_id: int

    fill_rate: float

    service_level: float

    stockout_days: int

    avg_inventory: float

    holding_cost: float

    stockout_cost: float

    total_cost: float

    orders_placed: int

    final_backlog: float


# =========================================================
# Monte Carlo Engine
# =========================================================


class MonteCarloEngine:

    def __init__(
        self,
        horizon_days: int = 30,
        holding_cost_rate: float = 1.0,
        stockout_penalty: float = 25.0,
        seed: int = 42,
    ):

        self.horizon_days = horizon_days

        self.holding_cost_rate = holding_cost_rate

        self.stockout_penalty = stockout_penalty

        self.rng = np.random.default_rng(seed)

        logger.info("MonteCarloEngine initialised")

    # -----------------------------------------------------
    # Dynamic simulation
    # -----------------------------------------------------

    def run_policy_simulation(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        sku_ids = df["item_id"].astype(str).tolist()

        # -------------------------------------------------
        # Build correlation engine
        # -------------------------------------------------

        propagator = CorrelationPropagator(sku_ids)

        timeline = ShockTimeline(
            T=self.horizon_days,
            sku_ids=sku_ids,
        )

        # -------------------------------------------------
        # Apply all scenarios
        # -------------------------------------------------

        for scenario in SCENARIO_LIBRARY:

            timeline.apply_scenario(
                scenario,
                propagator,
                self.rng,
            )

        results = []

        # -------------------------------------------------
        # Simulate SKU-by-SKU
        # -------------------------------------------------

        for _, row in df.iterrows():

            result = self.simulate_sku(
                row=row,
                timeline=timeline,
            )

            results.append(result.__dict__)

        return pd.DataFrame(results)

    # -----------------------------------------------------
    # Single SKU simulation
    # -----------------------------------------------------

    def simulate_sku(
        self,
        row: pd.Series,
        timeline: ShockTimeline,
    ) -> SimulationResult:

        sku_id = str(row["item_id"])

        mean_demand = float(
            row.get(
                "mean_demand",
                5.0,
            )
        )

        base_lt = float(
            row.get(
                "lead_time_days",
                10.0,
            )
        )

        rop = float(
            row.get(
                "rop",
                100.0,
            )
        )

        q_star = float(
            row.get(
                "q_star",
                50.0,
            )
        )

        inventory = rop

        backlog = 0.0

        stockout_days = 0

        total_demand = 0.0

        fulfilled = 0.0

        holding_cost = 0.0

        stockout_cost = 0.0

        orders_placed = 0

        pipeline_orders = []

        inventory_trace = []

        # -------------------------------------------------
        # Daily simulation loop
        # -------------------------------------------------

        for t in range(self.horizon_days):

            # ---------------------------------------------
            # Receive arrivals
            # ---------------------------------------------

            arrivals = [qty for arr_t, qty in pipeline_orders if arr_t <= t]

            inventory += sum(arrivals)

            pipeline_orders = [
                (arr_t, qty) for arr_t, qty in pipeline_orders if arr_t > t
            ]

            # ---------------------------------------------
            # Get dynamic shock multipliers
            # ---------------------------------------------

            lt_mult, dem_mult = timeline.get_multipliers(
                sku_id,
                t,
            )

            # ---------------------------------------------
            # Shocked demand
            # ---------------------------------------------

            shocked_mean = mean_demand * dem_mult

            demand = self.rng.poisson(
                max(
                    shocked_mean,
                    0.01,
                )
            )

            total_demand += demand

            # ---------------------------------------------
            # Fulfilment
            # ---------------------------------------------

            available = max(
                inventory,
                0,
            )

            shipped = min(
                available,
                demand + backlog,
            )

            inventory -= shipped

            fulfilled += min(
                shipped,
                demand,
            )

            unmet = demand + backlog - shipped

            backlog = unmet

            if unmet > 0:

                stockout_days += 1

            # ---------------------------------------------
            # Inventory costs
            # ---------------------------------------------

            holding_cost += (
                max(
                    inventory,
                    0,
                )
                * self.holding_cost_rate
            )

            stockout_cost += unmet * self.stockout_penalty

            inventory_trace.append(inventory)

            # ---------------------------------------------
            # Replenishment policy
            # ---------------------------------------------

            inventory_position = inventory + sum(qty for _, qty in pipeline_orders)

            if inventory_position <= rop:
                # -----------------------------------------
                # Robust shocked lead-time calculation
                # -----------------------------------------

                safe_base_lt = base_lt

                if not np.isfinite(safe_base_lt):
                    safe_base_lt = 10.0

                safe_lt_mult = lt_mult

                if not np.isfinite(safe_lt_mult):
                    safe_lt_mult = 1.0

                safe_base_lt = float(
                    np.clip(
                        safe_base_lt,
                        1.0,
                        365.0,
                    )
                )

                safe_lt_mult = float(
                    np.clip(
                        safe_lt_mult,
                        0.5,
                        10.0,
                    )
                )

                shocked_lt = max(
                    1,
                    int(round(safe_base_lt * safe_lt_mult)),
                )

                arrival_day = t + shocked_lt

                pipeline_orders.append(
                    (
                        arrival_day,
                        q_star,
                    )
                )

                orders_placed += 1

        # -------------------------------------------------
        # KPIs
        # -------------------------------------------------

        fill_rate = fulfilled / total_demand if total_demand > 0 else 1.0

        service_level = 1.0 - (stockout_days / self.horizon_days)

        avg_inventory = float(np.mean(inventory_trace))

        total_cost = holding_cost + stockout_cost

        return SimulationResult(
            item_id=int(row["item_id"]),
            fill_rate=float(fill_rate),
            service_level=float(service_level),
            stockout_days=int(stockout_days),
            avg_inventory=float(avg_inventory),
            holding_cost=float(holding_cost),
            stockout_cost=float(stockout_cost),
            total_cost=float(total_cost),
            orders_placed=int(orders_placed),
            final_backlog=float(backlog),
        )
