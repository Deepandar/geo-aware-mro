# src/optimization/bellman_engine.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BellmanEngine:
    """
    Dynamic inventory optimization engine.

    Implements:
        - bounded inventory states
        - stochastic demand sampling
        - approximate Bellman recursion
        - capped reorder policies
    """

    def __init__(
        self,
        discount_factor: float = 0.95,
        holding_cost: float = 1.0,
        stockout_cost: float = 10.0,
        max_inventory: int = 500,
        max_order: int = 250,
    ):

        self.beta = discount_factor

        self.holding_cost = holding_cost

        self.stockout_cost = stockout_cost

        self.max_inventory = max_inventory

        self.max_order = max_order

        # -----------------------------------------------------
        # Approximate value table
        # -----------------------------------------------------

        self.value_table = np.zeros(self.max_inventory + 1)

        logger.info(
            (
                "BellmanEngine initialised | "
                "beta=%.2f | "
                "holding=%.2f | "
                "stockout=%.2f"
            ),
            self.beta,
            self.holding_cost,
            self.stockout_cost,
        )

    # ---------------------------------------------------------
    # Main optimization
    # ---------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        self._validate(df)

        out = df.copy()

        q_star = []

        rop = []

        expected_cost = []

        state_value = []

        for _, row in out.iterrows():

            inventory_state = float(
                row.get(
                    "current_inventory",
                    100,
                )
            )

            mean_demand = float(row["mean_demand"])

            std_demand = float(
                row.get(
                    "std_demand",
                    max(mean_demand * 0.3, 1),
                )
            )

            lead_time = float(row["lead_time_days"])

            ci_score = float(row["ci_score"])

            geo_risk = float(row["geo_risk_score"])

            # -------------------------------------------------
            # Cap lead times
            # -------------------------------------------------

            lead_time = np.clip(
                lead_time,
                1.0,
                365.0,
            )

            # -------------------------------------------------
            # Dynamic stockout penalty
            # -------------------------------------------------

            effective_stockout = (
                self.stockout_cost * (1.0 + ci_score) * (1.0 + geo_risk)
            )

            best_q = 0

            best_cost = np.inf

            # -------------------------------------------------
            # Action search
            # -------------------------------------------------

            for q in range(
                0,
                self.max_order + 1,
                10,
            ):

                # ---------------------------------------------
                # Stochastic demand sample
                # ---------------------------------------------

                sampled_demand = np.random.poisson(max(mean_demand, 1))

                next_inventory = inventory_state + q - sampled_demand

                # ---------------------------------------------
                # Bounded state space
                # ---------------------------------------------

                next_inventory = np.clip(
                    next_inventory,
                    0,
                    self.max_inventory,
                )

                next_inventory_idx = int(next_inventory)

                # ---------------------------------------------
                # Immediate costs
                # ---------------------------------------------

                holding = self.holding_cost * max(
                    next_inventory,
                    0,
                )

                shortage = effective_stockout * max(
                    sampled_demand - (inventory_state + q),
                    0,
                )

                # ---------------------------------------------
                # Approximate Bellman recursion
                # ---------------------------------------------

                future_cost = self.beta * self.value_table[next_inventory_idx]

                total_cost = q + holding + shortage + future_cost

                if total_cost < best_cost:

                    best_cost = total_cost

                    best_q = q

            # -------------------------------------------------
            # Stabilized stochastic ROP
            # -------------------------------------------------

            z = 1.65

            mean_lt_demand = mean_demand * lead_time

            std_lt_demand = np.sqrt(lead_time) * std_demand

            dynamic_rop = mean_lt_demand + (z * std_lt_demand)

            dynamic_rop *= 1.0 + (0.25 * ci_score)

            # -------------------------------------------------
            # Cap extreme ROPs
            # -------------------------------------------------

            dynamic_rop = np.clip(
                dynamic_rop,
                1.0,
                50000.0,
            )

            # -------------------------------------------------
            # Store results
            # -------------------------------------------------

            q_star.append(float(best_q))

            rop.append(float(dynamic_rop))

            expected_cost.append(float(best_cost))

            state_value.append(float(-best_cost))

        # -----------------------------------------------------
        # Output columns
        # -----------------------------------------------------

        out["bellman_q_star"] = q_star

        out["bellman_rop"] = rop

        out["expected_future_cost"] = expected_cost

        out["state_value"] = state_value

        # -----------------------------------------------------
        # Backward compatibility
        # -----------------------------------------------------

        out["q_star"] = out["bellman_q_star"]

        out["rop"] = out["bellman_rop"]

        logger.info(
            ("Bellman optimisation complete | " "mean_q=%.2f | " "mean_rop=%.2f"),
            out["q_star"].mean(),
            out["rop"].mean(),
        )

        return out

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate(
        self,
        df: pd.DataFrame,
    ) -> None:

        required = {
            "mean_demand",
            "lead_time_days",
            "ci_score",
            "geo_risk_score",
        }

        missing = required - set(df.columns)

        if missing:

            raise ValueError(("BellmanEngine missing " f"columns: {missing}"))
