from __future__ import annotations

import itertools
import logging
import math

from dataclasses import dataclass
from typing import Callable

import mlflow
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# COALITION VALUE FUNCTION
# ============================================================

def compute_coalition_cost(

    depot_subset: list[str],

    sku_df: pd.DataFrame,

    holding_rate: float = 0.20,

    pooling_factor: float = 0.25,

) -> float:

    n = len(depot_subset)

    if n == 0:
        return 0.0

    total_cost = 0.0

    for _, row in sku_df.iterrows():

        uc = float(
            row.get("unit_cost", 100.0)
        )

        q = float(
            row.get(
                "dp_q_star",
                row.get("q_star", 10.0),
            )
        )

        depot = str(
            row.get("depot_tier", "Rear")
        )

        if depot not in depot_subset and n < 3:
            continue

        base_hold = (
            holding_rate
            * uc
            * q
        )

        if n > 1:

            pooling_reduction = (

                1.0

                - (

                    1.0
                    - 1.0 / math.sqrt(n)

                )

                * pooling_factor

            )

            hold_cost = (
                base_hold
                * pooling_reduction
            )

        else:

            hold_cost = base_hold

        total_cost += hold_cost

    return round(total_cost, 2)


# ============================================================
# RL REWARD FUNCTION
# ============================================================

def cooperative_reward(

    shapley_saving: float,

    shortage_penalty: float,

    geo_risk: float,

    coalition_health: float,

) -> float:

    reward = (

        shapley_saving * 0.40

        - shortage_penalty * 0.35

        - geo_risk * 1000.0

        + coalition_health * 500.0

    )

    return float(reward)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class ShapleyResult:

    depot_id: str

    shapley_value: float

    pro_rata_value: float

    solo_cost: float

    shapley_payment: float

    pro_rata_payment: float

    savings_vs_solo: float

    savings_vs_pro_rata: float

    sku_count: int

    mean_ci_score: float


@dataclass
class ShapleyAllocation:

    n_depots: int

    coalition_cost: float

    sum_solo_costs: float

    total_pooling_saving: float

    shapley_sum_check: float

    efficiency_satisfied: bool

    depot_results: list[ShapleyResult]

    coalition_table: pd.DataFrame


# ============================================================
# SHAPLEY ALLOCATOR
# ============================================================

class ShapleyAllocator:

    def __init__(

        self,

        depots=None,

        holding_rate=0.20,

        pooling_factor=0.25,

        value_function: Callable | None = None,

    ):

        self.depots = depots or [

            "Forward",

            "Border",

            "Rear",

        ]

        self.holding_rate = holding_rate

        self.pooling_factor = pooling_factor

        self.value_fn = (
            value_function
            or compute_coalition_cost
        )

        self.n = len(self.depots)

    # ========================================================
    # ALL COALITION VALUES
    # ========================================================

    def _all_coalition_values(

        self,

        sku_df,

    ):

        coalition_values = {

            frozenset(): 0.0

        }

        for r in range(1, self.n + 1):

            for combo in itertools.combinations(
                self.depots,
                r,
            ):

                S = frozenset(combo)

                v_S = self.value_fn(

                    list(combo),

                    sku_df,

                    self.holding_rate,

                    self.pooling_factor,

                )

                coalition_values[S] = v_S

        return coalition_values

    # ========================================================
    # SHAPLEY VALUE
    # ========================================================

    def _shapley_value_i(

        self,

        depot_i,

        coalition_values,

    ):

        n = self.n

        others = [

            d for d in self.depots
            if d != depot_i

        ]

        phi_i = 0.0

        for r in range(len(others) + 1):

            for combo in itertools.combinations(
                others,
                r,
            ):

                S_set = frozenset(combo)

                S_plus = (
                    frozenset(combo)
                    | {depot_i}
                )

                S_size = len(S_set)

                weight = (

                    math.factorial(S_size)

                    * math.factorial(
                        n - S_size - 1
                    )

                    / math.factorial(n)

                )

                v_S = coalition_values.get(
                    S_set,
                    0.0,
                )

                v_S_plus = coalition_values.get(
                    S_plus,
                    0.0,
                )

                phi_i += (
                    weight
                    * (v_S_plus - v_S)
                )

        return round(phi_i, 2)

    # ========================================================
    # MAIN ALLOCATION
    # ========================================================

    def allocate(self, sku_df):

        self._validate(sku_df)

        cv = self._all_coalition_values(
            sku_df
        )

        full_N = frozenset(self.depots)

        v_N = cv.get(full_N, 0.0)

        solo_costs = {

            d: cv.get(
                frozenset([d]),
                0.0,
            )

            for d in self.depots

        }

        sum_solo = sum(
            solo_costs.values()
        )

        shapley_vals = {

            d: self._shapley_value_i(
                d,
                cv,
            )

            for d in self.depots

        }

        shapley_sum = sum(
            shapley_vals.values()
        )

        depot_sku_counts = (

            sku_df["depot_tier"]
            .value_counts()
            .reindex(
                self.depots,
                fill_value=0,
            )
            .to_dict()

        )

        total_skus = sum(
            depot_sku_counts.values()
        )

        pro_rata_vals = {

            d: round(

                v_N
                * depot_sku_counts[d]
                / max(total_skus, 1),

                2,

            )

            for d in self.depots

        }

        results = []

        for d in self.depots:

            sub = sku_df[
                sku_df["depot_tier"] == d
            ]

            results.append(

                ShapleyResult(

                    depot_id=d,

                    shapley_value=shapley_vals[d],

                    pro_rata_value=pro_rata_vals[d],

                    solo_cost=solo_costs[d],

                    shapley_payment=shapley_vals[d],

                    pro_rata_payment=pro_rata_vals[d],

                    savings_vs_solo=round(
                        solo_costs[d]
                        - shapley_vals[d],
                        2,
                    ),

                    savings_vs_pro_rata=round(
                        pro_rata_vals[d]
                        - shapley_vals[d],
                        2,
                    ),

                    sku_count=depot_sku_counts[d],

                    mean_ci_score=round(
                        sub["ci_score"].mean(),
                        3,
                    )
                    if len(sub) > 0
                    else 0.0,

                )

            )

        coal_rows = []

        for S, v_S in cv.items():

            coal_rows.append({

                "coalition":
                    str(sorted(S))
                    if S
                    else "∅",

                "size": len(S),

                "cost_v_S": v_S,

            })

        coal_df = pd.DataFrame(
            coal_rows
        ).sort_values(
            ["size", "coalition"]
        )

        efficiency_ok = (

            abs(shapley_sum - v_N)

            < max(v_N * 0.01, 1.0)

        )

        return ShapleyAllocation(

            n_depots=self.n,

            coalition_cost=round(v_N, 2),

            sum_solo_costs=round(sum_solo, 2),

            total_pooling_saving=round(
                sum_solo - v_N,
                2,
            ),

            shapley_sum_check=round(
                shapley_sum,
                2,
            ),

            efficiency_satisfied=efficiency_ok,

            depot_results=results,

            coalition_table=coal_df,

        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate(self, df):

        required = {

            "item_id",

            "depot_tier",

            "unit_cost",

        }

        missing = required - set(df.columns)

        if missing:

            raise ValueError(
                f"missing columns: {missing}"
            )

    # ========================================================
    # RL STATE VECTOR
    # ========================================================

    def build_rl_state_vector(

        self,

        inventory_level: float,

        failure_risk: float,

        geo_risk: float,

        coalition_saving: float,

        supplier_rep: float,

    ) -> np.ndarray:

        return np.array([

            inventory_level,

            failure_risk,

            geo_risk,

            coalition_saving,

            supplier_rep,

        ], dtype=np.float32)

    # ========================================================
    # MLFLOW
    # ========================================================

    def log_to_mlflow(

        self,

        alloc,

        run_name="shapley_v1.3",

    ):

        with mlflow.start_run(

            run_name=run_name,

            nested=True,

        ):

            mlflow.log_metric(
                "coalition_cost",
                alloc.coalition_cost,
            )

            mlflow.log_metric(
                "pooling_saving",
                alloc.total_pooling_saving,
            )

            mlflow.log_metric(
                "efficiency_ok",
                int(
                    alloc.efficiency_satisfied
                ),
            )


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    n = 90

    df = pd.DataFrame({

        "item_id": [
            f"SKU{i:04d}"
            for i in range(n)
        ],

        "depot_tier":
            ["Forward"] * 30
            + ["Border"] * 35
            + ["Rear"] * 25,

        "unit_cost": np.random.uniform(
            100,
            5000,
            n,
        ),

        "q_star": np.random.uniform(
            5,
            30,
            n,
        ),

        "dp_q_star": np.random.uniform(
            8,
            35,
            n,
        ),

        "ci_score": np.random.uniform(
            0.2,
            0.95,
            n,
        ),

    })

    alloc = ShapleyAllocator().allocate(df)

    print("\n==================================================")
    print("W28 SHAPLEY VALUE RESULTS")
    print("==================================================")
    print(f"Coalition Cost     : {alloc.coalition_cost:,.2f}")
    print(f"Solo Cost Sum      : {alloc.sum_solo_costs:,.2f}")
    print(f"Pooling Saving     : {alloc.total_pooling_saving:,.2f}")
    print(f"Efficiency Check   : {alloc.efficiency_satisfied}")
    print("==================================================")

    for r in alloc.depot_results:

        print(
            f"{r.depot_id:8s} | "
            f"phi={r.shapley_value:10.2f} | "
            f"solo={r.solo_cost:10.2f} | "
            f"save={r.savings_vs_solo:10.2f}"
        )

    allocator = ShapleyAllocator()

    rl_state = allocator.build_rl_state_vector(

        inventory_level=120.0,

        failure_risk=0.15,

        geo_risk=0.30,

        coalition_saving=alloc.total_pooling_saving,

        supplier_rep=0.88,

    )

    reward = cooperative_reward(

        shapley_saving=alloc.total_pooling_saving,

        shortage_penalty=2500.0,

        geo_risk=0.30,

        coalition_health=0.92,

    )

    print("\n==================================================")
    print("RL COALITION STATE")
    print("==================================================")
    print("State Vector:")
    print(rl_state)

    print("\nReward:")
    print(round(reward, 2))

    print("==================================================")
