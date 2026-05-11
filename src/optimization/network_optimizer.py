from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Dict

import mlflow
import numpy as np
import pandas as pd
import pulp

logger = logging.getLogger(__name__)

# ============================================================
# DEPOT CONFIGURATION
# ============================================================

DEPOT_TIERS = ["Forward", "Border", "Rear"]

DEPOT_CAPACITY = {
    "Forward": 5000.0,
    "Border": 15000.0,
    "Rear": 999999.0,
}

DEPOT_WEIGHT_CAPACITY = {
    "Forward": 10000.0,
    "Border": 50000.0,
    "Rear": 999999.0,
}

DEPOT_PARAMS = {

    "Forward": {
        "holding_cost_mult": 1.30,
        "transport_cost_mult": 0.80,
    },

    "Border": {
        "holding_cost_mult": 1.00,
        "transport_cost_mult": 1.00,
    },

    "Rear": {
        "holding_cost_mult": 0.75,
        "transport_cost_mult": 1.40,
    },
}

# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class SolverResult:

    status: str
    objective_value: float
    solve_time_seconds: float


# ============================================================
# OPTIMIZER
# ============================================================

class MILPNetworkOptimizer:

    def __init__(
        self,
        solver: str = "CBC",
        time_limit_sec: int = 60,
    ):

        self.solver = solver
        self.time_limit_sec = time_limit_sec

        self.prob = None

    # ========================================================
    # SOLVE
    # ========================================================

    def solve(self, sku_df: pd.DataFrame):

        required = [
            "item_id",
            "unit_cost",
            "q_star",
        ]

        missing = [
            c for c in required
            if c not in sku_df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        t0 = time.time()

        depots = DEPOT_TIERS

        n = len(sku_df)

        # ----------------------------------------------------
        # NUMPY PREPROCESSING
        # ----------------------------------------------------

        unit_cost = sku_df["unit_cost"].to_numpy(dtype=float)

        q_star = sku_df["q_star"].to_numpy(dtype=float)

        geo_risk = sku_df.get(
            "geo_risk_score",
            0.0
        ).to_numpy(dtype=float)

        volume = sku_df.get(
            "volume_unit",
            10.0
        ).to_numpy(dtype=float)

        weight = sku_df.get(
            "weight_unit",
            5.0
        ).to_numpy(dtype=float)

        # ----------------------------------------------------
        # BUILD LP
        # ----------------------------------------------------

        prob = pulp.LpProblem(
            "Capacity_Aware_MRO",
            pulp.LpMinimize,
        )

        y = {}

        for i in range(n):

            for j in range(len(depots)):

                y[(i, j)] = pulp.LpVariable(
                    name=f"y_{i}_{j}",
                    cat="Binary",
                )

        # ----------------------------------------------------
        # PRECOMPUTE COSTS
        # ----------------------------------------------------

        holding_cost = np.zeros((n, len(depots)))
        transport_cost = np.zeros((n, len(depots)))
        risk_cost = np.zeros((n, len(depots)))

        for j, depot in enumerate(depots):

            params = DEPOT_PARAMS[depot]

            holding_cost[:, j] = (
                unit_cost
                * 0.20
                * params["holding_cost_mult"]
            )

            transport_cost[:, j] = (
                q_star
                * unit_cost
                * 0.05
                * params["transport_cost_mult"]
            )

            risk_cost[:, j] = (
                geo_risk
                * unit_cost
                * 0.15
            )

        total_cost = (
            holding_cost
            + transport_cost
            + risk_cost
        )

        # ----------------------------------------------------
        # OBJECTIVE
        # ----------------------------------------------------

        prob += pulp.lpSum(

            total_cost[i, j] * y[(i, j)]

            for i in range(n)
            for j in range(len(depots))

        )

        # ----------------------------------------------------
        # UNIQUE ASSIGNMENT
        # ----------------------------------------------------

        for i in range(n):

            prob += (

                pulp.lpSum(

                    y[(i, j)]

                    for j in range(len(depots))

                ) == 1,

                f"UniqueAssignment_{i}"

            )

        # ----------------------------------------------------
        # VOLUME CAPACITY
        # ----------------------------------------------------

        for j, depot in enumerate(depots):

            prob += (

                pulp.lpSum(

                    volume[i] * y[(i, j)]

                    for i in range(n)

                )

                <= DEPOT_CAPACITY[depot],

                f"VolumeCapacity_{depot}"

            )

        # ----------------------------------------------------
        # WEIGHT CAPACITY
        # ----------------------------------------------------

        for j, depot in enumerate(depots):

            prob += (

                pulp.lpSum(

                    weight[i] * y[(i, j)]

                    for i in range(n)

                )

                <= DEPOT_WEIGHT_CAPACITY[depot],

                f"WeightCapacity_{depot}"

            )

        # ----------------------------------------------------
        # TEMP CBC DIRECTORY
        # ----------------------------------------------------

        temp_dir = tempfile.mkdtemp(
            prefix="cbc_solver_"
        )

        try:

            solver_cmd = pulp.PULP_CBC_CMD(

                msg=False,

                warmStart=True,

                keepFiles=True,

                timeLimit=self.time_limit_sec,

                presolve=True,

                cuts=True,

            )

            # force CBC working directory
            solver_cmd.tmpDir = temp_dir

            # ------------------------------------------------
            # SOLVE
            # ------------------------------------------------

            status_code = prob.solve(solver_cmd)

        finally:

            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )

        status = pulp.LpStatus[status_code]

        elapsed = round(
            time.time() - t0,
            2
        )

        self.prob = prob

        # ----------------------------------------------------
        # SHADOW PRICES
        # ----------------------------------------------------

        print("\n==================================================")
        print("SHADOW PRICES / DUAL VALUES")
        print("==================================================")

        for name, constraint in prob.constraints.items():

            try:

                print(
                    f"{name:30s} | "
                    f"Dual={constraint.pi:10.4f} | "
                    f"Slack={constraint.slack:10.4f}"
                )

            except Exception:

                pass

        return SolverResult(
            status=status,
            objective_value=float(
                pulp.value(prob.objective) or 0.0
            ),
            solve_time_seconds=elapsed,
        )

    # ========================================================
    # SENSITIVITY REPORT
    # ========================================================

    def get_sensitivity_report(self) -> Dict:

        if self.prob is None:

            raise RuntimeError(
                "Run solve() first"
            )

        report = {}

        for name, constraint in self.prob.constraints.items():

            if "Capacity" in name:

                report[name] = {

                    "dual_value": round(
                        constraint.pi,
                        4
                    ),

                    "slack": round(
                        constraint.slack,
                        4
                    ),
                }

        return report

    # ========================================================
    # MLFLOW TRACKING
    # ========================================================

    def solve_with_tracking(

        self,

        sku_df: pd.DataFrame,

        experiment_name: str = "W25_Optimization"

    ):

        mlflow.set_experiment(
            experiment_name
        )

        with mlflow.start_run(

            run_name=f"MILP_{int(time.time())}"

        ):

            result = self.solve(sku_df)

            mlflow.log_param(
                "n_skus",
                len(sku_df)
            )

            mlflow.log_param(
                "solver",
                self.solver
            )

            mlflow.log_metric(
                "objective_value",
                result.objective_value
            )

            mlflow.log_metric(
                "solve_time_seconds",
                result.solve_time_seconds
            )

            sensitivity = (
                self.get_sensitivity_report()
            )

            for name, values in sensitivity.items():

                mlflow.log_metric(
                    f"dual_{name}",
                    values["dual_value"]
                )

            return result


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    sample_df = pd.DataFrame({

        "item_id": [
            "SKU001",
            "SKU002",
            "SKU003",
            "SKU004",
        ],

        "unit_cost": [
            1000,
            2000,
            1500,
            3000,
        ],

        "q_star": [
            20,
            15,
            30,
            25,
        ],

        "geo_risk_score": [
            0.2,
            0.4,
            0.1,
            0.6,
        ],

        "volume_unit": [
            50,
            75,
            100,
            125,
        ],

        "weight_unit": [
            100,
            200,
            150,
            300,
        ],
    })

    optimizer = MILPNetworkOptimizer()

    result = optimizer.solve_with_tracking(sample_df)

    print("\n==================================================")
    print("MILP SOLVER RESULT")
    print("==================================================")
    print(f"Status     : {result.status}")
    print(f"Objective  : {result.objective_value:.2f}")
    print(f"Solve Time : {result.solve_time_seconds}s")
    print("==================================================")
