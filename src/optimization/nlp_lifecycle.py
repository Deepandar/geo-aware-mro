from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import weibull_min

logger = logging.getLogger(__name__)

# ============================================================
# RESULT OBJECT
# ============================================================


@dataclass
class LifecycleResult:

    item_id: str

    optimal_holding_qty: float
    optimal_total_cost: float

    holding_cost_at_opt: float
    obsolescence_cost_at_opt: float

    shortage_penalty_at_opt: float
    readiness_penalty_at_opt: float

    replacement_cost_at_opt: float

    obsolescence_prob: float

    kkt_stationarity: float
    kkt_satisfied: bool

    action: str

    age_years: float
    write_off_threshold_yrs: float


# ============================================================
# OPTIMIZER
# ============================================================


class KKTLifecycleOptimizer:

    def __init__(
        self,
        h_rate: float = 0.20,
        lambda_obs: float = 8.0,
        k_weibull: float = 2.5,
        scarcity_base: float = 0.30,
        budget_cap_per_sku: float = 50000.0,
        kkt_tol: float = 1e-2,
        softplus_kappa: float = 15.0,
    ):

        self.h_rate = h_rate

        self.lambda_obs = lambda_obs

        self.k_weibull = k_weibull

        self.scarcity_base = scarcity_base

        self.budget_cap_per_sku = budget_cap_per_sku

        self.kkt_tol = kkt_tol

        self.softplus_kappa = softplus_kappa

    # ========================================================
    # OBSOLESCENCE PROBABILITY
    # ========================================================

    def _obsolescence_probability(
        self,
        age: float,
        geo_risk: float = 0.0,
    ) -> float:

        effective_lambda = self.lambda_obs / (1.0 + geo_risk * 0.5)

        return float(
            weibull_min.cdf(
                age,
                c=self.k_weibull,
                scale=effective_lambda,
            )
        )

    # ========================================================
    # VINTAGE PENALTY
    # ========================================================

    def _vintage_penalty(
        self,
        age: float,
    ) -> float:

        return 0.02 * (age**1.5)

    # ========================================================
    # SCARCITY FACTOR
    # ========================================================

    def _scarcity_factor(
        self,
        age: float,
    ) -> float:

        return self.scarcity_base * (1 + (age / self.lambda_obs) ** 2)

    # ========================================================
    # SOFTPLUS
    # ========================================================

    def _softplus(
        self,
        z: float,
    ) -> float:

        k = self.softplus_kappa

        z_clip = np.clip(
            z,
            -50,
            50,
        )

        return float(np.log1p(np.exp(k * z_clip)) / k)

    # ========================================================
    # TOTAL COST
    # ========================================================

    def _total_cost(
        self,
        x: float,
        unit_cost: float,
        age: float,
        geo_risk: float,
        q_optimal: float,
    ) -> float:

        x = max(float(x), 1e-6)

        # ----------------------------------------------------
        # HOLDING COST
        # ----------------------------------------------------

        holding_cost = self.h_rate * unit_cost * x * (1.0 + self._vintage_penalty(age))

        # ----------------------------------------------------
        # OBSOLESCENCE
        # ----------------------------------------------------

        obs_prob = self._obsolescence_probability(
            age,
            geo_risk,
        )

        replacement_cost = unit_cost * (1.0 + self._scarcity_factor(age))

        obsolescence_cost = (
            obs_prob
            * replacement_cost
            * min(
                x / max(q_optimal, 1.0),
                1.5,
            )
        )

        # ----------------------------------------------------
        # SOFTPLUS SHORTAGE
        # ----------------------------------------------------

        diff = q_optimal - x

        unmet_demand = self._softplus(diff)

        # ----------------------------------------------------
        # SHORTAGE PENALTY
        # ----------------------------------------------------

        shortage_penalty = unmet_demand * replacement_cost * (2.0 + geo_risk + obs_prob)

        # ----------------------------------------------------
        # READINESS PENALTY
        # ----------------------------------------------------

        readiness_penalty = unmet_demand**1.2 * unit_cost * 0.5

        return holding_cost + obsolescence_cost + shortage_penalty + readiness_penalty

    # ========================================================
    # KKT STATIONARITY
    # ========================================================

    def _kkt_stationarity(
        self,
        x: float,
        unit_cost: float,
        age: float,
        geo_risk: float,
        q_optimal: float,
    ) -> float:

        eps = 1e-5

        f1 = self._total_cost(
            x + eps,
            unit_cost,
            age,
            geo_risk,
            q_optimal,
        )

        f2 = self._total_cost(
            x - eps,
            unit_cost,
            age,
            geo_risk,
            q_optimal,
        )

        grad = (f1 - f2) / (2 * eps)

        return abs(float(grad))

    # ========================================================
    # OPTIMIZE SINGLE SKU
    # ========================================================

    def optimize_sku(
        self,
        row: Dict,
    ) -> LifecycleResult:

        iid = str(row.get("item_id", "UNKNOWN"))

        unit_cost = float(row.get("unit_cost", 1000.0))

        q_opt = float(row.get("dp_q_star", row.get("q_star", 10.0)))

        geo_risk = float(row.get("geo_risk_score", 0.0))

        age = float(row.get("item_age_years", 2.0))

        # ----------------------------------------------------
        # OBJECTIVE
        # ----------------------------------------------------

        def objective(x_arr):

            return self._total_cost(
                x_arr[0],
                unit_cost,
                age,
                geo_risk,
                q_opt,
            )

        # ----------------------------------------------------
        # BOUNDS
        # ----------------------------------------------------

        bounds = [(0.0, q_opt * 2.5)]

        # ----------------------------------------------------
        # INITIALIZATION
        # ----------------------------------------------------

        x0 = np.array([q_opt])

        # ----------------------------------------------------
        # OPTIMIZATION
        # ----------------------------------------------------

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            options={
                "maxiter": 200,
                "ftol": 1e-8,
                "disp": False,
            },
        )

        x_star = max(
            float(result.x[0]),
            0.0,
        )

        total_cost = float(result.fun)

        # ----------------------------------------------------
        # COST BREAKDOWN
        # ----------------------------------------------------

        holding_cost = (
            self.h_rate * unit_cost * x_star * (1.0 + self._vintage_penalty(age))
        )

        obs_prob = self._obsolescence_probability(
            age,
            geo_risk,
        )

        replacement_cost = unit_cost * (1.0 + self._scarcity_factor(age))

        obsolescence_cost = (
            obs_prob
            * replacement_cost
            * min(
                x_star / max(q_opt, 1.0),
                1.5,
            )
        )

        diff = q_opt - x_star

        unmet = self._softplus(diff)

        shortage_penalty = unmet * replacement_cost * (2.0 + geo_risk + obs_prob)

        readiness_penalty = unmet**1.2 * unit_cost * 0.5

        # ----------------------------------------------------
        # KKT
        # ----------------------------------------------------

        kkt_resid = self._kkt_stationarity(
            x_star,
            unit_cost,
            age,
            geo_risk,
            q_opt,
        )

        # ----------------------------------------------------
        # WRITE-OFF THRESHOLD
        # ----------------------------------------------------

        thresholds = np.linspace(
            0,
            20,
            200,
        )

        write_off_age = 15.0

        for t in thresholds:

            if (
                self._obsolescence_probability(
                    t,
                    geo_risk,
                )
                > 0.80
            ):

                write_off_age = float(t)
                break

        # ----------------------------------------------------
        # ACTION ENGINE
        # ----------------------------------------------------

        if obs_prob > 0.85 or age > write_off_age:

            action = "Write-Off"

        elif obs_prob > 0.60:

            action = "Reduce"

        elif x_star > q_opt * 1.3:

            action = "Prioritize"

        else:

            action = "Hold"

        return LifecycleResult(
            item_id=iid,
            optimal_holding_qty=round(
                x_star,
                4,
            ),
            optimal_total_cost=round(
                total_cost,
                2,
            ),
            holding_cost_at_opt=round(
                holding_cost,
                2,
            ),
            obsolescence_cost_at_opt=round(
                obsolescence_cost,
                2,
            ),
            shortage_penalty_at_opt=round(
                shortage_penalty,
                2,
            ),
            readiness_penalty_at_opt=round(
                readiness_penalty,
                2,
            ),
            replacement_cost_at_opt=round(
                replacement_cost,
                2,
            ),
            obsolescence_prob=round(
                obs_prob,
                4,
            ),
            kkt_stationarity=round(
                kkt_resid,
                6,
            ),
            kkt_satisfied=(kkt_resid < self.kkt_tol),
            action=action,
            age_years=round(
                age,
                2,
            ),
            write_off_threshold_yrs=round(
                write_off_age,
                2,
            ),
        )


# ============================================================
# REDUCED SMOKE TEST
# ============================================================

if __name__ == "__main__":

    print("==================================================")
    print("W26.2 DIFFERENTIABLE OPTIMIZER")
    print("==================================================")

    row = {
        "item_id": "SKU_TEST",
        "unit_cost": 1000.0,
        "q_star": 10.0,
        "dp_q_star": 12.0,
        "geo_risk_score": 0.2,
        "ci_score": 0.7,
        "tsl": 0.95,
        "item_age_years": 2.0,
    }

    optimizer = KKTLifecycleOptimizer()

    result = optimizer.optimize_sku(row)

    print(result)

    print("==================================================")
    print("SMOKE TEST COMPLETE")
    print("==================================================")
