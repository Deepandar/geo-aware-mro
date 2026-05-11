from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# =========================================================
# Scenario Types
# =========================================================


class ScenarioType(Enum):

    GEOPOLITICAL_SHOCK = "geopolitical_shock"

    PORT_DISRUPTION = "port_disruption"

    SUPPLIER_FAILURE = "supplier_failure"

    DEMAND_SURGE = "demand_surge"

    MULTI_ECHELON_CASCADE = "multi_echelon_cascade"


# =========================================================
# Black Swan Scenario
# =========================================================


@dataclass
class BlackSwanScenario:

    name: str

    scenario_type: ScenarioType

    trigger_period: int

    duration: int

    lt_multiplier: float = 1.0

    demand_multiplier: float = 1.0

    affected_skus: list = field(default_factory=list)

    severity: float = 1.0

    decay_rate: float = 0.15


# =========================================================
# Scenario Library
# =========================================================

SCENARIO_LIBRARY = [
    BlackSwanScenario(
        name="Suez Closure",
        scenario_type=ScenarioType.PORT_DISRUPTION,
        trigger_period=10,
        duration=8,
        lt_multiplier=2.8,
        demand_multiplier=1.1,
        severity=0.9,
        decay_rate=0.12,
    ),
    BlackSwanScenario(
        name="Taiwan Semiconductor Shock",
        scenario_type=ScenarioType.GEOPOLITICAL_SHOCK,
        trigger_period=15,
        duration=12,
        lt_multiplier=1.6,
        demand_multiplier=1.4,
        severity=0.85,
        decay_rate=0.08,
    ),
    BlackSwanScenario(
        name="Single Supplier Bankruptcy",
        scenario_type=ScenarioType.SUPPLIER_FAILURE,
        trigger_period=8,
        duration=20,
        lt_multiplier=3.5,
        demand_multiplier=0.8,
        severity=1.0,
        decay_rate=0.05,
    ),
    BlackSwanScenario(
        name="Post-Pandemic Demand Surge",
        scenario_type=ScenarioType.DEMAND_SURGE,
        trigger_period=5,
        duration=10,
        lt_multiplier=1.2,
        demand_multiplier=2.2,
        severity=0.75,
        decay_rate=0.20,
    ),
    BlackSwanScenario(
        name="Multi-Tier Cascade Failure",
        scenario_type=ScenarioType.MULTI_ECHELON_CASCADE,
        trigger_period=12,
        duration=15,
        lt_multiplier=2.1,
        demand_multiplier=1.7,
        severity=0.95,
        decay_rate=0.07,
    ),
]


# =========================================================
# Correlation Propagator
# =========================================================


class CorrelationPropagator:

    def __init__(
        self,
        sku_ids: list[str],
        corr_matrix: Optional[np.ndarray] = None,
    ):

        self.sku_ids = sku_ids

        n = len(sku_ids)

        # -------------------------------------------------
        # Default positive correlation structure
        # -------------------------------------------------

        if corr_matrix is None:

            corr_matrix = 0.35 * np.ones((n, n)) + 0.65 * np.eye(n)

        self.validate_corr(corr_matrix)

        self.corr_matrix = corr_matrix

        self.L = np.linalg.cholesky(corr_matrix)

    # -----------------------------------------------------
    # Validate matrix
    # -----------------------------------------------------

    def validate_corr(
        self,
        C: np.ndarray,
    ):

        assert C.shape == (
            len(self.sku_ids),
            len(self.sku_ids),
        )

        assert np.allclose(
            C,
            C.T,
        )

        eigvals = np.linalg.eigvalsh(C)

        assert (eigvals >= -1e-8).all()

    # -----------------------------------------------------
    # Sample correlated shocks
    # -----------------------------------------------------

    def sample_correlated_shocks(
        self,
        scenario: BlackSwanScenario,
        rng: np.random.Generator,
        shock_sigma: float = 0.15,
    ) -> dict:

        n = len(self.sku_ids)

        z = rng.standard_normal(n)

        correlated_z = self.L @ z

        result = {}

        for i, sku in enumerate(self.sku_ids):

            affected = not scenario.affected_skus or sku in scenario.affected_skus

            if not affected:

                result[sku] = {
                    "lt_mult": 1.0,
                    "demand_mult": 1.0,
                }

                continue

            noise = correlated_z[i] * shock_sigma

            severity = scenario.severity * (1 + noise)

            severity = float(
                np.clip(
                    severity,
                    0.1,
                    3.0,
                )
            )

            lt_m = 1.0 + (scenario.lt_multiplier - 1.0) * severity

            dem_m = 1.0 + (scenario.demand_multiplier - 1.0) * severity

            result[sku] = {
                "lt_mult": round(
                    float(
                        np.clip(
                            lt_m,
                            0.5,
                            6.0,
                        )
                    ),
                    4,
                ),
                "demand_mult": round(
                    float(
                        np.clip(
                            dem_m,
                            0.2,
                            5.0,
                        )
                    ),
                    4,
                ),
            }

        return result


# =========================================================
# Shock Timeline
# =========================================================


class ShockTimeline:

    def __init__(
        self,
        T: int,
        sku_ids: list[str],
    ):

        self.T = T

        self.sku_ids = sku_ids

        self._lt_mult = {s: np.ones(T) for s in sku_ids}

        self._dem_mult = {s: np.ones(T) for s in sku_ids}

    # -----------------------------------------------------
    # Apply scenario
    # -----------------------------------------------------

    def apply_scenario(
        self,
        scenario: BlackSwanScenario,
        propagator: CorrelationPropagator,
        rng: np.random.Generator,
    ):

        shocks = propagator.sample_correlated_shocks(
            scenario,
            rng,
        )

        t0 = scenario.trigger_period

        dur = scenario.duration

        for t in range(
            t0,
            min(
                t0 + dur,
                self.T,
            ),
        ):

            elapsed = t - t0

            decay = np.exp(-scenario.decay_rate * elapsed)

            for sku in self.sku_ids:

                lt_peak = shocks[sku]["lt_mult"]

                dem_peak = shocks[sku]["demand_mult"]

                self._lt_mult[sku][t] = max(
                    self._lt_mult[sku][t],
                    1.0 + (lt_peak - 1.0) * decay,
                )

                self._dem_mult[sku][t] = max(
                    self._dem_mult[sku][t],
                    1.0 + (dem_peak - 1.0) * decay,
                )

    # -----------------------------------------------------
    # Get multipliers
    # -----------------------------------------------------

    def get_multipliers(
        self,
        sku_id: str,
        t: int,
    ):

        return (
            self._lt_mult[sku_id][t],
            self._dem_mult[sku_id][t],
        )

    # -----------------------------------------------------
    # Export dataframe
    # -----------------------------------------------------

    def to_dataframe(
        self,
    ) -> pd.DataFrame:

        rows = []

        for sku in self.sku_ids:

            for t in range(self.T):

                rows.append(
                    {
                        "sku_id": sku,
                        "period": t,
                        "lt_mult": round(
                            self._lt_mult[sku][t],
                            4,
                        ),
                        "demand_mult": round(
                            self._dem_mult[sku][t],
                            4,
                        ),
                    }
                )

        return pd.DataFrame(rows)
