"""
Demand Arrival Process Models — v1.2
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class DemandParams:
    item_id: str
    fns_class: str
    mean_demand: float
    std_demand: float
    adi: float = 1.0
    cv_squared: float = 0.5


class DemandProcess:

    def __init__(
        self,
        params: DemandParams,
        rng: np.random.Generator,
    ):
        self.params = params
        self.rng = rng

    def sample(self) -> float:

        p = self.params

        if p.fns_class == "F":
            return float(self.rng.poisson(lam=max(p.mean_demand, 0.1)))

        if p.fns_class == "N":

            var = p.std_demand**2

            if var <= p.mean_demand:
                return float(self.rng.poisson(lam=max(p.mean_demand, 0.1)))

            r = p.mean_demand**2 / (var - p.mean_demand)

            prob = r / (r + p.mean_demand)

            return float(
                self.rng.negative_binomial(
                    n=max(r, 0.01),
                    p=min(prob, 0.999),
                )
            )

        zero_prob = max(
            0.0,
            min(
                1.0 - 1.0 / max(p.adi, 1.01),
                0.90,
            ),
        )

        if self.rng.random() < zero_prob:
            return 0.0

        return float(self.rng.poisson(lam=max(p.mean_demand, 0.1)))

    def sample_batch(
        self,
        n_periods: int,
    ) -> np.ndarray:

        return np.array([self.sample() for _ in range(n_periods)])


def build_demand_process(
    row: dict,
    rng: np.random.Generator,
) -> DemandProcess:

    params = DemandParams(
        item_id=str(row.get("item_id", "UNKNOWN")),
        fns_class=str(row.get("fns_class", "N")),
        mean_demand=float(row.get("mean_demand", 10.0)),
        std_demand=float(row.get("std_demand", 5.0)),
        adi=float(row.get("adi", 1.5)),
        cv_squared=float(row.get("cv_squared", 0.5)),
    )

    return DemandProcess(params, rng)
