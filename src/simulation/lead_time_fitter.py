"""
Stochastic Lead-Time Distribution Fitter — v1.2
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =========================================================
# COUNTRY CLUSTERS
# =========================================================

COUNTRY_CLUSTERS = {
    # LOW RISK
    "IN": "LOW_RISK",
    "DE": "LOW_RISK",
    "FR": "LOW_RISK",
    "US": "LOW_RISK",
    "IL": "LOW_RISK",
    "GB": "LOW_RISK",
    "JP": "LOW_RISK",
    # MEDIUM RISK
    "CN": "MEDIUM_RISK",
    "TW": "MEDIUM_RISK",
    "KR": "MEDIUM_RISK",
    "AE": "MEDIUM_RISK",
    "SA": "MEDIUM_RISK",
    # HIGH RISK
    "RU": "HIGH_RISK",
    "BY": "HIGH_RISK",
    "IR": "HIGH_RISK",
    "PK": "HIGH_RISK",
    # CONFLICT
    "UA": "CONFLICT",
    "SY": "CONFLICT",
    "YE": "CONFLICT",
    "LY": "CONFLICT",
}


# =========================================================
# GAMMA DISTRIBUTIONS
# =========================================================

CLUSTER_GAMMA_PARAMS = {
    "LOW_RISK": (2.0, 2.0),
    "MEDIUM_RISK": (3.0, 8.0),
    "HIGH_RISK": (2.0, 30.0),
    "CONFLICT": (1.5, 80.0),
    "UNKNOWN": (2.5, 10.0),
}


# =========================================================
# SCENARIO MULTIPLIERS
# =========================================================

SCENARIO_LT_MULTIPLIERS = {
    "baseline": {},
    "sanctions": {
        "HIGH_RISK": 4.0,
        "CONFLICT": 5.0,
        "MEDIUM_RISK": 1.5,
    },
    "conflict": {
        "HIGH_RISK": 5.0,
        "CONFLICT": 8.0,
        "MEDIUM_RISK": 2.0,
    },
    "pandemic": {
        "LOW_RISK": 2.0,
        "MEDIUM_RISK": 2.5,
        "HIGH_RISK": 3.0,
        "CONFLICT": 3.5,
        "UNKNOWN": 2.0,
    },
    "port_closure": {
        "MEDIUM_RISK": 3.0,
        "HIGH_RISK": 3.5,
    },
    "logistics_collapse": {
        "LOW_RISK": 2.0,
        "MEDIUM_RISK": 2.0,
        "HIGH_RISK": 2.5,
        "CONFLICT": 3.0,
        "UNKNOWN": 2.0,
    },
}


# =========================================================
# DISTRIBUTION OBJECT
# =========================================================


@dataclass
class LeadTimeDistribution:

    country: str
    cluster: str

    alpha: float
    beta: float

    mean_lt: float
    std_lt: float

    scenario_mult: float = 1.0


# =========================================================
# MAIN FITTER
# =========================================================


class StochasticLeadTimeFitter:

    def __init__(
        self,
        penalty_factor: float = 0.50,
        scenario: str = "baseline",
    ):

        self.penalty_factor = penalty_factor
        self.scenario = scenario

        self._distributions = {}

        self._build_distributions()

    # -----------------------------------------------------

    def _build_distributions(self):

        mults = SCENARIO_LT_MULTIPLIERS.get(
            self.scenario,
            {},
        )

        for cluster, (
            alpha,
            beta,
        ) in CLUSTER_GAMMA_PARAMS.items():

            mult = mults.get(
                cluster,
                1.0,
            )

            self._distributions[cluster] = LeadTimeDistribution(
                country=cluster,
                cluster=cluster,
                alpha=alpha,
                beta=beta * mult,
                mean_lt=alpha * beta * mult,
                std_lt=np.sqrt(alpha) * beta * mult,
                scenario_mult=mult,
            )

    # -----------------------------------------------------

    def get_distribution(
        self,
        country: str,
    ):

        cluster = COUNTRY_CLUSTERS.get(
            country.upper(),
            "UNKNOWN",
        )

        return self._distributions[cluster]

    # -----------------------------------------------------

    def sample_lead_time(
        self,
        country: str,
        geo_risk: float,
        baseline_lt: float,
        rng: np.random.Generator,
    ) -> float:

        dist = self.get_distribution(country)

        baseline = rng.gamma(
            shape=max(
                dist.alpha,
                0.1,
            ),
            scale=max(
                dist.beta,
                0.1,
            ),
        )

        delta_geo = geo_risk * dist.mean_lt * self.penalty_factor

        total_lt = baseline + delta_geo

        return max(
            total_lt,
            1.0,
        )

    # -----------------------------------------------------

    def sample_batch(
        self,
        country: str,
        geo_risk: float,
        baseline_lt: float,
        n: int,
        rng: np.random.Generator,
    ):

        return np.array(
            [
                self.sample_lead_time(
                    country,
                    geo_risk,
                    baseline_lt,
                    rng,
                )
                for _ in range(n)
            ]
        )

    # -----------------------------------------------------

    def fit_to_observed(
        self,
        observed_lts,
        country: str,
    ):

        observed_lts = np.asarray(
            observed_lts,
            dtype=float,
        )

        observed_lts = observed_lts[observed_lts > 0]

        if len(observed_lts) < 5:

            logger.warning("Too few observations")

            return self.get_distribution(country)

        alpha, loc, beta = stats.gamma.fit(
            observed_lts,
            floc=0,
        )

        cluster = COUNTRY_CLUSTERS.get(
            country.upper(),
            "UNKNOWN",
        )

        fitted = LeadTimeDistribution(
            country=country,
            cluster=cluster,
            alpha=alpha,
            beta=beta,
            mean_lt=alpha * beta,
            std_lt=np.sqrt(alpha) * beta,
        )

        self._distributions[cluster] = fitted

        return fitted

    # -----------------------------------------------------

    def generate_overrides(
        self,
        sku_df: pd.DataFrame,
        rng_seed: int = 0,
    ):

        mu_overrides = {}
        sig_overrides = {}

        for _, row in sku_df.iterrows():

            iid = str(row["item_id"])

            country = str(
                row.get(
                    "supply_origin_country",
                    "UNKNOWN",
                )
            )

            geo_risk = float(
                row.get(
                    "geo_risk_score",
                    0.0,
                )
            )

            dist = self.get_distribution(country)

            mu = dist.mean_lt + geo_risk * dist.mean_lt * self.penalty_factor

            mu_overrides[iid] = mu

            sig_overrides[iid] = dist.std_lt

        return (
            mu_overrides,
            sig_overrides,
        )

    # -----------------------------------------------------

    def distribution_summary(
        self,
    ):

        rows = []

        for cluster, d in self._distributions.items():

            rows.append(
                {
                    "cluster": cluster,
                    "alpha": round(
                        d.alpha,
                        3,
                    ),
                    "beta": round(
                        d.beta,
                        3,
                    ),
                    "mean_lt_days": round(
                        d.mean_lt,
                        1,
                    ),
                    "std_lt_days": round(
                        d.std_lt,
                        1,
                    ),
                    "scenario_mult": d.scenario_mult,
                    "scenario": self.scenario,
                }
            )

        return pd.DataFrame(rows).sort_values(
            "mean_lt_days",
            ascending=False,
        )
