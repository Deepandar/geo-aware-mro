from __future__ import annotations

import logging

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scipy.stats import (
    gamma,
    lognorm,
    weibull_min,
    norm,
    expon,
    kstest,
)

logger = logging.getLogger(__name__)

# =========================================================
# Candidate distributions
# =========================================================

CANDIDATE_DISTS = {
    "gamma": gamma,
    "lognorm": lognorm,
    "weibull_min": weibull_min,
    "norm": norm,
    "expon": expon,
}


# =========================================================
# Fit result
# =========================================================

@dataclass
class FitResult:

    dist_name: str
    params: tuple
    aic: float
    ks_stat: float
    ks_pvalue: float
    mean: float
    std: float
    cv: float


# =========================================================
# Distribution fitter
# =========================================================

class DistributionFitter:

    def __init__(
        self,
        data: np.ndarray,
    ):

        data = np.array(
            data,
            dtype=float,
        )

        data = data[
            np.isfinite(data)
        ]

        data = data[
            data > 0
        ]

        if len(data) < 5:

            raise ValueError(
                "Need at least 5 observations"
            )

        self.data = data

    # -----------------------------------------------------
    # Fit one
    # -----------------------------------------------------

    def fit_one(
        self,
        dist_name: str,
    ) -> FitResult | None:

        dist = CANDIDATE_DISTS[
            dist_name
        ]

        try:

            params = dist.fit(
                self.data,
                floc=0,
            )

            log_lik = np.sum(
                dist.logpdf(
                    self.data,
                    *params,
                )
            )

            k = len(params)

            aic = (
                2 * k
                - 2 * log_lik
            )

            ks_stat, ks_p = (
                kstest(
                    self.data,
                    dist.cdf,
                    args=params,
                )
            )

            mean = dist.mean(
                *params
            )

            std = dist.std(
                *params
            )

            cv = (
                std / mean
                if mean > 0
                else 0.0
            )

            return FitResult(
                dist_name=dist_name,
                params=params,
                aic=float(aic),
                ks_stat=float(ks_stat),
                ks_pvalue=float(ks_p),
                mean=float(mean),
                std=float(std),
                cv=float(cv),
            )

        except Exception as e:

            logger.warning(
                "Fit failed | %s | %s",
                dist_name,
                e,
            )

            return None

    # -----------------------------------------------------
    # Best fit
    # -----------------------------------------------------

    def best_fit(
        self,
    ) -> FitResult:

        results = []

        for dist_name in CANDIDATE_DISTS:

            result = self.fit_one(
                dist_name
            )

            if result is not None:

                results.append(
                    result
                )

        if not results:

            raise ValueError(
                "No valid fits"
            )

        passed = [
            r for r in results
            if r.ks_pvalue > 0.05
        ]

        if passed:

            results = passed

        return min(
            results,
            key=lambda x: x.aic,
        )


# =========================================================
# GLOBAL PORTFOLIO FIT
# =========================================================

def fit_global_distribution(
    df: pd.DataFrame,
    lt_col: str = "lead_time_days",
) -> FitResult:

    if lt_col not in df.columns:

        raise ValueError(
            f"{lt_col} missing"
        )

    fitter = DistributionFitter(
        df[lt_col].values
    )

    best = fitter.best_fit()

    logger.info(
        (
            "Global distribution fitted | "
            "dist=%s | p=%.4f"
        ),
        best.dist_name,
        best.ks_pvalue,
    )

    return best
