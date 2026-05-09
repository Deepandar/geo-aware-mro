# src/classifiers/newsvendor.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml

from scipy import stats


logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# Load Config
# -------------------------------------------------------------

def load_config():

    with open(
        "config/tsl_config.yaml",
        "r",
    ) as f:

        return yaml.safe_load(f)[
            "tsl_config"
        ]


# -------------------------------------------------------------
# Newsvendor Engine
# -------------------------------------------------------------

class NewsvendorEngine:

    """
    Risk-aware Newsvendor Engine (v1.4)

    Features:
        - stochastic demand fitting
        - robust z-score handling
        - Black Swan stability
        - service-level sanitation
        - lead-time clipping
        - NaN-safe ROP computation
    """

    def __init__(self):

        cfg = load_config()

        self.tsl_map = cfg[
            "tsl_map"
        ]

        self.fns_mod = cfg[
            "fns_modulation"
        ]

        logger.info(
            "NewsvendorEngine initialised"
        )

    # ---------------------------------------------------------
    # Resolve TSL
    # ---------------------------------------------------------

    def resolve_tsl(
        self,
        abc,
        ved,
        fns,
    ):

        low, high = self.tsl_map.get(
            f"{abc}_{ved}",
            [0.80, 0.85],
        )

        mod = self.fns_mod.get(
            fns,
            0.50,
        )

        tsl = (
            low
            + mod * (high - low)
        )

        return float(
            np.clip(
                tsl,
                0.80,
                0.999,
            )
        )

    # ---------------------------------------------------------
    # Distribution fitting
    # ---------------------------------------------------------

    def _fit_dist(
        self,
        mean,
        std,
        fns,
    ):

        mean = max(
            float(mean),
            0.01,
        )

        std = max(
            float(std),
            0.01,
        )

        # -----------------------------------------------------
        # Fast movers
        # -----------------------------------------------------

        if fns in [
            "Fast",
            "F",
        ]:

            return (
                stats.poisson(
                    mu=mean
                ),
                "poisson",
            )

        # -----------------------------------------------------
        # Intermittent / erratic
        # -----------------------------------------------------

        elif fns in [
            "Normal",
            "Slow",
            "N",
            "S",
            "Erratic",
            "Lumpy",
            "Intermittent",
        ]:

            var = std ** 2

            if var > mean:

                r = max(
                    mean ** 2
                    / max(
                        (
                            var
                            - mean
                        ),
                        1e-6,
                    ),
                    1e-3,
                )

                p = r / (r + mean)

                return (
                    stats.nbinom(
                        r,
                        p,
                    ),
                    "negbin",
                )

        # -----------------------------------------------------
        # Default normal
        # -----------------------------------------------------

        return (
            stats.norm(
                loc=mean,
                scale=max(
                    std,
                    1e-6,
                ),
            ),
            "normal",
        )

    # ---------------------------------------------------------
    # Main compute
    # ---------------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        # -----------------------------------------------------
        # Service level targets
        # -----------------------------------------------------

        df["tsl"] = df.apply(
            lambda r: self.resolve_tsl(
                r["abc_class"],
                r["ved_class"],
                r["fns_class"],
            ),
            axis=1,
        )

        df["critical_ratio"] = (
            df["tsl"]
        )

        # -----------------------------------------------------
        # Demand moments
        # -----------------------------------------------------

        if "mean_demand" not in df.columns:

            df["mean_demand"] = (
                df["demand"]
            )

        if "std_demand" not in df.columns:

            df["std_demand"] = (
                df["demand"]
                * 0.30
            )

        # -----------------------------------------------------
        # Safe clipping
        # -----------------------------------------------------

        df["mean_demand"] = np.clip(
            df["mean_demand"]
            .astype(float),
            0.01,
            None,
        )

        df["std_demand"] = np.clip(
            df["std_demand"]
            .astype(float),
            0.01,
            None,
        )

        df["lead_time_days"] = np.clip(
            df["lead_time_days"]
            .astype(float),
            1.0,
            365.0,
        )

        # -----------------------------------------------------
        # Compute Q* and ROP
        # -----------------------------------------------------

        q_list = []

        rop_list = []

        for _, r in df.iterrows():

            mean = float(
                r["mean_demand"]
            )

            std = float(
                r["std_demand"]
            )

            lead_time = float(
                r["lead_time_days"]
            )

            cr = float(
                np.clip(
                    r["critical_ratio"],
                    0.80,
                    0.999,
                )
            )

            # -------------------------------------------------
            # Distribution fit
            # -------------------------------------------------

            dist, dist_name = (
                self._fit_dist(
                    mean,
                    std,
                    r["fns_class"],
                )
            )

            # -------------------------------------------------
            # Q*
            # -------------------------------------------------

            try:

                q = float(
                    dist.ppf(cr)
                )

            except Exception:

                logger.warning(
                    (
                        "Distribution failure | "
                        "fallback mean demand"
                    )
                )

                q = mean

            q = np.nan_to_num(
                q,
                nan=mean,
                posinf=mean * 3,
                neginf=mean,
            )

            q = max(
                q,
                1.0,
            )

            # -------------------------------------------------
            # Robust z-score
            # -------------------------------------------------

            try:

                z = float(
                    stats.norm.ppf(cr)
                )

            except Exception:

                z = 1.28

            z = np.nan_to_num(
                z,
                nan=1.28,
                posinf=3.0,
                neginf=1.28,
            )

            # -------------------------------------------------
            # Robust ROP
            # -------------------------------------------------

            rop = (
                (
                    mean
                    * lead_time
                )
                + (
                    z
                    * std
                    * np.sqrt(
                        lead_time
                    )
                )
            )

            rop = np.nan_to_num(
                rop,
                nan=1.0,
                posinf=999999.0,
                neginf=1.0,
            )

            rop = max(
                float(rop),
                1.0,
            )

            q_list.append(q)

            rop_list.append(rop)

        # -----------------------------------------------------
        # Final outputs
        # -----------------------------------------------------

        df["q_star"] = np.clip(
            np.array(q_list),
            1.0,
            None,
        )

        df["rop"] = np.clip(
            np.array(rop_list),
            1.0,
            None,
        )

        # -----------------------------------------------------
        # Diagnostics
        # -----------------------------------------------------

        logger.info(
            (
                "Newsvendor complete | "
                "mean_q=%.2f | "
                "mean_rop=%.2f | "
                "mean_tsl=%.3f"
            ),
            float(
                df["q_star"].mean()
            ),
            float(
                df["rop"].mean()
            ),
            float(
                df["tsl"].mean()
            ),
        )

        return df