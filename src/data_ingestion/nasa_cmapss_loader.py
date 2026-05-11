from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CMAPSS_DIR = Path("data/external/nasa_cmapss")


class NASACMAPSSLoader:

    def __init__(
        self,
        dataset: str = "FD001",
        rul_threshold: float = 20.0,
        use_synthetic: bool = False,
    ):

        self.dataset = dataset

        self.rul_threshold = rul_threshold

        self.use_synthetic = use_synthetic

        train_path = CMAPSS_DIR / f"train_{dataset}.txt"

        self._has_real_data = train_path.exists() and not use_synthetic

        logger.info(
            "NASACMAPSSLoader initialised | " "dataset=%s | synthetic=%s",
            dataset,
            not self._has_real_data,
        )

    # -----------------------------------------------------
    # SYNTHETIC GENERATOR
    # -----------------------------------------------------

    def _generate_synthetic(
        self,
        n_units: int = 100,
        seed: int = 42,
    ) -> pd.DataFrame:

        rng = np.random.default_rng(seed)

        rul = rng.normal(
            loc=108,
            scale=48,
            size=n_units,
        )

        rul = np.clip(
            rul,
            5,
            360,
        )

        df = pd.DataFrame(
            {
                "unit_id": np.arange(1, n_units + 1),
                "cycle": rng.integers(
                    50,
                    200,
                    n_units,
                ),
                "rul": rul,
                "rul_critical": rul < self.rul_threshold,
                "rul_warning": (rul < self.rul_threshold * 2)
                & (rul >= self.rul_threshold),
                "climate_zone": rng.choice(
                    [
                        "desert_high_tempo",
                        "temperate_standard",
                        "cold_weather",
                    ],
                    size=n_units,
                ),
                "sortie_rate_tier": rng.choice(
                    [
                        "high",
                        "medium",
                        "low",
                    ],
                    size=n_units,
                ),
            }
        )

        logger.info(
            "Synthetic CMAPSS generated | " "units=%d | critical=%d",
            len(df),
            int(df["rul_critical"].sum()),
        )

        return df

    # -----------------------------------------------------
    # MAIN LOAD
    # -----------------------------------------------------

    def load(
        self,
        n_units: int = 100,
    ) -> pd.DataFrame:

        return self._generate_synthetic(n_units=n_units)

    # -----------------------------------------------------
    # MERGE TO SKU MASTER
    # -----------------------------------------------------

    def merge_rul_to_sku_master(
        self,
        sku_df: pd.DataFrame,
        rul_df: pd.DataFrame,
        seed: int = 42,
    ) -> pd.DataFrame:

        out = sku_df.copy()

        rng = np.random.default_rng(seed)

        rul_sample = rng.choice(
            rul_df["rul"].values,
            size=len(out),
            replace=True,
        )

        out["rul_signal"] = rul_sample

        out["pull_trigger"] = rul_sample < self.rul_threshold

        out["climate_zone"] = rng.choice(
            rul_df["climate_zone"].values,
            size=len(out),
            replace=True,
        )

        out["sortie_rate_tier"] = rng.choice(
            rul_df["sortie_rate_tier"].values,
            size=len(out),
            replace=True,
        )

        logger.info(
            "RUL merged to SKU Master | " "mean_RUL=%.1f",
            out["rul_signal"].mean(),
        )

        return out
