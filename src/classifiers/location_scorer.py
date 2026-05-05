# src/classifiers/location_scorer.py

import pandas as pd
import numpy as np


class LocationScorer:

    def __init__(self):
        self.base_scores = {
            "Forward": 3,
            "Border": 2,
            "Rear": 1
        }
        self.norm_min = 1.0
        self.norm_max = 3.0  # v1.1 (multiplier = 1.0)

    def _get_base_score(self, tier: str) -> float:
        return float(self.base_scores.get(tier, 1.0))

    def _normalize(self, val: float) -> float:
        return (val - self.norm_min) / (self.norm_max - self.norm_min)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        if "depot_tier" not in df.columns:
            raise ValueError("Missing required column: depot_tier")

        df = df.copy()

        # default multiplier (v1.1)
        if "environment_multiplier" not in df.columns:
            df["environment_multiplier"] = 1.0

        df["base_position_score"] = df["depot_tier"].apply(self._get_base_score)

        raw = df["base_position_score"] * df["environment_multiplier"]

        df["location_score_adj"] = raw.apply(self._normalize).clip(0, 1)

        return df
