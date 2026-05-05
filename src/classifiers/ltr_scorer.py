# src/classifiers/ltr_scorer.py

import pandas as pd
import numpy as np
import yaml


def load_config():
    with open("config/criticality_config.yaml", "r") as f:
        return yaml.safe_load(f)["criticality_index"]["ltr"]


class LTRScorer:

    def __init__(self):
        cfg = load_config()
        self.use_geo_risk = cfg["use_geo_risk"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if "lead_time_days" not in df.columns:
            raise ValueError("Missing required column: lead_time_days")

        df = df.copy()

        mu_lt = df["lead_time_days"].mean()
        if mu_lt == 0:
            raise ValueError("Mean lead_time_days is zero")

        # geo risk (v1.1 = 0)
        if self.use_geo_risk and "geo_risk_score" in df.columns:
            geo_risk = df["geo_risk_score"].fillna(0.0)
        else:
            geo_risk = 0.0

        # raw score
        raw = (df["lead_time_days"] / mu_lt) * (1 + geo_risk)

        # 🔥 FIX: dynamic normalization (not fixed bounds)
        min_val = raw.min()
        max_val = raw.max()

        if max_val == min_val:
            df["ltr_score"] = 0.0
        else:
            df["ltr_score"] = (raw - min_val) / (max_val - min_val)

        return df
