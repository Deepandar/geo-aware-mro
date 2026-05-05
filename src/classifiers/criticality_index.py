# src/classifiers/criticality_index.py

import pandas as pd
import yaml


def load_config():
    with open("config/criticality_config.yaml", "r") as f:
        return yaml.safe_load(f)["criticality_index"]


class CriticalityIndexer:

    def __init__(self):
        cfg = load_config()

        self.weights = cfg["weights"]

        self.abc_map = {"A": 1.0, "B": 0.5, "C": 0.0}
        self.ved_map = {"V": 1.0, "E": 0.5, "D": 0.0}
        self.fns_map = {"F": 1.0, "N": 0.5, "S": 0.0}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        required = [
            "abc_class",
            "ved_class",
            "fns_class",
            "location_score_adj",
            "ltr_score"
        ]

        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.copy()

        # encode
        df["abc_score"] = df["abc_class"].map(self.abc_map)
        df["ved_score"] = df["ved_class"].map(self.ved_map)
        df["fns_score"] = df["fns_class"].map(self.fns_map)

        # 5D Criticality Index
        df["ci_score"] = (
            self.weights["w1_abc"] * df["abc_score"] +
            self.weights["w2_ved"] * df["ved_score"] +
            self.weights["w3_fns"] * df["fns_score"] +
            self.weights["w4_loc"] * df["location_score_adj"] +
            self.weights["w5_ltr"] * df["ltr_score"]
        )

        df["ci_score"] = df["ci_score"].clip(0.0, 1.0)

        # tiering
        def assign_tier(x):
            if x >= 0.66:
                return "High"
            elif x >= 0.33:
                return "Medium"
            return "Low"

        df["ci_tier"] = df["ci_score"].apply(assign_tier)

        return df
