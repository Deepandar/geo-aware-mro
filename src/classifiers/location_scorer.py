from dataclasses import dataclass

import pandas as pd


@dataclass
class LocationScorer:

    config_path: str | None = None

    forward_score: float = 3.0
    border_score: float = 2.0
    rear_score: float = 1.0

    use_env_profiles: bool = False

    REQUIRED_COLUMNS = {
        "depot_tier",
        "equipment_density_score",
    }

    def score_sku_master(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        missing = (
            self.REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "missing required columns"
            )

        if "environment_multiplier" not in df.columns:
            df["environment_multiplier"] = 1.0

        mapping = {
            "Forward": self.forward_score,
            "Border": self.border_score,
            "Rear": self.rear_score,
        }

        df["base_position_score"] = (
            df["depot_tier"]
            .map(mapping)
            .fillna(self.rear_score)
        )

        raw_score = (
            df["base_position_score"]
            * df["environment_multiplier"]
        )

        min_raw = raw_score.min()
        max_raw = raw_score.max()

        if max_raw == min_raw:
            df["location_score_adj"] = 0.5
        else:
            df["location_score_adj"] = (
                (raw_score - min_raw)
                / (max_raw - min_raw)
            )

        return df

    def score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        df = df.copy()

        if "environment_multiplier" not in df.columns:
            df["environment_multiplier"] = 1.0

        mapping = {
            "Forward": self.forward_score,
            "Border": self.border_score,
            "Rear": self.rear_score,
        }

        df["base_position_score"] = (
            df["depot_tier"]
            .map(mapping)
            .fillna(self.rear_score)
        )

        raw_score = (
            df["base_position_score"]
            * df["environment_multiplier"]
        )

        min_raw = raw_score.min()
        max_raw = raw_score.max()

        if max_raw == min_raw:
            df["location_score"] = 0.5
        else:
            df["location_score"] = (
                (raw_score - min_raw)
                / (max_raw - min_raw)
            )

        return df