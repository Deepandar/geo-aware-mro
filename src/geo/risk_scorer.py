from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Bayesian country priors
# Beta(alpha, beta)
# ---------------------------------------------------------------------

COUNTRY_PRIORS: dict[str, tuple[float, float]] = {
    "RU": (8.0, 2.0),
    "CN": (5.0, 5.0),
    "UA": (9.0, 1.0),
    "IR": (7.0, 3.0),
    "KP": (9.5, 0.5),
    "BY": (7.0, 3.0),
    "PK": (4.0, 6.0),
    "IN": (2.0, 8.0),
    "US": (1.0, 9.0),
    "DE": (0.5, 9.5),
    "FR": (0.5, 9.5),
    "IL": (3.0, 7.0),
    "AE": (2.0, 8.0),
    "DEFAULT": (3.0, 7.0),
}


class BayesianRiskScorer:

    def __init__(self, config_path: str | Path = "config/criticality_config.yaml"):

        with open(config_path) as f:
            cfg = yaml.safe_load(f)["criticality_index"]

        ltr_cfg = cfg.get("ltr", {})

        self.use_geo_risk = ltr_cfg.get("use_geo_risk", True)

        logger.info(
            "BayesianRiskScorer initialised | use_geo_risk=%s", self.use_geo_risk
        )

    # -----------------------------------------------------------------
    # Main scoring
    # -----------------------------------------------------------------

    def score(self, sku_df: pd.DataFrame) -> pd.DataFrame:

        self._validate(sku_df)

        df = sku_df.copy()

        if not self.use_geo_risk:
            df["geo_risk_score"] = 0.0
            return df

        scores = []

        for _, row in df.iterrows():

            country = str(row.get("supply_origin_country", "DEFAULT")).upper()

            hhi = float(row.get("hhi_score", 0.3))

            posterior = self._compute_posterior(
                country=country,
                hhi=hhi,
            )

            scores.append(posterior)

        df["geo_risk_score"] = np.clip(scores, 0.0, 1.0)

        assert df["geo_risk_score"].between(0, 1).all()

        top_origin = (
            df.groupby("supply_origin_country")["geo_risk_score"].mean().idxmax()
            if not df.empty
            else "N/A"
        )

        logger.info(
            "Geo risk scoring complete | mean=%.3f | std=%.3f | top_origin=%s",
            df["geo_risk_score"].mean(),
            df["geo_risk_score"].std(),
            top_origin,
        )

        return df

    # -----------------------------------------------------------------
    # Bayesian posterior
    # -----------------------------------------------------------------

    def _compute_posterior(
        self,
        country: str,
        hhi: float,
    ) -> float:

        alpha_0, beta_0 = COUNTRY_PRIORS.get(country, COUNTRY_PRIORS["DEFAULT"])

        hhi_signal = float(np.clip(hhi, 0.0, 1.0))

        pseudo_obs = 5.0

        alpha_post = alpha_0 + pseudo_obs * hhi_signal
        beta_post = beta_0 + pseudo_obs * (1.0 - hhi_signal)

        posterior = alpha_post / (alpha_post + beta_post)

        return float(posterior)

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:

        required = {
            "supply_origin_country",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(f"BayesianRiskScorer missing columns: {missing}")

    # -----------------------------------------------------------------
    # MLflow logging
    # -----------------------------------------------------------------

    def log_to_mlflow(
        self,
        df: pd.DataFrame,
        run_name: str = "bayesian_risk_v1.2",
    ) -> None:

        with mlflow.start_run(
            run_name=run_name,
            nested=True,
        ):

            mlflow.log_param(
                "use_geo_risk",
                self.use_geo_risk,
            )

            mlflow.log_metric("mean_geo_risk", float(df["geo_risk_score"].mean()))

            mlflow.log_metric("std_geo_risk", float(df["geo_risk_score"].std()))

            mlflow.log_metric(
                "high_risk_count", int((df["geo_risk_score"] > 0.7).sum())
            )
