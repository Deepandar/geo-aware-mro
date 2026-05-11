# src/pipelines/sku_pipeline.py

from pathlib import Path
import json
import logging

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

from src.classifiers.abc_classifier import (
    classify_abc,
)
from src.classifiers.criticality_index import (
    CriticalityIndexer,
)
from src.classifiers.dominance_check import (
    DominanceChecker,
)
from src.classifiers.fns_classifier import (
    classify_fns,
)
from src.classifiers.location_scorer import (
    LocationScorer,
)
from src.classifiers.ltr_scorer import (
    LTRScorer,
)
from src.classifiers.newsvendor import (
    NewsvendorEngine,
)
from src.classifiers.ved_classifier import (
    classify_ved,
)
from src.data_ingestion.synthetic_sku_master import (
    generate_sku_master,
)

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
)

logger = logging.getLogger(__name__)


OUTPUT_PATH = Path("data/processed/sku_master_v1.1.parquet")

METRICS_PATH = Path("data/processed/pipeline_metrics.json")

FIGURES_DIR = Path("reports/figures")


def run_pipeline(
    n_skus: int = 500,
) -> pd.DataFrame:

    mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")

    mlflow.set_experiment("geo-aware-mro")

    with mlflow.start_run(run_name="v1.1_pipeline"):

        mlflow.log_param(
            "n_skus",
            n_skus,
        )

        logger.info("Stage 1/9 — Generate SKU Master")

        df = generate_sku_master(
            n_skus=n_skus,
        )

        mlflow.log_metric(
            "sku_count",
            len(df),
        )

        logger.info("Stage 2/9 — Dominance Check")

        df["annual_consumption_value"] = df["demand"] * df["unit_cost"]

        dominance = DominanceChecker()

        df, dominance_result = dominance.check_and_remediate(
            df,
            acv_col="annual_consumption_value",
        )

        mlflow.log_metric(
            "concentration_ratio",
            dominance_result["concentration_ratio"],
        )

        mlflow.log_metric(
            "dominance_detected",
            int(dominance_result["bias_detected"]),
        )

        logger.info("Stage 3/9 — ABC Classification")

        df = classify_abc(df)

        mlflow.log_metric(
            "a_ratio",
            ((df["abc_class"] == "A").mean()),
        )

        logger.info("Stage 4/9 — VED Classification")

        df = classify_ved(df)

        mlflow.log_metric(
            "v_ratio",
            ((df["ved_class"] == "V").mean()),
        )

        logger.info("Stage 5/9 — FNS Classification")

        df = classify_fns(df)

        mlflow.log_metric(
            "fast_ratio",
            ((df["fns_class"] == "Smooth").mean()),
        )

        logger.info("Stage 6/9 — Location Scoring")

        scorer = LocationScorer()

        df = scorer.score(df)

        mlflow.log_metric(
            "mean_location_score",
            df["location_score"].mean(),
        )

        logger.info("Stage 7/9 — Lead-Time Risk")

        ltr = LTRScorer()

        df = ltr.compute(df)

        mlflow.log_metric(
            "mean_ltr_score",
            df["ltr_score"].mean(),
        )

        logger.info("Stage 8/9 — Criticality Index")

        ci = CriticalityIndexer()

        df = ci.compute(df)

        mlflow.log_metric(
            "mean_ci_score",
            df["ci_score"].mean(),
        )

        logger.info("Stage 9/9 — Newsvendor")

        engine = NewsvendorEngine()

        df = engine.compute(df)

        mlflow.log_metric(
            "mean_tsl",
            df["tsl"].mean(),
        )

        mlflow.log_metric(
            "mean_qstar",
            df["q_star"].mean(),
        )

        mlflow.log_metric(
            "mean_rop",
            df["rop"].mean(),
        )

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_parquet(
            OUTPUT_PATH,
            index=False,
        )

        metrics = {
            "rows": len(df),
            "mean_ci_score": float(df["ci_score"].mean()),
            "mean_tsl": float(df["tsl"].mean()),
            "mean_qstar": float(df["q_star"].mean()),
            "mean_rop": float(df["rop"].mean()),
        }

        with open(
            METRICS_PATH,
            "w",
        ) as f:
            json.dump(
                metrics,
                f,
                indent=2,
            )

        plt.figure(figsize=(8, 5))

        sns.histplot(
            df["ci_score"],
            bins=20,
        )

        plt.title("Criticality Index Distribution")

        ci_hist_path = FIGURES_DIR / "ci_histogram.png"

        plt.savefig(
            ci_hist_path,
            bbox_inches="tight",
        )

        plt.close()

        plt.figure(figsize=(8, 5))

        sns.scatterplot(
            data=df,
            x="ltr_score",
            y="ci_score",
            hue="abc_class",
        )

        plt.title("LTR vs Criticality")

        scatter_path = FIGURES_DIR / "ltr_vs_ci.png"

        plt.savefig(
            scatter_path,
            bbox_inches="tight",
        )

        plt.close()

        pareto_df = df.sort_values(
            by="annual_consumption_value",
            ascending=False,
        ).reset_index(drop=True)

        pareto_df["cum_pct"] = (
            pareto_df["annual_consumption_value"].cumsum()
            / pareto_df["annual_consumption_value"].sum()
        ) * 100

        plt.figure(figsize=(10, 5))

        plt.bar(
            pareto_df.index,
            pareto_df["annual_consumption_value"],
        )

        plt.plot(
            pareto_df.index,
            pareto_df["cum_pct"],
        )

        plt.title("Pareto Concentration")

        pareto_path = FIGURES_DIR / "pareto_chart.png"

        plt.savefig(
            pareto_path,
            bbox_inches="tight",
        )

        plt.close()

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(METRICS_PATH))

        mlflow.log_artifact(str(OUTPUT_PATH))

        mlflow.log_artifact(str(ci_hist_path))

        mlflow.log_artifact(str(scatter_path))

        mlflow.log_artifact(str(pareto_path))

        logger.info(
            "Pipeline complete | output=%s",
            OUTPUT_PATH,
        )

        return df


if __name__ == "__main__":

    result = run_pipeline()

    print(
        result[
            [
                "abc_class",
                "ved_class",
                "fns_class",
                "ci_score",
                "ci_tier",
                "tsl",
                "q_star",
                "rop",
            ]
        ].head()
    )
