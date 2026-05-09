
from src.suppliers.decision_tree_qualifier import (
    DecisionTreeQualifier,
)

from src.suppliers.nash_equilibrium import (
    NashEquilibriumEngine,
)

# src/pipelines/sku_pipeline.py

from pathlib import Path
import json
import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


from src.data_ingestion.nasa_cmapss_loader import (
    NASACMAPSSLoader,
)
import seaborn as sns

from src.pdm.rul_engine import (
    RULEngine,
)

from src.geo.risk_scorer import (
    BayesianRiskScorer,
)

from src.risk.scenario_manager import (
    ScenarioManager,
)

from src.risk.resilience_engine import (
    ResilienceEngine,
)

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

from src.classifiers.ved_classifier import (
    classify_ved,
)

from src.data_ingestion.synthetic_sku_master import (
    generate_sku_master,
)

from src.optimization.bellman_engine import (
    BellmanEngine,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


OUTPUT_PATH = Path(
    "data/processed/sku_master_v1.3.parquet"
)

METRICS_PATH = Path(
    "data/processed/pipeline_metrics.json"
)

FIGURES_DIR = Path(
    "reports/figures"
)


def run_pipeline(
    n_skus: int = 500,
    sim_time: int = 30,
) -> pd.DataFrame:

    # ---------------------------------------------------------
    # MLflow local filesystem backend
    # ---------------------------------------------------------

    mlruns_dir = Path("mlruns")

    mlruns_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        "file:./mlruns"
    )

    mlflow.set_experiment(
        "geo-aware-mro"
    )

    with mlflow.start_run(
        run_name="v1.3_pipeline"
    ):

        mlflow.log_param(
            "n_skus",
            n_skus,
        )

        mlflow.log_param(
            "pipeline_version",
            "v1.3",
        )

        mlflow.log_param(
            "sim_time",
            sim_time,
        )

        # ---------------------------------------------------------
        # Stage 1
        # ---------------------------------------------------------

        logger.info(
            "Stage 1/13 â€” Generate SKU Master"
        )

        df = generate_sku_master(
            n_skus=n_skus,
        )

        mlflow.log_metric(
            "sku_count",
            len(df),
        )

        # ---------------------------------------------------------
        # Stage 2
        # ---------------------------------------------------------

        logger.info(
            "Stage 2/13 â€” Dominance Check"
        )

        df["annual_consumption_value"] = (
            df["demand"]
            * df["unit_cost"]
        )

        dominance = DominanceChecker()

        df, dominance_result = (
            dominance.check_and_remediate(
                df,
                acv_col=(
                    "annual_consumption_value"
                ),
            )
        )

        mlflow.log_metric(
            "concentration_ratio",
            dominance_result[
                "concentration_ratio"
            ],
        )

        # ---------------------------------------------------------
        # Stage 3
        # ---------------------------------------------------------

        logger.info(
            "Stage 3/13 â€” ABC Classification"
        )

        df = classify_abc(df)

        # ---------------------------------------------------------
        # Stage 4
        # ---------------------------------------------------------

        logger.info(
            "Stage 4/13 â€” VED Classification"
        )

        df = classify_ved(df)

        # ---------------------------------------------------------
        # Stage 5
        # ---------------------------------------------------------

        logger.info(
            "Stage 5/13 â€” FNS Classification"
        )

        df = classify_fns(df)

        # ---------------------------------------------------------
        # Stage 6
        # ---------------------------------------------------------

        logger.info(
            "Stage 6/13 â€” Location Scoring"
        )

        scorer = LocationScorer()

        df = scorer.score(df)

        # ---------------------------------------------------------
        # Stage 7
        # ---------------------------------------------------------

        logger.info(
            "Stage 7/13 â€” Bayesian Geo-Risk"
        )

        risk_scorer = BayesianRiskScorer(
            config_path=(
                "config/criticality_config.yaml"
            )
        )

        df = risk_scorer.score(df)

        # ---------------------------------------------------------
        # Stage 8
        # ---------------------------------------------------------

        logger.info(
            "Stage 8/13 â€” Scenario Injection"
        )

        scenario_mgr = ScenarioManager()

        df = scenario_mgr.inject(
            df,
            sim_time=sim_time,
        )

        mlflow.log_metric(
            "scenario_active_count",
            int(
                df["scenario_active"].sum()
            ),
        )

        # ---------------------------------------------------------
        # Stage 9
        # ---------------------------------------------------------

        logger.info(
            "Stage 9/13 â€” Resilience Decay"
        )

        resilience = ResilienceEngine()

        df = resilience.apply_decay(
            df,
            sim_time=sim_time,
        )

        mlflow.log_metric(
            "active_disruptions",
            int(
                df["scenario_active"].sum()
            ),
        )

        # ---------------------------------------------------------
        # Stage 10
        # ---------------------------------------------------------

        logger.info(
            "Stage 10/13 â€” Compound LTR"
        )

        ltr = LTRScorer()

        df = ltr.compute(df)

        mlflow.log_metric(
            "mean_ltr_score",
            float(
                df["ltr_score"].mean()
            ),
        )

        # ---------------------------------------------------------
        # Stage 11
        # ---------------------------------------------------------

        logger.info(
            "Stage 11/13 â€” Criticality Index"
        )

        ci = CriticalityIndexer()

        df = ci.compute(df)

        if "ci_band" in df.columns:

            df["ci_tier"] = (
                df["ci_band"]
            )

        mlflow.log_metric(
            "mean_ci_score",
            float(
                df["ci_score"].mean()
            ),
        )

        # ---------------------------------------------------------
        # Stage 12
        # ---------------------------------------------------------

        logger.info(
            "Stage 12/13 â€” RUL Engine"
        )

        rul_engine = RULEngine()

        df = rul_engine.compute(df)

        mlflow.log_metric(
            "imminent_failures",
            int(
                df["imminent_failure"].sum()
            ),
        )

        # ---------------------------------------------------------
        # Stage 13 â€” Bellman Optimisation
        # ---------------------------------------------------------

        logger.info(
            "Stage 13/13 â€” Bellman Optimisation"
        )

        bellman = BellmanEngine()

        df = bellman.compute(df)

        # =========================================================
        # STAGE 14 â€” SUPPLIER QUALIFICATION
        # =========================================================

        logger.info(
            "Stage 14/15 â€” Supplier Qualification"
        )

        supplier_engine = DecisionTreeQualifier(
            max_depth=5,
            random_state=42,
        )

        df = supplier_engine.fit_predict(
            df
        )[0]

        logger.info(
            (
                "Supplier qualification complete | "
                "critical=%d"
            ),
            int(
                (
                    df[
                        "supplier_risk_class"
                    ]
                    == "Critical"
                ).sum()
            ),
        )

        # =========================================================
        # STAGE 15 â€” NASH EQUILIBRIUM
        # =========================================================

        logger.info(
            "Stage 15/15 â€” Nash Equilibrium"
        )

        nash_engine = NashEquilibriumEngine()

        df = nash_engine.compute(df)

        logger.info(
            (
                "Nash equilibrium complete | "
                "strategic=%d"
            ),
            int(
                (
                    df[
                        "supplier_strategy"
                    ]
                    == "Strategic"
                ).sum()
            ),
        )

        # ---------------------------------------------------------
        # Backward compatibility aliases
        # ---------------------------------------------------------

        df["q_star"] = (
            df["bellman_q_star"]
        )

        df["rop"] = (
            df["bellman_rop"]
        )

        # ---------------------------------------------------------
        # Dynamic service level approximation
        # ---------------------------------------------------------

        df["tsl"] = np.clip(
            0.85
            + (
                df["ci_score"] * 0.15
            ),
            0.85,
            0.995,
        )

        # ---------------------------------------------------------
        # ROP sanitation
        # ---------------------------------------------------------

        df["rop"] = np.nan_to_num(
            df["rop"],
            nan=1.0,
            posinf=999999.0,
            neginf=1.0,
        )

        df["rop"] = np.clip(
            df["rop"],
            1.0,
            None,
        )

        # ---------------------------------------------------------
        # Validation
        # ---------------------------------------------------------

        required_outputs = [
            "bellman_q_star",
            "bellman_rop",
            "expected_future_cost",
            "state_value",
            "ci_tier",
        ]

        for col in required_outputs:

            if col not in df.columns:

                raise ValueError(
                    (
                        f"Pipeline missing "
                        f"required column: {col}"
                    )
                )

        # ---------------------------------------------------------
        # MLflow metrics
        # ---------------------------------------------------------

        mlflow.log_metric(
            "mean_tsl",
            float(
                df["tsl"].mean()
            ),
        )

        mlflow.log_metric(
            "mean_q_star",
            float(
                df["q_star"].mean()
            ),
        )

        mlflow.log_metric(
            "mean_rop",
            float(
                df["rop"].mean()
            ),
        )

        # ---------------------------------------------------------
        # Save outputs
        # ---------------------------------------------------------

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

            "rows":
                len(df),

            "mean_geo_risk":
                float(
                    df[
                        "geo_risk_score"
                    ].mean()
                ),

            "mean_ltr_score":
                float(
                    df[
                        "ltr_score"
                    ].mean()
                ),

            "mean_ci_score":
                float(
                    df[
                        "ci_score"
                    ].mean()
                ),

            "mean_q_star":
                float(
                    df[
                        "q_star"
                    ].mean()
                ),

            "n_critical_suppliers":
                int(
                    (
                        df[
                            "supplier_risk_class"
                        ]
                        == "Critical"
                    ).sum()
                ),

            "mean_strategic_risk":
                float(
                    df[
                        "strategic_risk_score"
                    ].mean()
                ),

            "active_disruptions":
                int(
                    df[
                        "scenario_active"
                    ].sum()
                ),

            "mean_resilience_multiplier":
                float(
                    df.get(
                        "resilience_multiplier",
                        pd.Series([1.0])
                    ).mean()
                ),
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


        # =========================================================
        # STAGE 16 - NASA CMAPSS
        # =========================================================

        logger.info(
            "Stage 16/18 - NASA CMAPSS"
        )

        cmapss_loader = NASACMAPSSLoader(
            rul_threshold=20.0,
            use_synthetic=True,
        )

        rul_df = cmapss_loader.load(
            n_units=100
        )

        df = (
            cmapss_loader
            .merge_rul_to_sku_master(
                df,
                rul_df,
            )
        )

        logger.info(
            "CMAPSS complete | mean_RUL=%.2f",
            df["rul_signal"].mean(),
        )

        # =========================================================
        # STAGE 17 - PUSH/PULL
        # =========================================================

        logger.info(
            "Stage 17/18 - Push/Pull"
        )

        pp_engine = PushPullEngine(

            push_density_threshold=0.50,

            pull_rul_threshold=20.0,

            push_weight=0.60,
        )

        df = pp_engine.compute(df)

        mode_dist = (
            df["decoupling_mode"]
            .value_counts()
        )

        logger.info(
            "Push/Pull complete | %s",
            dict(mode_dist),
        )

        # =========================================================
        # STAGE 18 - REPEATED GAME
        # =========================================================

        logger.info(
            "Stage 18/18 - Repeated Game"
        )

        rg_model = RepeatedGameModel(

            T=24,

            discount_factor=0.92,

            late_threshold_days=7.0,

            cooperation_surplus=100.0,

            defection_gain=20.0,

            grim_trigger_threshold=1,
        )

        df, rep_matrix = rg_model.score(df)

        rep_path = Path(
            "data/processed/reputation_matrix.parquet"
        )

        rep_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        rep_matrix.to_parquet(
            rep_path,
            index=False,
        )

        logger.info(
            "Repeated game complete | mean_rep=%.3f",
            df["reputation_score"].mean(),
        )

        # =========================================================
        # MODEL REGISTRY v2
        # =========================================================

        registry = register_all_v1_2_models(

            dt_qualifier=dt_qualifier,

            nash_model=nash_model,

            repeated_game=rg_model,

            bellman_engine=bellman,

            df=df,
        )

        logger.info(
            "Registry complete | models=%d",
            len(
                registry.get(
                    "models",
                    {}
                )
            ),
        )

        metrics.update({

            "mean_rul":
                round(
                    df["rul_signal"].mean(),
                    2,
                ),

            "mean_reputation":
                round(
                    df[
                        "reputation_score"
                    ].mean(),
                    4,
                ),

            "triggers_fired":
                int(
                    df[
                        "grim_trigger_fired"
                    ].sum()
                ),
        })


        # ---------------------------------------------------------
        # Figures
        # ---------------------------------------------------------

        plt.figure(figsize=(8, 5))

        sns.histplot(
            df["ci_score"],
            bins=20,
        )

        plt.title(
            "Criticality Distribution"
        )

        ci_hist_path = (
            FIGURES_DIR
            / "ci_histogram.png"
        )

        plt.savefig(
            ci_hist_path,
            bbox_inches="tight",
        )

        plt.close()

        # ---------------------------------------------------------
        # MLflow artifacts
        # ---------------------------------------------------------

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(
            str(OUTPUT_PATH)
        )

        mlflow.log_artifact(
            str(METRICS_PATH)
        )

        mlflow.log_artifact(
            str(ci_hist_path)
        )

        logger.info(
            (
                "Pipeline complete | "
                "output=%s"
            ),
            OUTPUT_PATH,
        )

        return df


if __name__ == "__main__":

    result = run_pipeline()

    print(
        result[
            [
                "supply_origin_country",
                "geo_risk_score",
                "ltr_score",
                "ci_score",
                "bellman_q_star",
                "bellman_rop",
                "expected_future_cost",
            ]
        ].head()
    )