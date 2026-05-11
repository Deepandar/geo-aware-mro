from src.suppliers.decision_tree_qualifier import (
    DecisionTreeQualifier,
)

from src.simulation.bullwhip_model import BullwhipModel

from src.suppliers.nash_equilibrium import (
    NashEquilibriumEngine,
)

from src.suppliers.repeated_game import (
    RepeatedGameModel,
)

# src/pipelines/sku_pipeline.py

from pathlib import Path
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
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
    format=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
)

logger = logging.getLogger(__name__)


OUTPUT_PATH = Path("data/processed/sku_master_v1.3.parquet")

METRICS_PATH = Path("data/processed/pipeline_metrics.json")

FIGURES_DIR = Path("reports/figures")


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

    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("geo-aware-mro")

    with mlflow.start_run(run_name="v1.3_pipeline"):

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

        logger.info("Stage 1/13 — Generate SKU Master")

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

        logger.info("Stage 2/13 — Dominance Check")

        df["annual_consumption_value"] = df["demand"] * df["unit_cost"]

        dominance = DominanceChecker()

        df, dominance_result = dominance.check_and_remediate(
            df,
            acv_col=("annual_consumption_value"),
        )

        mlflow.log_metric(
            "concentration_ratio",
            dominance_result["concentration_ratio"],
        )

        # ---------------------------------------------------------
        # Stage 3
        # ---------------------------------------------------------

        logger.info("Stage 3/13 — ABC Classification")

        df = classify_abc(df)

        # ---------------------------------------------------------
        # Stage 4
        # ---------------------------------------------------------

        logger.info("Stage 4/13 — VED Classification")

        df = classify_ved(df)

        # ---------------------------------------------------------
        # Stage 5
        # ---------------------------------------------------------

        logger.info("Stage 5/13 — FNS Classification")

        df = classify_fns(df)

        # ---------------------------------------------------------
        # Stage 6
        # ---------------------------------------------------------

        logger.info("Stage 6/13 — Location Scoring")

        scorer = LocationScorer()

        df = scorer.score(df)

        # ---------------------------------------------------------
        # Stage 7
        # ---------------------------------------------------------

        logger.info("Stage 7/13 — Bayesian Geo-Risk")

        risk_scorer = BayesianRiskScorer(config_path=("config/criticality_config.yaml"))

        df = risk_scorer.score(df)

        # ---------------------------------------------------------
        # Stage 8
        # ---------------------------------------------------------

        logger.info("Stage 8/13 — Scenario Injection")

        scenario_mgr = ScenarioManager()

        df = scenario_mgr.inject(
            df,
            sim_time=sim_time,
        )

        mlflow.log_metric(
            "scenario_active_count",
            int(df["scenario_active"].sum()),
        )

        # ---------------------------------------------------------
        # Stage 9
        # ---------------------------------------------------------

        logger.info("Stage 9/13 — Resilience Decay")

        resilience = ResilienceEngine()

        df = resilience.apply_decay(
            df,
            sim_time=sim_time,
        )

        mlflow.log_metric(
            "active_disruptions",
            int(df["scenario_active"].sum()),
        )

        # ---------------------------------------------------------
        # Stage 10
        # ---------------------------------------------------------

        logger.info("Stage 10/13 — Compound LTR")

        ltr = LTRScorer()

        df = ltr.compute(df)

        mlflow.log_metric(
            "mean_ltr_score",
            float(df["ltr_score"].mean()),
        )

        # ---------------------------------------------------------
        # Stage 11
        # ---------------------------------------------------------

        logger.info("Stage 11/13 — Criticality Index")

        ci = CriticalityIndexer()

        df = ci.compute(df)

        if "ci_band" in df.columns:

            df["ci_tier"] = df["ci_band"]

        mlflow.log_metric(
            "mean_ci_score",
            float(df["ci_score"].mean()),
        )

        # ---------------------------------------------------------
        # Stage 12
        # ---------------------------------------------------------

        logger.info("Stage 12/13 — RUL Engine")

        rul_engine = RULEngine()

        df = rul_engine.compute(df)

        mlflow.log_metric(
            "imminent_failures",
            int(df["imminent_failure"].sum()),
        )

        # ---------------------------------------------------------
        # Stage 13 — Bellman Optimisation
        # ---------------------------------------------------------

        logger.info("Stage 13/13 — Bellman Optimisation")

        bellman = BellmanEngine()

        df = bellman.compute(df)

        # =========================================================
        # STAGE 14 — SUPPLIER QUALIFICATION
        # =========================================================

        logger.info("Stage 14/15 — Supplier Qualification")

        supplier_engine = DecisionTreeQualifier(
            max_depth=5,
            random_state=42,
        )

        df = supplier_engine.fit_predict(df)[0]

        logger.info(
            ("Supplier qualification complete | " "critical=%d"),
            int((df["supplier_risk_class"] == "Critical").sum()),
        )

        # =========================================================
        # STAGE 15 — NASH EQUILIBRIUM
        # =========================================================

        logger.info("Stage 15/15 — Nash Equilibrium")

        nash_engine = NashEquilibriumEngine()

        df = nash_engine.compute(df)

        logger.info(
            ("Nash equilibrium complete | " "strategic=%d"),
            int((df["supplier_strategy"] == "Strategic").sum()),
        )

        # =========================================================
        # STAGE 15.5 — REPEATED GAME REPUTATION
        # =========================================================

        logger.info("Stage 15.5/15 — Repeated Game Reputation")

        reputation_model = RepeatedGameModel(
            T=24,
            discount_factor=0.92,
            late_threshold_days=7.0,
            cooperation_surplus=100.0,
            defection_gain=20.0,
            grim_trigger_threshold=3,
        )

        reputation_df, reputation_matrix = reputation_model.score(df)

        # -------------------------------------------------
        # SCHEMA NORMALIZATION
        # -------------------------------------------------

        df["item_id"] = df["item_id"].astype(str)

        reputation_df["item_id"] = reputation_df["item_id"].astype(str)

        reputation_cols = [
            "item_id",
            "reputation_score",
            "grim_trigger_fired",
            "n_defections",
            "delta_satisfied",
            "recommended_action",
        ]

        df = df.merge(
            reputation_df[reputation_cols],
            on="item_id",
            how="left",
        )

        logger.info(
            ("Repeated game complete | " "mean_reputation=%.3f"),
            float(df["reputation_score"].mean()),
        )

        # ---------------------------------------------------------
        # Backward compatibility aliases
        # ---------------------------------------------------------

        df["q_star"] = df["bellman_q_star"]

        df["rop"] = df["bellman_rop"]

        # ---------------------------------------------------------
        # Dynamic service level approximation
        # ---------------------------------------------------------

        df["tsl"] = np.clip(
            0.85 + (df["ci_score"] * 0.15),
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
        # Integration compatibility aliases
        # ---------------------------------------------------------

        df["rul_signal"] = np.clip(
            df["rul_days"],
            1,
            None,
        )

        df["pull_trigger"] = df["imminent_failure"]

        df["decoupling_mode"] = np.where(
            df["supplier_strategy"] == "Strategic",
            "CODP",
            "Push+Pull",
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

                raise ValueError((f"Pipeline missing " f"required column: {col}"))

        # ---------------------------------------------------------
        # MLflow metrics
        # ---------------------------------------------------------

        mlflow.log_metric(
            "mean_tsl",
            float(df["tsl"].mean()),
        )

        mlflow.log_metric(
            "mean_q_star",
            float(df["q_star"].mean()),
        )

        mlflow.log_metric(
            "mean_rop",
            float(df["rop"].mean()),
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
            "rows": len(df),
            "mean_geo_risk": float(df["geo_risk_score"].mean()),
            "mean_ltr_score": float(df["ltr_score"].mean()),
            "mean_ci_score": float(df["ci_score"].mean()),
            "mean_q_star": float(df["q_star"].mean()),
            "n_critical_suppliers": int(
                (df["supplier_risk_class"] == "Critical").sum()
            ),
            "mean_strategic_risk": float(df["strategic_risk_score"].mean()),
            "active_disruptions": int(df["scenario_active"].sum()),
            "mean_resilience_multiplier": float(
                df.get("resilience_multiplier", pd.Series([1.0])).mean()
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

        # ---------------------------------------------------------
        # Figures
        # ---------------------------------------------------------

        plt.figure(figsize=(8, 5))

        sns.histplot(
            df["ci_score"],
            bins=20,
        )

        plt.title("Criticality Distribution")

        ci_hist_path = FIGURES_DIR / "ci_histogram.png"

        plt.savefig(
            ci_hist_path,
            bbox_inches="tight",
        )

        plt.close()

        # ---------------------------------------------------------
        # MLflow artifacts
        # ---------------------------------------------------------

        mlflow.log_metrics(metrics)

        mlflow.log_artifact(str(OUTPUT_PATH))

        mlflow.log_artifact(str(METRICS_PATH))

        mlflow.log_artifact(str(ci_hist_path))

        logger.info(
            ("Pipeline complete | " "output=%s"),
            OUTPUT_PATH,
        )

        # ── Stage 15: Bullwhip Quantification ─────────────────────────────

        logger.info("Stage 15/15 — Bullwhip Effect Quantification")

        bw_model = BullwhipModel(
            n_periods=52,
            n_echelons=4,
            lead_times=[1, 2, 3, 4],
            smoothing_alpha=0.20,
            seed=42,
        )

        bw_summary_dp = bw_model.analyze_all(
            df,
            policy_type="dp_optimized",
        )

        bw_summary_std = bw_model.analyze_all(
            df,
            policy_type="standard",
        )

        BW_PATH = Path("data/processed/bullwhip_results.parquet")

        bw_df = bw_model.to_dataframe(bw_summary_dp)

        bw_df.to_parquet(
            BW_PATH,
            index=False,
        )

        reduction = (
            bw_summary_std.mean_total_amplification
            - bw_summary_dp.mean_total_amplification
        ) / max(
            bw_summary_std.mean_total_amplification,
            1,
        )

        mlflow.log_metric(
            "bwr_standard_mean",
            bw_summary_std.mean_total_amplification,
        )

        mlflow.log_metric(
            "bwr_dp_optimized_mean",
            bw_summary_dp.mean_total_amplification,
        )

        mlflow.log_metric(
            "bullwhip_reduction_pct",
            reduction * 100,
        )

        logger.info(
            "Bullwhip | standard_BWR=%.2f | dp_BWR=%.2f | reduction=%.1f%%",
            bw_summary_std.mean_total_amplification,
            bw_summary_dp.mean_total_amplification,
            reduction * 100,
        )

        metrics.update(
            {
                "mean_ci_score": round(
                    df["ci_score"].mean(),
                    4,
                ),
                "mean_tsl": round(
                    df["tsl"].mean(),
                    4,
                ),
                "mean_q_star": round(
                    df["q_star"].mean(),
                    4,
                ),
                "mean_geo_risk": round(
                    df["geo_risk_score"].mean(),
                    4,
                ),
                "bwr_standard": round(
                    bw_summary_std.mean_total_amplification,
                    4,
                ),
                "bwr_dp_optimized": round(
                    bw_summary_dp.mean_total_amplification,
                    4,
                ),
                "bullwhip_reduction_pct": round(
                    reduction * 100,
                    2,
                ),
                "v1_2_stage_count": 15,
            }
        )

        METRICS_PATH.write_text(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        V12_PATH = Path("data/processed/sku_master_v1.2.parquet")

        df.to_parquet(
            V12_PATH,
            index=False,
        )

        logger.info(
            "v1.2 SKU Master saved: %s",
            V12_PATH,
        )

        return df
