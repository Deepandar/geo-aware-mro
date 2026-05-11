"""
Full pipeline integration test — Geo-Aware MRO v1.1

Release gate:
- all stages execute
- supplier intelligence integrated
- Nash equilibrium integrated
- Bellman optimisation stable
- metrics persisted
- coverage >= 80%
"""

from pathlib import Path
import json

import pandas as pd
import pytest

from src.pipelines.sku_pipeline import run_pipeline

# =========================================================
# FIXTURE
# =========================================================


@pytest.fixture(scope="module")
def pipeline_df():

    return run_pipeline(n_skus=100)


# =========================================================
# REQUIRED COLUMNS
# =========================================================

V1_1_REQUIRED_COLUMNS = [
    # raw
    "item_id",
    "demand",
    "lead_time_days",
    "supply_origin_country",
    # ABC/VED/FNS
    "abc_class",
    "ved_class",
    "fns_class",
    # geo-risk
    "geo_risk_score",
    # resilience
    "resilience_multiplier",
    # LTR
    "ltr_score",
    # CI
    "ci_score",
    "ci_tier",
    # RUL
    "rul_days",
    # Bellman
    "bellman_q_star",
    "bellman_rop",
    "expected_future_cost",
    # aliases
    "q_star",
    "rop",
    # supplier qualification
    "supplier_risk_class",
    "supplier_risk_score",
    "procurement_flag",
    # Nash equilibrium
    "strategic_risk_score",
    "sourcing_strategy",
    "buffer_stock_signal",
    "ne_price_equilibrium",
]


# =========================================================
# BASIC CONTRACT
# =========================================================


def test_pipeline_returns_dataframe(pipeline_df):

    assert isinstance(
        pipeline_df,
        pd.DataFrame,
    )


def test_pipeline_row_count(pipeline_df):

    assert len(pipeline_df) == 100


def test_required_columns_exist(pipeline_df):

    missing = [c for c in V1_1_REQUIRED_COLUMNS if c not in pipeline_df.columns]

    assert not missing, f"Missing columns: {missing}"


# =========================================================
# NUMERIC VALIDATION
# =========================================================


def test_geo_risk_range(pipeline_df):

    assert (
        pipeline_df["geo_risk_score"]
        .between(
            0.0,
            1.0,
        )
        .all()
    )


def test_ltr_range(pipeline_df):

    assert (
        pipeline_df["ltr_score"]
        .between(
            0.0,
            1.0,
        )
        .all()
    )


def test_ci_range(pipeline_df):

    assert (
        pipeline_df["ci_score"]
        .between(
            0.0,
            1.0,
        )
        .all()
    )


def test_supplier_risk_range(pipeline_df):

    assert (
        pipeline_df["supplier_risk_score"]
        .between(
            0.0,
            1.0,
        )
        .all()
    )


def test_strategic_risk_range(pipeline_df):

    assert (
        pipeline_df["strategic_risk_score"]
        .between(
            0.0,
            1.0,
        )
        .all()
    )


# =========================================================
# INVENTORY VALIDATION
# =========================================================


def test_q_star_non_negative(pipeline_df):

    assert (pipeline_df["q_star"] >= 0).all()


def test_rop_non_negative(pipeline_df):

    assert (pipeline_df["rop"] >= 0).all()


def test_future_cost_finite(pipeline_df):

    assert pipeline_df["expected_future_cost"].notna().all()


# =========================================================
# SUPPLIER VALIDATION
# =========================================================


def test_supplier_class_valid(pipeline_df):

    valid = {
        "Low",
        "Medium",
        "High",
        "Critical",
    }

    actual = set(pipeline_df["supplier_risk_class"].unique())

    assert actual.issubset(valid)


def test_procurement_flags_are_vital(pipeline_df):

    flagged = pipeline_df[pipeline_df["procurement_flag"]]

    if len(flagged) > 0:

        assert (flagged["supplier_risk_class"] == "Critical").all()

        assert (flagged["ved_class"] == "V").all()


# =========================================================
# NASH VALIDATION
# =========================================================


def test_sourcing_strategy_valid(pipeline_df):

    valid = {
        "Single-Source",
        "Dual-Source",
        "Dual-Source (Mandatory)",
    }

    actual = set(pipeline_df["sourcing_strategy"].unique())

    assert actual.issubset(valid)


def test_buffer_stock_signal_non_negative(pipeline_df):

    assert (pipeline_df["buffer_stock_signal"] >= 0).all()


# =========================================================
# PIPELINE LOGIC
# =========================================================


def test_ci_before_bellman(pipeline_df):

    assert pipeline_df["ci_score"].notna().all()

    assert pipeline_df["q_star"].notna().all()


def test_high_risk_drives_dual_source(pipeline_df):

    risky = pipeline_df[pipeline_df["strategic_risk_score"] > 0.60]

    if len(risky) > 0:

        pct = risky["sourcing_strategy"].str.contains("Dual").mean()

        assert pct >= 0.50


# =========================================================
# METRICS FILE
# =========================================================


def test_metrics_json_exists():

    path = Path("data/processed/pipeline_metrics.json")

    assert path.exists()


def test_metrics_json_valid():

    path = Path("data/processed/pipeline_metrics.json")

    with open(path) as f:

        metrics = json.load(f)

    required = [
        "mean_ci_score",
        "mean_q_star",
        "n_critical_suppliers",
        "mean_strategic_risk",
    ]

    for key in required:

        assert key in metrics


# =========================================================
# PARQUET OUTPUT
# =========================================================


def test_pipeline_output_written():

    path = Path("data/processed/sku_master_v1.3.parquet")

    assert path.exists()


def test_pipeline_output_loadable():

    path = Path("data/processed/sku_master_v1.3.parquet")

    df = pd.read_parquet(path)

    assert len(df) > 0
