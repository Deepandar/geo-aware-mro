"""
v1.2 Full Integration Test — Release Gate
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from src.pipelines.sku_pipeline import run_pipeline


@pytest.fixture(scope="module")
def v12_df():

    return run_pipeline(
        n_skus=100
    )


V1_2_REQUIRED_COLUMNS = [

    "item_id",
    "abc_class",
    "ved_class",
    "fns_class",

    "ci_score",
    "ci_tier",

    "ltr_score",
    "geo_risk_score",

    "q_star",
    "rop",
    "tsl",

    "supplier_risk_class",
    "strategic_risk_score",

    "rul_signal",
    "pull_trigger",

    "decoupling_mode",

    "reputation_score",
    "grim_trigger_fired",
    "recommended_action",

]


def test_pipeline_returns_dataframe(v12_df):

    assert isinstance(
        v12_df,
        pd.DataFrame
    )


def test_pipeline_row_count(v12_df):

    assert len(v12_df) == 100


def test_required_columns_present(v12_df):

    missing = [
        c for c in V1_2_REQUIRED_COLUMNS
        if c not in v12_df.columns
    ]

    assert not missing, (
        f"Missing columns: {missing}"
    )


def test_ci_score_range(v12_df):

    assert v12_df[
        "ci_score"
    ].between(0, 1).all()


def test_geo_risk_range(v12_df):

    assert v12_df[
        "geo_risk_score"
    ].between(0, 1).all()


def test_tsl_range(v12_df):

    assert v12_df[
        "tsl"
    ].between(0, 1).all()


def test_reputation_score_range(v12_df):

    assert v12_df[
        "reputation_score"
    ].between(0, 1).all()


def test_q_star_non_negative(v12_df):

    assert (
        v12_df["q_star"] >= 0
    ).all()


def test_rul_signal_positive(v12_df):

    assert (
        v12_df["rul_signal"] > 0
    ).all()


def test_decoupling_modes_valid(v12_df):

    valid = {
        "Push",
        "Pull",
        "Push+Pull",
        "Newsvendor",
    }

    assert set(
        v12_df[
            "decoupling_mode"
        ].unique()
    ).issubset(valid)


def test_folk_theorem_satisfied(v12_df):

    assert (
        v12_df[
            "delta_satisfied"
        ].mean()
        > 0.5
    )


def test_reputation_actions_exist(v12_df):

    assert (
        v12_df[
            "recommended_action"
        ].nunique()
        >= 3
    )


def test_metrics_json_written():

    path = Path(
        "data/processed/pipeline_metrics.json"
    )

    assert path.exists()

    metrics = json.loads(
        path.read_text()
    )

    required = [
        "mean_ci_score",
        "mean_tsl",
        "mean_q_star",
        "mean_geo_risk",
        "bwr_standard",
        "bwr_dp_optimized",
    ]

    for key in required:

        assert key in metrics


def test_v12_parquet_written():

    path = Path(
        "data/processed/sku_master_v1.2.parquet"
    )

    assert path.exists()

    df = pd.read_parquet(path)

    assert len(df) > 0
