import numpy as np
import pandas as pd
import pytest

from src.suppliers.nash_equilibrium import (
    NashEquilibriumModel,
)


@pytest.fixture
def base_df():

    np.random.seed(42)

    n = 100

    return pd.DataFrame({

        "item_id": [
            f"SKU{i:04d}"
            for i in range(n)
        ],

        "ved_class": np.random.choice(
            ["V", "E", "D"],
            n,
        ),

        "geo_risk_score": np.random.uniform(
            0,
            1,
            n,
        ),

        "hhi_score": np.random.uniform(
            0.1,
            0.9,
            n,
        ),
    })


@pytest.fixture
def model():

    return NashEquilibriumModel()


def test_score_adds_columns(
    model,
    base_df,
):

    df_out, _ = model.score(base_df)

    required = [

        "strategic_risk_score",
        "sourcing_strategy",
        "ne_type",
        "buffer_stock_signal",
        "ne_price_equilibrium",
    ]

    for col in required:

        assert col in df_out.columns


def test_srs_range(
    model,
    base_df,
):

    df_out, _ = model.score(base_df)

    assert (
        df_out[
            "strategic_risk_score"
        ].between(
            0,
            1,
        ).all()
    )


def test_buffer_non_negative(
    model,
    base_df,
):

    df_out, _ = model.score(base_df)

    assert (
        df_out[
            "buffer_stock_signal"
        ] >= 0
    ).all()


def test_high_risk_vital():

    model = NashEquilibriumModel()

    df = pd.DataFrame({

        "item_id": ["X"],

        "ved_class": ["V"],

        "geo_risk_score": [0.95],

        "hhi_score": [0.95],
    })

    df_out, _ = model.score(df)

    assert (
        df_out[
            "sourcing_strategy"
        ].iloc[0]
        ==
        "Dual-Source (Mandatory)"
    )


def test_low_risk_single_source():

    model = NashEquilibriumModel()

    df = pd.DataFrame({

        "item_id": ["Y"],

        "ved_class": ["D"],

        "geo_risk_score": [0.05],

        "hhi_score": [0.05],
    })

    df_out, _ = model.score(df)

    assert (
        df_out[
            "sourcing_strategy"
        ].iloc[0]
        ==
        "Single-Source"
    )
