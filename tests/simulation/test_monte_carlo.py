import pytest
import numpy as np
import pandas as pd

from src.simulation.monte_carlo import (
    MonteCarloPipeline,
)


@pytest.fixture
def sku_df():

    np.random.seed(42)

    n = 25

    return pd.DataFrame({

        "item_id":
            [f"SKU{i:03d}" for i in range(n)],

        "ci_tier":
            np.random.choice(
                ["High", "Medium", "Low"],
                n,
            ),

        "fns_class":
            np.random.choice(
                ["F", "N", "S"],
                n,
            ),

        "mean_demand":
            np.random.uniform(2, 15, n),

        "std_demand":
            np.random.uniform(1, 5, n),

        "q_star":
            np.random.uniform(5, 25, n),

        "rop":
            np.random.uniform(2, 10, n),

        "tsl":
            np.random.uniform(0.80, 0.99, n),

        "unit_cost":
            np.random.uniform(100, 1000, n),

        "stockout_cost_usd":
            np.random.uniform(500, 5000, n),

        "mean_lead_time":
            np.random.uniform(14, 90, n),

        "std_lead_time":
            np.random.uniform(3, 20, n),

        "supply_origin_country":
            np.random.choice(
                ["IN", "CN", "RU", "US"],
                n,
            ),

        "geo_risk_score":
            np.random.uniform(0, 1, n),
    })


@pytest.fixture
def mc():

    return MonteCarloPipeline(
        n_trials=5,
        n_periods=6,
        fast_mode=True,
        seed=42,
    )


def test_run_scenario_returns_dataframe(
    mc,
    sku_df,
):

    out = mc.run_scenario(
        sku_df,
        "baseline",
    )

    assert isinstance(out, pd.DataFrame)

    assert len(out) == 5


def test_run_all_returns_dataframe(
    mc,
    sku_df,
):

    out = mc.run_all(sku_df)

    assert isinstance(out, pd.DataFrame)

    assert len(out) > 0


def test_expected_columns_exist(
    mc,
    sku_df,
):

    out = mc.run_all(sku_df)

    required = [

        "trial_id",
        "scenario",

        "mean_fill_rate",

        "fill_rate_high",
        "fill_rate_medium",
        "fill_rate_low",

        "total_stockout_cost",

        "tsl_compliance_rate",

        "cvs_fill_rate",
        "cds_fill_rate",
        "cvs_fix_holds",
    ]

    for col in required:

        assert col in out.columns


def test_fill_rate_bounded(
    mc,
    sku_df,
):

    out = mc.run_all(sku_df)

    assert (
        out["mean_fill_rate"]
        .between(0, 1)
        .all()
    )


def test_tsl_compliance_bounded(
    mc,
    sku_df,
):

    out = mc.run_all(sku_df)

    assert (
        out["tsl_compliance_rate"]
        .between(0, 1)
        .all()
    )


def test_stockout_cost_non_negative(
    mc,
    sku_df,
):

    out = mc.run_all(sku_df)

    assert (
        out["total_stockout_cost"] >= 0
    ).all()


def test_all_scenarios_present(
    mc,
    sku_df,
):

    out = mc.run_all(sku_df)

    expected = {
        "baseline",
        "sanctions",
        "conflict",
        "pandemic",
        "port_closure",
        "logistics_collapse",
    }

    assert expected.issubset(
        set(out["scenario"].unique())
    )


def test_cvs_fix_returns_boolean(
    mc,
    sku_df,
):

    out = mc.run_all(sku_df)

    assert (
        out["cvs_fix_holds"]
        .isin([True, False])
        .all()
    )
