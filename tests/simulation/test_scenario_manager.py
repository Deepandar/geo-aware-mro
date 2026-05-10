import pytest
import numpy as np
import pandas as pd

from src.simulation.scenario_injector import (
    ScenarioInjector,
    ScenarioImpact,
)


@pytest.fixture
def injector():

    return ScenarioInjector(
        "config/scenario_library.yaml"
    )


@pytest.fixture
def sku_df():

    np.random.seed(42)

    n = 50

    return pd.DataFrame({

        "item_id": [
            f"SKU{i:03d}"
            for i in range(n)
        ],

        "supply_origin_country":
            np.random.choice(

                [
                    "IN",
                    "CN",
                    "RU",
                    "UA",
                    "US",
                    "DE",
                ],

                n,

                p=[
                    0.25,
                    0.20,
                    0.15,
                    0.10,
                    0.15,
                    0.15,
                ],
            ),

        "mean_lead_time":
            np.random.uniform(
                14,
                180,
                n,
            ),

        "std_lead_time":
            np.random.uniform(
                3,
                30,
                n,
            ),

        "geo_risk_score":
            np.random.uniform(
                0,
                1,
                n,
            ),
    })


def test_inject_baseline_returns_impact(
    injector,
    sku_df,
):

    impact = injector.inject(
        sku_df,
        "baseline",
    )

    assert isinstance(
        impact,
        ScenarioImpact,
    )


def test_all_scenarios_injectable(
    injector,
    sku_df,
):

    for scenario in [

        "baseline",

        "sanctions",

        "conflict",

        "pandemic",

        "port_closure",

        "logistics_collapse",
    ]:

        impact = injector.inject(
            sku_df,
            scenario,
        )

        assert (
            impact.scenario_name
            == scenario
        )


def test_unknown_scenario_raises(
    injector,
    sku_df,
):

    with pytest.raises(
        ValueError
    ):

        injector.inject(
            sku_df,
            "bad_scenario",
        )


def test_impact_summary_dataframe(
    injector,
    sku_df,
):

    impacts = injector.inject_all(
        sku_df
    )

    df = injector.impact_summary(
        impacts
    )

    assert len(df) == 6

    assert (
        "scenario"
        in df.columns
    )

