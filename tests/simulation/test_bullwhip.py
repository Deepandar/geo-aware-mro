import pytest
import numpy as np
import pandas as pd

from src.simulation.bullwhip_model import (
    BullwhipModel,
    BullwhipSummary,
)


@pytest.fixture
def model():
    return BullwhipModel(
        n_periods=24,
        n_echelons=4,
        lead_times=[1,2,3,4],
        smoothing_alpha=0.20,
        seed=42,
    )


@pytest.fixture
def sku_df():

    np.random.seed(42)

    n = 40

    return pd.DataFrame({

        "item_id":
            [f"SKU{i:03d}" for i in range(n)],

        "abc_class":
            np.random.choice(
                ["A","B","C"], n
            ),

        "ved_class":
            np.random.choice(
                ["V","E","D"], n
            ),

        "fns_class":
            np.random.choice(
                ["F","N","S"], n
            ),

        "ci_tier":
            np.random.choice(
                ["High","Medium","Low"], n
            ),

        "mean_demand":
            np.random.uniform(
                3, 15, n
            ),

        "std_demand":
            np.random.uniform(
                1, 5, n
            ),

        "dp_q_star":
            np.random.uniform(
                10, 40, n
            ),

        "q_star":
            np.random.uniform(
                8, 30, n
            ),

        "rop":
            np.random.uniform(
                3, 12, n
            ),

        "base_stock_level":
            np.random.uniform(
                15, 45, n
            ),

        "push_qty":
            np.random.uniform(
                0, 8, n
            ),

        "pull_trigger":
            np.random.choice(
                [True, False],
                n,
                p=[0.15,0.85]
            ),
    })


def test_analyze_sku_returns_result(
    model,
    sku_df,
):
    row = sku_df.iloc[0].to_dict()

    result = model.analyze_sku(
        row,
        policy_type="dp_optimized",
    )

    assert result.item_id is not None
    assert len(result.echelons) == 4


def test_echelon0_bwr_equals_1(
    model,
    sku_df,
):
    row = sku_df.iloc[0].to_dict()

    result = model.analyze_sku(row)

    assert abs(
        result.echelons[0].bullwhip_ratio - 1.0
    ) < 1e-6


def test_compare_policies(
    model,
    sku_df,
):

    comp = model.compare_policies(
        sku_df
    )

    assert len(comp) == 3

    assert set(comp["policy"]) == {
        "standard",
        "dp_optimized",
        "codp",
    }


def test_codp_reduces_bullwhip(
    model,
    sku_df,
):

    comp = model.compare_policies(
        sku_df
    )

    std = float(
        comp[
            comp.policy=="standard"
        ]["mean_total_amp"].iloc[0]
    )

    codp = float(
        comp[
            comp.policy=="codp"
        ]["mean_total_amp"].iloc[0]
    )

    assert codp <= std + 0.5


def test_summary_generation(
    model,
    sku_df,
):

    summary = model.analyze_all(
        sku_df
    )

    assert isinstance(
        summary,
        BullwhipSummary,
    )

    assert summary.n_skus == len(
        sku_df
    )


def test_dataframe_export(
    model,
    sku_df,
):

    summary = model.analyze_all(
        sku_df
    )

    df = model.to_dataframe(
        summary
    )

    assert len(df) == len(sku_df)

    assert "bwr_e0" in df.columns

