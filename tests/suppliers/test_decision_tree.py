import pytest
import numpy as np
import pandas as pd

from src.suppliers.decision_tree_qualifier import (
    DecisionTreeQualifier,
)


@pytest.fixture
def sample_sku_df():

    np.random.seed(42)

    n = 200

    return pd.DataFrame({

        "item_id": [
            f"SKU{i:04d}"
            for i in range(n)
        ],

        "abc_class": np.random.choice(
            ["A","B","C"],
            n,
        ),

        "ved_class": np.random.choice(
            ["V","E","D"],
            n,
        ),

        "fns_class": np.random.choice(
            ["F","N","S"],
            n,
        ),

        "ved_score": np.random.choice(
            [1.0, 0.5, 0.0],
            n,
        ),

        "geo_risk_score": np.random.uniform(
            0,
            1,
            n,
        ),

        "lead_time_days": np.random.randint(
            7,
            180,
            n,
        ),

        "std_lead_time": np.random.uniform(
            2,
            30,
            n,
        ),

        "hhi_score": np.random.uniform(
            0.1,
            0.9,
            n,
        ),

        "ltr_score": np.random.uniform(
            0,
            0.5,
            n,
        ),

        "ci_score": np.random.uniform(
            0.2,
            0.9,
            n,
        ),

        "supply_origin_country":
            np.random.choice(
                [
                    "IN",
                    "CN",
                    "RU",
                    "US",
                    "DE",
                    "FR",
                ],
                n,
            ),
    })


@pytest.fixture
def qualifier():

    return DecisionTreeQualifier(
        max_depth=5
    )


def test_fit_returns_result(
    qualifier,
    sample_sku_df,
):

    result = qualifier.fit(
        sample_sku_df
    )

    assert (
        result.n_skus
        ==
        len(sample_sku_df)
    )


def test_predict_adds_columns(
    qualifier,
    sample_sku_df,
):

    qualifier.fit(
        sample_sku_df
    )

    out = qualifier.predict(
        sample_sku_df
    )

    assert (
        "supplier_risk_class"
        in out.columns
    )

    assert (
        "supplier_risk_score"
        in out.columns
    )


def test_risk_scores_range(
    qualifier,
    sample_sku_df,
):

    qualifier.fit(
        sample_sku_df
    )

    out = qualifier.predict(
        sample_sku_df
    )

    assert (
        out[
            "supplier_risk_score"
        ]
        .between(0,1)
        .all()
    )
