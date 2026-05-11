import pytest
import numpy as np
import pandas as pd

from src.suppliers.repeated_game import (
    RepeatedGameModel,
)


@pytest.fixture
def full_sku_df():

    np.random.seed(42)

    n = 500

    return pd.DataFrame(
        {
            "item_id": [f"SKU{i:04d}" for i in range(n)],
            "abc_class": np.random.choice(["A", "B", "C"], n),
            "ved_class": np.random.choice(["V", "E", "D"], n),
            "fns_class": np.random.choice(["F", "N", "S"], n),
            "ci_tier": np.random.choice(["High", "Medium", "Low"], n),
            "ci_score": np.random.uniform(
                0.2,
                0.95,
                n,
            ),
            "geo_risk_score": np.random.uniform(
                0,
                1,
                n,
            ),
            "supplier_risk_class": np.random.choice(
                [
                    "Low",
                    "Medium",
                    "High",
                    "Critical",
                ],
                n,
                p=[
                    0.20,
                    0.45,
                    0.25,
                    0.10,
                ],
            ),
            "strategic_risk_score": np.random.uniform(
                0,
                1,
                n,
            ),
            "sourcing_strategy": np.random.choice(
                [
                    "Single-Source",
                    "Dual-Source",
                    "Dual-Source (Mandatory)",
                ],
                n,
            ),
        }
    )


@pytest.fixture
def model():

    return RepeatedGameModel(
        T=24,
        discount_factor=0.92,
        late_threshold_days=7.0,
        cooperation_surplus=100.0,
        defection_gain=20.0,
        grim_trigger_threshold=1,
    )


def test_500_sku_reputation_no_errors(
    model,
    full_sku_df,
):

    df_out, rep_matrix = model.score(full_sku_df)

    assert len(df_out) == 500

    assert len(rep_matrix) == 500


def test_reputation_scores_in_range(
    model,
    full_sku_df,
):

    df_out, _ = model.score(full_sku_df)

    assert df_out["reputation_score"].between(0.0, 1.0).all()


def test_rep_matrix_has_required_columns(
    model,
    full_sku_df,
):

    _, rep_matrix = model.score(full_sku_df)

    required = [
        "item_id",
        "reputation_score",
        "grim_trigger_fired",
        "n_defections",
        "delta_satisfied",
        "recommended_action",
    ]

    for col in required:

        assert col in rep_matrix.columns


def test_folk_theorem_satisfied_with_high_delta(model):

    ft = model.folk_theorem_summary()

    assert ft["folk_theorem_satisfied"] is True

    assert ft["delta_required"] < model.discount_factor


def test_action_distribution_covers_all_actions(
    model,
    full_sku_df,
):

    df_out, _ = model.score(full_sku_df)

    unique_actions = df_out["recommended_action"].nunique()

    assert unique_actions >= 3
