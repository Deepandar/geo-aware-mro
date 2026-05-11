import pytest
import numpy as np
import pandas as pd

from src.suppliers.repeated_game import RepeatedGameModel


@pytest.fixture
def model():

    return RepeatedGameModel(
        T=12,
        discount_factor=0.92,
        late_threshold_days=7.0,
        cooperation_surplus=100.0,
        defection_gain=20.0,
        grim_trigger_threshold=1,
    )


@pytest.fixture
def base_df():

    np.random.seed(42)

    n = 80

    return pd.DataFrame(
        {
            "item_id": [f"SKU{i:04d}" for i in range(n)],
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
                    0.2,
                    0.4,
                    0.3,
                    0.1,
                ],
            ),
            "ved_class": np.random.choice(
                [
                    "V",
                    "E",
                    "D",
                ],
                n,
            ),
        }
    )


def test_score_adds_required_columns(
    model,
    base_df,
):

    out, rep = model.score(base_df)

    required = {
        "reputation_score",
        "grim_trigger_fired",
        "n_defections",
        "delta_satisfied",
        "recommended_action",
    }

    assert required.issubset(out.columns)


def test_reputation_range(
    model,
    base_df,
):

    out, _ = model.score(base_df)

    assert out["reputation_score"].between(0, 1).all()


def test_folk_theorem(
    model,
):

    ft = model.folk_theorem_summary()

    assert ft["folk_theorem_satisfied"] is True


def test_missing_item_id_raises(
    model,
):

    bad = pd.DataFrame({"geo_risk_score": [0.5]})

    with pytest.raises(ValueError):

        model.score(bad)
