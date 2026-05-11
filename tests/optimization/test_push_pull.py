import pytest
import numpy as np
import pandas as pd

from src.optimization.push_pull_engine import (
    PushPullEngine,
)


@pytest.fixture
def engine():

    return PushPullEngine(
        push_density_threshold=0.50,
        pull_rul_threshold=20.0,
        push_weight=0.60,
    )


@pytest.fixture
def base_df():

    np.random.seed(42)

    n = 60

    return pd.DataFrame(
        {
            "item_id": [f"SKU{i:03d}" for i in range(n)],
            "depot_tier": np.random.choice(
                [
                    "Forward",
                    "Border",
                    "Rear",
                ],
                n,
            ),
            "equipment_density_score": np.random.uniform(
                0.0,
                1.0,
                n,
            ),
            "rul_signal": np.random.uniform(
                5,
                200,
                n,
            ),
            "base_stock_level": np.random.uniform(
                5,
                50,
                n,
            ),
            "q_star": np.random.uniform(
                5,
                40,
                n,
            ),
            "ci_score": np.random.uniform(
                0.2,
                0.9,
                n,
            ),
        }
    )


def test_compute_adds_required_columns(
    engine,
    base_df,
):

    out = engine.compute(base_df)

    required = {
        "decoupling_mode",
        "push_qty",
        "pull_qty",
        "total_position",
        "codp_tier",
        "pp_rationale",
    }

    assert required.issubset(out.columns)


def test_decoupling_mode_valid_values(
    engine,
    base_df,
):

    out = engine.compute(base_df)

    valid = {
        "Push",
        "Pull",
        "Push+Pull",
        "Newsvendor",
    }

    assert set(out["decoupling_mode"].unique()).issubset(valid)


def test_high_density_low_rul_is_push_pull(
    engine,
):

    df = pd.DataFrame(
        {
            "item_id": ["SKU_A"],
            "depot_tier": ["Forward"],
            "equipment_density_score": [0.95],
            "rul_signal": [5.0],
            "base_stock_level": [20.0],
            "q_star": [15.0],
        }
    )

    out = engine.compute(df)

    assert out["decoupling_mode"].iloc[0] == "Push+Pull"


def test_missing_item_id_raises(
    engine,
):

    bad = pd.DataFrame({"equipment_density_score": [0.5]})

    with pytest.raises(ValueError):

        engine.compute(bad)
