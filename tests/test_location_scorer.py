"""
pytest suite for LocationScorer — v1.1
Run: pytest tests/test_location_scorer.py -v
"""

import pytest
import numpy as np
import pandas as pd

from src.classifiers.location_scorer import LocationScorer

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def scorer():
    return LocationScorer(config_path="config/location_config.yaml")


@pytest.fixture
def minimal_sku_df():
    """Minimal valid SKU Master for testing."""
    return pd.DataFrame(
        {
            "item_id": ["SKU001", "SKU002", "SKU003", "SKU004"],
            "depot_tier": ["Forward", "Border", "Rear", "Forward"],
            "equipment_density_score": [0.9, 0.6, 0.3, 0.7],
            "environment_multiplier": [1.0, 1.0, 1.0, 1.0],  # v1.1 default
        }
    )


# ── Tests: correctness ────────────────────────────────────────────────────────


def test_required_columns_missing_raises(scorer):
    """Missing required columns must raise ValueError immediately."""
    bad_df = pd.DataFrame({"item_id": ["SKU001"], "depot_tier": ["Forward"]})
    with pytest.raises(ValueError, match="missing required columns"):
        scorer.score_sku_master(bad_df)


def test_base_scores_are_ordinal(scorer, minimal_sku_df):
    """Forward must score higher than Border, Border higher than Rear."""
    result = scorer.score_sku_master(minimal_sku_df)
    forward = result[result["depot_tier"] == "Forward"]["location_score_adj"].mean()
    border = result[result["depot_tier"] == "Border"]["location_score_adj"].mean()
    rear = result[result["depot_tier"] == "Rear"]["location_score_adj"].mean()
    assert forward > border > rear, (
        f"Ordinal ordering violated: "
        f"Forward={forward:.3f}, Border={border:.3f}, Rear={rear:.3f}"
    )


def test_location_score_adj_range(scorer, minimal_sku_df):
    """All adjusted scores must be in [0, 1]."""
    result = scorer.score_sku_master(minimal_sku_df)
    assert (
        result["location_score_adj"].between(0.0, 1.0).all()
    ), "location_score_adj values outside [0, 1] detected."


def test_default_multiplier_is_one(scorer, minimal_sku_df):
    """In v1.1, environment_multiplier must remain 1.0 for all rows."""
    result = scorer.score_sku_master(minimal_sku_df)
    assert (
        result["environment_multiplier"] == 1.0
    ).all(), "v1.1: environment_multiplier should be 1.0 for all SKUs."


def test_adj_increases_with_multiplier(scorer, minimal_sku_df):
    """If we manually bump the multiplier, adj score should increase."""
    df = minimal_sku_df.copy()
    df.loc[df["depot_tier"] == "Forward", "environment_multiplier"] = 1.2
    result = scorer.score_sku_master(df)
    base = result["base_position_score"]
    env = result["environment_multiplier"]
    raw = base * env
    # Check monotonicity: higher env should produce higher raw before normalization
    assert (
        raw[df["depot_tier"] == "Forward"].min()
        > raw[df["depot_tier"] == "Border"].max()
    )


def test_v12_hook_inactive_in_v11(scorer):
    """use_environment_profiles must be False in v1.1 config."""
    assert scorer.use_env_profiles is False, (
        "v1.1: use_environment_profiles should be False. "
        "Flip to True only in v1.2 when NASA CMAPSS data is ready."
    )


def test_500_sku_no_nulls(scorer):
    """Run on 500 synthetic SKUs — no NaN in output columns."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame(
        {
            "item_id": [f"SKU{i:04d}" for i in range(n)],
            "depot_tier": np.random.choice(
                ["Forward", "Border", "Rear"],
                size=n,
                p=[0.35, 0.40, 0.25],
            ),
            "equipment_density_score": np.random.uniform(0, 1, n),
            "environment_multiplier": np.ones(n),
        }
    )
    result = scorer.score_sku_master(df)
    assert result["location_score_adj"].isna().sum() == 0
    assert result["base_position_score"].isna().sum() == 0
