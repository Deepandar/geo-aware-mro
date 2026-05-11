"""Tests for VED classifier"""

import pytest
import pandas as pd
from src.classifiers.ved_classifier import compute_ved, CRITICAL_CATEGORIES


def test_ved_critical_override():
    """Test that critical categories override to E"""
    df = pd.DataFrame(
        {
            "equipment_category": list(CRITICAL_CATEGORIES),
            "equipment_density_score": [0.1, 0.1, 0.1],
        }
    )

    result = compute_ved(df)

    assert all(result["ved_class"] == "E")


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.9, "V"),
        (0.7, "V"),
        (0.69, "E"),
        (0.5, "E"),
        (0.4, "E"),
        (0.39, "D"),
        (0.1, "D"),
    ],
)
def test_ved_density_thresholds(score, expected):
    """Test VED classification thresholds"""
    df = pd.DataFrame(
        {"equipment_category": ["Standard"], "equipment_density_score": [score]}
    )

    result = compute_ved(df)

    assert result["ved_class"].iloc[0] == expected


def test_ved_mixed_categories():
    """Test VED with mix of critical and non-critical"""
    df = pd.DataFrame(
        {
            "equipment_category": ["Safety", "Standard", "Electrical"],
            "equipment_density_score": [0.2, 0.8, 0.3],
        }
    )

    result = compute_ved(df)

    assert result["ved_class"].iloc[0] == "E"
    assert result["ved_class"].iloc[1] == "V"
    assert result["ved_class"].iloc[2] == "E"
