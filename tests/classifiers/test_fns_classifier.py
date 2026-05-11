"""Tests for FNS classifier"""

import pytest
import pandas as pd
from src.classifiers.fns_classifier import compute_fns


@pytest.mark.parametrize(
    "adi,cv2,expected_class,expected_method",
    [
        (1.0, 0.3, "Smooth", "holt_winters"),
        (1.31, 0.48, "Smooth", "holt_winters"),
        (1.0, 0.5, "Erratic", "arima"),
        (1.31, 1.0, "Erratic", "arima"),
        (1.32, 0.3, "Intermittent", "croston"),
        (2.0, 0.48, "Intermittent", "croston"),
        (1.32, 0.49, "Lumpy", "sba"),
        (2.0, 1.0, "Lumpy", "sba"),
        (5.0, 2.0, "Lumpy", "sba"),
    ],
)
def test_fns_quadrants(adi, cv2, expected_class, expected_method):
    """Test all four Syntetos-Boylan quadrants"""
    df = pd.DataFrame({"adi": [adi], "cv2": [cv2]})

    result = compute_fns(df)

    assert result["fns_class"].iloc[0] == expected_class
    assert result["forecast_method"].iloc[0] == expected_method


def test_fns_boundary_precision():
    """Test exact boundary values"""
    df = pd.DataFrame(
        {"adi": [1.32, 1.32, 1.31, 1.31], "cv2": [0.49, 0.48, 0.49, 0.48]}
    )

    result = compute_fns(df)

    assert result["fns_class"].iloc[0] == "Lumpy"
    assert result["fns_class"].iloc[1] == "Intermittent"
    assert result["fns_class"].iloc[2] == "Erratic"
    assert result["fns_class"].iloc[3] == "Smooth"


def test_fns_all_quadrants_present():
    """Test that all four classes can be produced"""
    df = pd.DataFrame({"adi": [1.0, 1.0, 2.0, 2.0], "cv2": [0.3, 0.6, 0.3, 0.6]})

    result = compute_fns(df)

    classes = set(result["fns_class"])
    assert classes == {"Smooth", "Erratic", "Intermittent", "Lumpy"}
