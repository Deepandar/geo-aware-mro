"""Tests for ABC classifier"""
import pandas as pd
from src.classifiers.abc_classifier import compute_abc


def test_abc_basic_classification():
    """Test basic ABC classification with known distribution"""
    df = pd.DataFrame({
        "demand_mean": [100, 50, 20, 10, 5],
        "unit_cost": [10, 10, 10, 10, 10]
    })
    
    result = compute_abc(df)
    
    assert "abc_class" in result.columns
    assert set(result["abc_class"]) <= {"A", "B", "C"}
    assert result["abc_class"].iloc[0] == "A"


def test_abc_zero_total_value():
    """Test ABC when total value is zero"""
    df = pd.DataFrame({
        "demand_mean": [0, 0, 0],
        "unit_cost": [10, 10, 10]
    })
    
    result = compute_abc(df)
    
    assert all(result["abc_class"] == "C")


def test_abc_custom_thresholds():
    """Test ABC with custom cut-off points"""
    df = pd.DataFrame({
        "demand_mean": [100, 50, 20],
        "unit_cost": [10, 10, 10]
    })
    
    result = compute_abc(df, cut_a=0.5, cut_b=0.8)
    
    assert "abc_class" in result.columns


def test_abc_single_item():
    """Test ABC classification with single item - gets C due to 100% cumulative"""
    df = pd.DataFrame({"demand_mean": [100], "unit_cost": [10]})
    result = compute_abc(df)
    # Single item has 100% cumulative value, falls in C class (> 0.95)
    assert result["abc_class"].iloc[0] == "C"


def test_abc_pareto_distribution():
    """Test that ABC follows expected Pareto pattern"""
    df = pd.DataFrame({
        "demand_mean": [100, 80, 60, 40, 20, 10, 5, 2, 1, 1],
        "unit_cost": [10] * 10
    })
    
    result = compute_abc(df)
    
    # Top items should be A
    assert result["abc_class"].iloc[0] == "A"
    assert result["abc_class"].iloc[1] == "A"
    # Bottom items should be C
    assert result["abc_class"].iloc[-1] == "C"
