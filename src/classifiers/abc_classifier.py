"""ABC Classification using Pareto principle (80-15-5 rule)"""
import pandas as pd


def compute_abc(
    df: pd.DataFrame,
    demand_col: str = "demand_mean",
    cost_col: str = "unit_cost",
    cut_a: float = 0.80,
    cut_b: float = 0.95
) -> pd.DataFrame:
    """
    Classify items into A/B/C based on cumulative annual value.
    
    Args:
        df: DataFrame with demand and cost columns
        demand_col: Column name for mean weekly demand
        cost_col: Column name for unit cost
        cut_a: Cumulative % threshold for A class (default 0.80)
        cut_b: Cumulative % threshold for B class (default 0.95)
    
    Returns:
        DataFrame with abc_class column added
    """
    df = df.copy()
    
    # Annual value = weekly demand × 52 weeks × unit cost
    df["annual_value"] = df[demand_col] * 52 * df[cost_col]
    
    # Sort descending by value
    df = df.sort_values("annual_value", ascending=False).reset_index(drop=True)
    
    # Cumulative percentage
    df["cum_value"] = df["annual_value"].cumsum()
    total_value = df["annual_value"].sum()
    
    if total_value == 0:
        df["abc_class"] = "C"
        return df
    
    df["cum_pct"] = df["cum_value"] / total_value
    
    # Classify
    def assign_class(pct):
        if pct <= cut_a:
            return "A"
        elif pct <= cut_b:
            return "B"
        return "C"
    
    df["abc_class"] = df["cum_pct"].apply(assign_class)
    
    return df
