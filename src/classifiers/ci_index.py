"""Composite Criticality Index (Ci) - weighted score across ABC/VED/FNS/Geo"""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class CiWeights:
    """Weights for composite index - must sum to 1.0"""
    w_abc: float
    w_ved: float
    w_fns: float
    w_geo: float
    
    def __post_init__(self):
        total = self.w_abc + self.w_ved + self.w_fns + self.w_geo
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


# Scoring maps
ABC_MAP = {"A": 1.0, "B": 0.6, "C": 0.2}
VED_MAP = {"V": 1.0, "E": 0.6, "D": 0.2}
FNS_MAP = {"Smooth": 1.0, "Erratic": 0.7, "Intermittent": 0.5, "Lumpy": 0.3}


def compute_ci(
    df: pd.DataFrame,
    weights: CiWeights,
    abc_col: str = "abc_class",
    ved_col: str = "ved_class",
    fns_col: str = "fns_class",
    geo_col: str = "geo_risk_score"
) -> pd.DataFrame:
    """
    Compute composite criticality index (Ci) and 27-cell taxonomy.
    
    Args:
        df: DataFrame with ABC/VED/FNS classifications and geo risk
        weights: CiWeights instance defining dimension weights
        abc_col: Column name for ABC class
        ved_col: Column name for VED class
        fns_col: Column name for FNS class
        geo_col: Column name for geo risk score [0,1]
    
    Returns:
        DataFrame with ci, cell_27, and score columns added
    """
    df = df.copy()
    
    # Map categorical classes to scores
    df["abc_score"] = df[abc_col].map(ABC_MAP)
    df["ved_score"] = df[ved_col].map(VED_MAP)
    df["fns_score"] = df[fns_col].map(FNS_MAP)
    
    # Compute weighted composite index
    df["ci"] = (
        weights.w_abc * df["abc_score"] +
        weights.w_ved * df["ved_score"] +
        weights.w_fns * df["fns_score"] +
        weights.w_geo * df[geo_col]
    )
    
    # 27-cell taxonomy: ABC_VED_FNS
    df["cell_27"] = df[abc_col] + df[ved_col] + "_" + df[fns_col]
    
    return df
