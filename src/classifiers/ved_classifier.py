"""VED Classification: Vital, Essential, Desirable based on criticality"""
import pandas as pd


CRITICAL_CATEGORIES = {"Safety", "Electrical", "Rotating"}


def compute_ved(
    df: pd.DataFrame,
    category_col: str = "equipment_category",
    density_col: str = "equipment_density_score"
) -> pd.DataFrame:
    """
    Classify items into V/E/D based on equipment criticality.
    
    Args:
        df: DataFrame with equipment category and density score
        category_col: Column name for equipment category
        density_col: Column name for density score [0,1]
    
    Returns:
        DataFrame with ved_class column added
    """
    df = df.copy()
    
    def classify_row(row):
        # Critical categories override to Essential
        if row[category_col] in CRITICAL_CATEGORIES:
            return "E"
        
        score = row[density_col]
        
        if score >= 0.7:
            return "V"
        elif score >= 0.4:
            return "E"
        else:
            return "D"
    
    df["ved_class"] = df.apply(classify_row, axis=1)
    
    return df
