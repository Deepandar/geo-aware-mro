"""FNS Classification: Fast/Normal/Slow based on ADI and CV²
Syntetos-Boylan quadrant classification"""
import pandas as pd


def compute_fns(
    df: pd.DataFrame,
    adi_col: str = "adi",
    cv2_col: str = "cv2"
) -> pd.DataFrame:
    """
    Classify demand patterns using Syntetos-Boylan quadrants.
    
    ADI (Average Demand Interval): avg days between non-zero demands
    CV² (Coefficient of Variation squared): (std/mean)²
    
    Quadrants:
    - Smooth: ADI < 1.32, CV² < 0.49 → Holt-Winters
    - Erratic: ADI < 1.32, CV² ≥ 0.49 → ARIMA
    - Intermittent: ADI ≥ 1.32, CV² < 0.49 → Croston
    - Lumpy: ADI ≥ 1.32, CV² ≥ 0.49 → SBA
    
    Args:
        df: DataFrame with ADI and CV² columns
        adi_col: Column name for average demand interval
        cv2_col: Column name for CV²
    
    Returns:
        DataFrame with fns_class and forecast_method columns
    """
    df = df.copy()
    
    def classify_row(row):
        adi = row[adi_col]
        cv2 = row[cv2_col]
        
        if adi < 1.32 and cv2 < 0.49:
            return "Smooth", "holt_winters"
        elif adi < 1.32:
            return "Erratic", "arima"
        elif cv2 < 0.49:
            return "Intermittent", "croston"
        else:
            return "Lumpy", "sba"
    
    result = df.apply(classify_row, axis=1, result_type="expand")
    df["fns_class"] = result[0]
    df["forecast_method"] = result[1]
    
    return df
