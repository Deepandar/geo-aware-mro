import pandas as pd


def classify_fns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "cv2" in df.columns:
        df["cv_squared"] = df["cv2"]

    required = {"adi", "cv_squared"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    conditions = [
        ((df["adi"] < 1.32) & (df["cv_squared"] < 0.49)),
        ((df["adi"] < 1.32) & (df["cv_squared"] >= 0.49)),
        ((df["adi"] >= 1.32) & (df["cv_squared"] < 0.49)),
        ((df["adi"] >= 1.32) & (df["cv_squared"] >= 0.49)),
    ]

    labels = [
        ("Smooth", "holt_winters"),
        ("Erratic", "arima"),
        ("Intermittent", "croston"),
        ("Lumpy", "sba"),
    ]

    df["fns_class"] = None
    df["forecast_method"] = None

    for condition, (cls, method) in zip(
        conditions,
        labels,
    ):
        df.loc[condition, "fns_class"] = cls
        df.loc[condition, "forecast_method"] = method

    return df


compute_fns = classify_fns
