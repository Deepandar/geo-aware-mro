import pandas as pd


def classify_abc(
    df: pd.DataFrame,
    cut_a: float = 0.80,
    cut_b: float = 0.95,
) -> pd.DataFrame:

    df = df.copy()

    if "annual_consumption_value" not in df.columns:

        if "demand" in df.columns:
            demand_col = "demand"
        elif "demand_mean" in df.columns:
            demand_col = "demand_mean"
        else:
            raise ValueError(
                "Missing required column: demand or demand_mean"
            )

        if "unit_cost" not in df.columns:
            raise ValueError(
                "Missing required column: unit_cost"
            )

        df["annual_consumption_value"] = (
            df["unit_cost"]
            * df[demand_col]
        )

    total = df["annual_consumption_value"].sum()

    if total == 0:
        df["abc_class"] = "C"
        return df

    df = df.sort_values(
        by="annual_consumption_value",
        ascending=False,
    )

    df["cum_pct"] = (
        df["annual_consumption_value"].cumsum()
        / total
    )

    def assign_class(value: float) -> str:
        if value <= cut_a:
            return "A"
        if value <= cut_b:
            return "B"
        return "C"

    df["abc_class"] = df["cum_pct"].apply(assign_class)

    return df


compute_abc = classify_abc