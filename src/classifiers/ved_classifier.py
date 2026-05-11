import pandas as pd

CRITICAL_CATEGORIES = {
    "Safety",
    "Electrical",
    "Critical",
}


def classify_ved(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if "equipment_density_score" not in df.columns:
        raise ValueError("Missing required column: equipment_density_score")

    def assign(row):

        category = row.get(
            "equipment_category",
            "",
        )

        score = row["equipment_density_score"]

        # Critical categories override to E
        if category in CRITICAL_CATEGORIES:
            return "E"

        if score >= 0.70:
            return "V"

        if score >= 0.40:
            return "E"

        return "D"

    df["ved_class"] = df.apply(
        assign,
        axis=1,
    )

    return df


compute_ved = classify_ved
