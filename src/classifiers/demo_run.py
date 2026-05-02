"""Demo: Apply all classifiers to synthetic data and validate 27-cell taxonomy"""
import pandas as pd
import numpy as np
from src.classifiers.abc_classifier import compute_abc
from src.classifiers.ved_classifier import compute_ved
from src.classifiers.fns_classifier import compute_fns
from src.classifiers.ci_index import compute_ci, CiWeights


def generate_synthetic_inventory(n_items: int = 500) -> pd.DataFrame:
    np.random.seed(42)

    categories = ["Safety", "Electrical", "Rotating", "Standard", "Consumable"]

    df = pd.DataFrame({
        "sku_id": [f"SKU{i:04d}" for i in range(n_items)],
        "demand_mean": np.random.exponential(scale=20, size=n_items),
        "unit_cost": np.random.uniform(5, 500, size=n_items),
        "equipment_category": np.random.choice(categories, size=n_items),
        "equipment_density_score": np.random.uniform(0, 1, size=n_items),
        "adi": np.random.uniform(0.5, 5.0, size=n_items),
        "cv2": np.random.uniform(0.1, 2.0, size=n_items),
        "geo_risk_score": np.random.uniform(0, 1, size=n_items),
    })

    return df


def main():
    print("Generating synthetic inventory...")
    df = generate_synthetic_inventory(500)

    print("\nApplying classifiers...")
    df = compute_abc(df)
    df = compute_ved(df)
    df = compute_fns(df)

    weights = CiWeights(w_abc=0.25, w_ved=0.25, w_fns=0.25, w_geo=0.25)
    df = compute_ci(df, weights)

    print("\n=== Classification Results ===")
    print(f"Total items: {len(df)}")
    print(f"\nABC distribution:\n{df['abc_class'].value_counts().sort_index()}")
    print(f"\nVED distribution:\n{df['ved_class'].value_counts().sort_index()}")
    print(f"\nFNS distribution:\n{df['fns_class'].value_counts().sort_index()}")

    print(f"\n=== 27-Cell Taxonomy ===")
    cell_counts = df['cell_27'].value_counts().sort_index()
    print(f"Unique cells populated: {len(cell_counts)} / 27")
    print("\nTop 10 cells by count:")
    print(cell_counts.head(10))

    print(f"\n=== Composite Index (Ci) ===")
    print(f"Ci range: [{df['ci'].min():.3f}, {df['ci'].max():.3f}]")
    print(f"Ci mean: {df['ci'].mean():.3f}")
    print(f"Ci std: {df['ci'].std():.3f}")

    print(f"\n=== Forecast Method Routing ===")
    print(df['forecast_method'].value_counts())

    assert len(cell_counts) >= 20, f"Expected >= 20 cells, got {len(cell_counts)}"
    assert 0.35 <= df['ci'].mean() <= 0.65, f"Ci mean {df['ci'].mean()} outside [0.35, 0.65]"

    print("\n✅ All validations passed")
    print(f"✅ {len(cell_counts)}/27 cells populated")
    print(f"✅ Ci mean: {df['ci'].mean():.3f} (target: 0.35-0.65)")


if __name__ == "__main__":
    main()
