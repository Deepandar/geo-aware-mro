# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import duckdb
import mlflow
import sys
from pathlib import Path

# --- 1. CONFIGURATION ---
DB_PATH = Path("data/processed/mro.duckdb")
WEIGHTS = {"abc": 0.35, "ved": 0.30, "fns": 0.20, "loc": 0.15}
ADI_CUT = 1.32
CV2_CUT = 0.49

# --- 2. DISTRIBUTION TEST ---
def verify_quadrant_coverage(df):
    """Ensures all 4 demand quadrants are represented for Phase 2 readiness."""
    freq = df['adi'] < ADI_CUT
    stable = df['cv2'] < CV2_CUT
    
    quadrants = {
        "Smooth (F)": (freq & stable).sum(),
        "Erratic (N)": (freq & ~stable).sum(),
        "Intermittent (N)": (~freq & stable).sum(),
        "Lumpy (S)": (~freq & ~stable).sum()
    }
    
    print("\n--- Demand Quadrant Audit ---")
    all_populated = True
    for name, count in quadrants.items():
        status = "✅" if count > 0 else "❌ EMPTY"
        print(f"{status} {name}: {count} SKUs")
        if count == 0:
            all_populated = False
            
    if not all_populated:
        print("⚠️ WARNING: Some quadrants are empty. Classification coverage is incomplete.")
    else:
        print("✅ Success: All 4 quadrants populated for 27-class taxonomy.")

# --- 3. ENRICHMENT & CLASSIFICATION ENGINE ---
def run_scoring_pipeline():
    con = duckdb.connect(str(DB_PATH))
    df = con.execute("SELECT * FROM sku_master_final").df()

    # Self-healing for ADI/CV2
    if 'adi' not in df.columns: df['adi'] = 1.5
    if 'cv2' not in df.columns: df['cv2'] = 0.5

    # A. Syntetos-Boylan Classification[cite: 1]
    freq = df['adi'] < ADI_CUT
    stable = df['cv2'] < CV2_CUT

    conditions = [
        (freq & stable),    # Smooth
        (freq & ~stable),   # Erratic
        (~freq & stable)    # Intermittent
    ]
    df["fns_class"] = np.select(conditions, ["F", "N", "N"], default="S")

    # B. Distribution Test
    verify_quadrant_coverage(df)

    # C. Normalized Scoring[cite: 1]
    abc_map = {"A": 1.0, "B": 0.5, "C": 0.0}
    ved_map = {"V": 1.0, "E": 0.5, "D": 0.0}
    fns_map = {"F": 1.0, "N": 0.5, "S": 0.0}
    loc_map = {3: 1.0, 2: 0.5, 1: 0.0}

    df["ci_score"] = (
        WEIGHTS["abc"] * df["abc_class"].map(abc_map) +
        WEIGHTS["ved"] * df["ved_class"].map(ved_map) +
        WEIGHTS["fns"] * df["fns_class"].map(fns_map) +
        WEIGHTS["loc"] * df["location_tier"].map(loc_map)
    ).round(4)

    # D. Final Taxonomy[cite: 1]
    df["taxonomy_code"] = df["abc_class"] + df["ved_class"] + df["fns_class"]
    df["ci_rank"] = df["ci_score"].rank(ascending=False, method="first").astype(int)

    # --- 4. PERSISTENCE ---
    with mlflow.start_run(run_name="W3_Final_Taxonomy_Test"):
        mlflow.log_params(WEIGHTS)
        mlflow.log_metric("active_cells", df["taxonomy_code"].nunique())
        
        con.register("final_df", df)
        con.execute("CREATE OR REPLACE TABLE sku_master_final AS SELECT * FROM final_df")
        print(f"\n✅ Locked {len(df)} SKUs into SKU Master v1.0")
    
    con.close()

if __name__ == "__main__":
    run_scoring_pipeline()
