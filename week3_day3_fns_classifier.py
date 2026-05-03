# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import duckdb
import mlflow
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURATION ---
DB_PATH = Path("data/processed/mro.duckdb")
DATA_DIR = Path("data/processed")
SB_ADI_CUT = 1.32 
SB_CV2_CUT = 0.49 

# 1. LOAD SKU MASTER & CLEAN OLD COLUMNS
def load_and_clean_master(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    df = con.execute("SELECT * FROM sku_master_final").df()
    con.close()
    
    # Drop columns from previous runs to avoid merge conflicts
    cols_to_drop = [
        'fns_class', 'sb_quadrant', 'adi', 'cv2', 'n_demand_periods', 'n_periods',
        'fns_class_x', 'fns_class_y', 'sb_quadrant_x', 'sb_quadrant_y',
        'adi_x', 'adi_y', 'cv2_x', 'cv2_y', 'n_demand_periods_x', 'n_demand_periods_y',
        'n_periods_x', 'n_periods_y'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    return df

# 2A. PRIMARY: COMPUTE FROM ACTUAL TRANSACTION HISTORY
def _history_table_exists(con: duckdb.DuckDBPyConnection) -> bool:
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    return "transaction_history" in tables

def compute_adi_cv2_from_history(db_path: Path = DB_PATH) -> pd.DataFrame:
    con = duckdb.connect(str(db_path))
    query = """
    WITH nonzero AS (
        SELECT item_id, period, quantity,
               LAG(period) OVER (PARTITION BY item_id ORDER BY period) AS prev_period
        FROM transaction_history
        WHERE quantity > 0
    ),
    intervals AS (
        SELECT item_id,
               COUNT(*) AS n_demand_periods,
               AVG(period - prev_period) AS adi,
               CASE WHEN AVG(quantity) > 0 
                    THEN VARIANCE(quantity) / (AVG(quantity) * AVG(quantity)) 
                    ELSE 0 END AS cv2
        FROM nonzero
        WHERE prev_period IS NOT NULL
        GROUP BY item_id
    ),
    total_periods AS (
        SELECT item_id, COUNT(DISTINCT period) AS n_periods
        FROM transaction_history
        GROUP BY item_id
    )
    SELECT i.item_id, 
           COALESCE(i.adi, 99.0) AS adi, 
           COALESCE(i.cv2, 1.0) AS cv2,
           COALESCE(i.n_demand_periods, 0) AS n_demand_periods,
           t.n_periods
    FROM total_periods t
    LEFT JOIN intervals i USING (item_id)
    """
    df = con.execute(query).df()
    con.close()
    return df

# 2B. FALLBACK: SYNTHETIC GENERATOR
def generate_diverse_demand(df_master: pd.DataFrame, n_periods: int = 52) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    records = []
    
    for _, row in df_master.iterrows():
        lam = max(row["demand"] / n_periods, 0.05)
        qty = rng.poisson(lam=lam, size=n_periods).astype(float)
        
        # Sparsity linked to ABC class
        sparsity = rng.uniform(0.4, 0.85) if row.get("abc_class") == 'C' else rng.uniform(0.05, 0.2)
        qty[rng.random(n_periods) < sparsity] = 0
        
        # Variability
        nz_mask = qty > 0
        if nz_mask.any():
            qty[nz_mask] *= rng.uniform(0.5, 3.0, size=nz_mask.sum())
        
        nz_idx = np.where(qty > 0)[0]
        n_demand = len(nz_idx)
        
        if n_demand < 2:
            adi, cv2 = float(n_periods), 1.0
        else:
            adi = float(np.diff(nz_idx).mean())
            nz_qty = qty[nz_idx]
            cv2 = float(nz_qty.var() / (nz_qty.mean() ** 2)) if nz_qty.mean() > 0 else 1.0
            
        records.append({
            "item_id": row["item_id"], "adi": round(adi, 3), "cv2": round(cv2, 4),
            "n_demand_periods": int(n_demand), "n_periods": n_periods
        })
    return pd.DataFrame(records)

# 3. CLASSIFICATION LOGIC
def assign_fns(df: pd.DataFrame) -> pd.DataFrame:
    freq = df["adi"] < SB_ADI_CUT
    stable = df["cv2"] < SB_CV2_CUT
    
    conditions = [(freq & stable), (~freq & stable), (freq & ~stable)]
    df["sb_quadrant"] = np.select(conditions, ["smooth", "intermittent", "erratic"], default="lumpy")
    
    mapping = {"smooth": "F", "intermittent": "N", "erratic": "N", "lumpy": "S"}
    df["fns_class"] = df["sb_quadrant"].map(mapping)
    return df

# 4. RUN PIPELINE
if __name__ == "__main__":
    print("-" * 60 + "\nWEEK 3 DAY 3: FNS CLASSIFIER (HISTORY AWARE)\n" + "-" * 60)
    
    df_master = load_and_clean_master()
    print(f"Loaded {len(df_master)} SKUs. Old FNS columns cleared.")
    
    # Decide source
    con_check = duckdb.connect(str(DB_PATH))
    has_history = _history_table_exists(con_check)
    con_check.close()
    
    if has_history:
        print("✅ Found 'transaction_history' table. Computing real ADI/CV2...")
        df_metrics = compute_adi_cv2_from_history()
    else:
        print("⚠️ 'transaction_history' not found. Using diverse synthetic fallback...")
        df_metrics = generate_diverse_demand(df_master)
        
    df_metrics = assign_fns(df_metrics)
    
    # Merge and Save
    df_final = df_master.merge(df_metrics, on="item_id", how="left")
    
    print("\nFNS Distribution:")
    print(df_final["fns_class"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
    
    con = duckdb.connect(str(DB_PATH))
    con.register("df_final_view", df_final)
    con.execute("CREATE OR REPLACE TABLE sku_master_final AS SELECT * FROM df_final_view")
    con.close()
    
    df_final.to_csv(DATA_DIR / "sku_master_abc_ved_fns.csv", index=False)
    print("\nSUCCESS: Database updated and CSV saved.")
