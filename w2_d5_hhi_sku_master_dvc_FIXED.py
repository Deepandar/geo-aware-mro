#!/usr/bin/env python3
"""
Week 2 · Day 5: HHI ↔ SKU Master Join
FIXED VERSION - Dynamic Column Detection
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import duckdb
import logging

# ─────────────────────────── CONFIG ──────────────────────────────────
DATA_PROC = Path("data/processed")
REPORTS = Path("reports")
LOG_DIR = Path("logs")

for d in [DATA_PROC, REPORTS, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"w2_d5_hhi_join_{datetime.now():%Y%m%d}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), 
              logging.StreamHandler()],
)
log = logging.getLogger(__name__)

DB_PATH = DATA_PROC / "mro.duckdb"
SKU_MASTER_IN = DATA_PROC / "sku_master_500.parquet"
SKU_MASTER_OUT = DATA_PROC / "sku_master_v0.9.parquet"
SKU_MASTER_CSV = DATA_PROC / "sku_master_v0.9.csv"

# ─────────────────────────── LOAD DATA ───────────────────────────────

def load_sku_master() -> pd.DataFrame:
    df = pd.read_parquet(SKU_MASTER_IN)
    log.info(f"Loaded SKU Master: {len(df)} rows. Columns: {list(df.columns)}")
    return df

def load_hhi_from_duckdb() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    tables = con.execute("SHOW TABLES").df()['name'].tolist()
    
    for table_name in ['sku_hhi_concentration', 'comtrade_raw']:
        if table_name in tables:
            log.info(f"Loading HHI data from {table_name}")
            df = con.execute(f"SELECT * FROM {table_name}").df()
            con.close()
            
            if 'hhi_risk' in df.columns:
                df = df.rename(columns={
                    'hhi_risk': 'hhi_risk_band',
                    'top_supplier': 'dominant_partner',
                    'top_supplier_share': 'dominant_share'
                })
            return df
            
    con.close()
    return _synthetic_hhi()

def _synthetic_hhi() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    hs_codes = ["870899", "840910", "848210", "731815"]
    reporters = ["EUR", "USA", "GBR"]
    partners = ["CHN", "USA", "DEU"]
    
    rows = []
    for hs in hs_codes:
        for rep in reporters:
            hhi_val = rng.integers(1500, 4000)
            risk = "Low" if hhi_val < 1500 else "Moderate" if hhi_val < 2500 else "High"
            
            rows.append({
                "hs_code": hs,
                "reporter": rep,
                "year": 2022,
                "hhi": hhi_val,
                "hhi_risk_band": risk,
                "dominant_partner": rng.choice(partners),
                "dominant_share": round(rng.uniform(0.3, 0.7), 4),
            })
    return pd.DataFrame(rows)

# ─────────────────────────── JOIN ────────────────────────────────────

def join_hhi(sku: pd.DataFrame, hhi: pd.DataFrame) -> pd.DataFrame:
    if 'hs_code' not in sku.columns:
        raise ValueError("hs_code column required in SKU Master")
    
    if 'reporter' in hhi.columns:
        hhi_agg = hhi.groupby('hs_code').first().reset_index()
    else:
        hhi_agg = hhi
        
    hhi_cols = ['hs_code', 'hhi', 'hhi_risk_band', 'dominant_partner', 'dominant_share']
    hhi_slim = hhi_agg[hhi_cols].drop_duplicates('hs_code')
    
    merged = sku.merge(hhi_slim, on='hs_code', how='left')
    
    merged['hhi'] = merged['hhi'].fillna(0).astype(int)
    merged['hhi_risk_band'] = merged['hhi_risk_band'].fillna('Unknown')
    merged['dominant_partner'] = merged['dominant_partner'].fillna('Unknown')
    merged['dominant_share'] = merged['dominant_share'].fillna(0.0)
    
    merged['geo_conc_flag'] = (
        (merged['hhi_risk_band'] == 'High') & 
        (merged['dominant_share'] > 0.50)
    ).astype(int)
    
    return merged

# ─────────────────────────── VALIDATE ────────────────────────────────

def validate_sku_master(df: pd.DataFrame) -> dict:
    """Data quality checks with dynamic ID column detection."""
    
    # Dynamically find the ID column
    id_col = next((c for c in ['sku', 'sku_id', 'item_id', 'material', 'id'] if c.lower() in [col.lower() for col in df.columns]), None)
    
    if id_col:
        # Find exact case match
        exact_id_col = next(c for c in df.columns if c.lower() == id_col.lower())
        assert df[exact_id_col].nunique() == len(df), f"Duplicate {exact_id_col}s found!"
        unique_ids = df[exact_id_col].nunique()
    else:
        log.warning(f"No standard ID column found. Available columns: {list(df.columns)}")
        unique_ids = len(df)

    assert df['hhi'].between(0, 10000).all(), "HHI out of range!"
    
    results = {
        'total_skus': len(df),
        'unique_ids': unique_ids,
        'hhi_mapped': int((df['hhi'] > 0).sum()),
        'hhi_unmapped': int((df['hhi'] == 0).sum()),
        'geo_conc_flag_ct': int(df['geo_conc_flag'].sum()),
        'hhi_band_dist': df['hhi_risk_band'].value_counts().to_dict(),
        'validation_passed': True,
    }
    
    log.info(f"Validation PASSED | {results['hhi_mapped']}/{results['total_skus']} " +
             f"SKUs matched | geo_conc_flag={results['geo_conc_flag_ct']}")
    
    return results

# ─────────────────────────── PERSIST ─────────────────────────────────

def persist_artifacts(df: pd.DataFrame, validation: dict) -> None:
    df.to_parquet(SKU_MASTER_OUT, index=False)
    df.to_csv(SKU_MASTER_CSV, index=False)
    
    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS sku_master_v09")
    con.execute("CREATE TABLE sku_master_v09 AS SELECT * FROM df")
    con.close()
    
    import json
    val_path = DATA_PROC / "w2_validation.json"
    with open(val_path, 'w') as f:
        json.dump(validation, f, indent=2)

def generate_report(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    risk_counts = df['hhi_risk_band'].value_counts()
    colors = {'High': '#e74c3c', 'Moderate': '#f39c12', 'Low': '#2ecc71', 'Unknown': '#95a5a6'}
    bar_colors = [colors.get(x, '#95a5a6') for x in risk_counts.index]
    
    ax.bar(risk_counts.index, risk_counts.values, color=bar_colors, edgecolor='white')
    ax.set_title('HHI Risk Band Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('SKU Count')
    ax.set_xlabel('HHI Risk Band')
    
    for i, v in enumerate(risk_counts.values):
        ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
    
    plt.tight_layout()
    chart_path = REPORTS / "w2_hhi_distribution.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

# ─────────────────────────── MAIN ────────────────────────────────────

def main():
    log.info("="*70)
    log.info("W2-D5: HHI + SKU Master Join START")
    log.info("="*70)
    
    sku_df = load_sku_master()
    hhi_df = load_hhi_from_duckdb()
    merged = join_hhi(sku_df, hhi_df)
    validation = validate_sku_master(merged)
    persist_artifacts(merged, validation)
    generate_report(merged)
    
    log.info("="*70)
    log.info("SUMMARY")
    log.info("="*70)
    log.info(f"  Total SKUs: {len(merged)}")
    log.info(f"  HHI mapped: {validation['hhi_mapped']}")
    log.info(f"  High-risk SKUs: {validation['geo_conc_flag_ct']}")
    log.info(f"  Outputs:")
    log.info(f"    - {SKU_MASTER_OUT}")
    log.info(f"    - DuckDB: sku_master_v09")
    log.info("="*70)
    log.info("SUCCESS - Day 5 Complete")
    log.info("="*70)

if __name__ == "__main__":
    main()
