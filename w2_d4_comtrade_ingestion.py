from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import duckdb

RAW_DIR = Path("data/raw/COMTRADE")
OUTPUT_DIR = Path("data/external/comtrade_raw")
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs")

# Ensure directories exist
for d in [OUTPUT_DIR, PROCESSED_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def parse_comtrade_csv(csv_path: Path) -> pd.DataFrame:
    print(f"Reading {csv_path}...")

    # index_col=False is mandatory here to prevent the 'extra comma' shift
    df = pd.read_csv(
        csv_path, 
        low_memory=False, 
        encoding='utf-8-sig', 
        index_col=False, 
        skipinitialspace=True
    )

    # Clean up column names (handles whitespace and trailing commas)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Validation check for the shift bug
    sample_cmd = df['cmdCode'].iloc[0]
    sample_flow = df['flowDesc'].iloc[0]
    print(f"\nVerification:")
    print(f"  cmdCode: {sample_cmd} (Should be numeric code)")
    print(f"  flowDesc: {sample_flow} (Should be 'Import' or 'Export')")
    
    # Map to internal schema
    df_clean = pd.DataFrame({
        "hs_code": df["cmdCode"].astype(str),
        "reporter": df["reporterISO"].astype(str),
        "partner": df["partnerISO"].astype(str),
        "year": pd.to_numeric(df["refYear"], errors="coerce"),
        "trade_value_usd": pd.to_numeric(df["primaryValue"], errors="coerce"),
        "flow_desc": df["flowDesc"].astype(str),
        "commodity_desc": df["cmdDesc"].astype(str),
    })

    # Case-insensitive filter
    df_clean = df_clean[
        (df_clean["flow_desc"].str.strip().str.upper() == "IMPORT") &
        (df_clean["trade_value_usd"] > 0)
    ].copy()

    print(f"Cleaned records: {len(df_clean)}")
    return df_clean

def compute_hhi(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    
    latest_year = df["year"].max()
    df_latest = df[df["year"] == latest_year].copy()
    
    synth_rows = []
    rng = np.random.default_rng(42)
    partners = ["CHN", "USA", "DEU"]

    for _, row in df_latest.iterrows():
        shares = rng.dirichlet([5, 3, 2])
        for p, s in zip(partners, shares):
            synth_rows.append({
                "hs_code": row["hs_code"],
                "reporter": row["reporter"],
                "partner": p,
                "year": row["year"],
                "trade_value_usd": row["trade_value_usd"] * s,
            })

    df_synth = pd.DataFrame(synth_rows)
    hhi_records = []
    
    for (hs_code, reporter), group in df_synth.groupby(["hs_code", "reporter"]):
        total_imports = group["trade_value_usd"].sum()
        shares = group["trade_value_usd"] / total_imports
        hhi = (shares ** 2).sum() * 10000

        risk = "Low" if hhi < 1500 else "Moderate" if hhi < 2500 else "High"
        idx_max = group["trade_value_usd"].idxmax()
        
        hhi_records.append({
            "hs_code": hs_code,
            "reporter": reporter,
            "year": int(latest_year),
            "hhi": round(hhi, 2),
            "hhi_risk": risk,
            "n_suppliers": len(group),
            "total_import_value_usd": round(total_imports, 2),
            "top_supplier": group.loc[idx_max, "partner"],
            "top_supplier_share": round((group.loc[idx_max, "trade_value_usd"] / total_imports) * 100, 2),
            "_is_synthetic_hhi": True,
        })

    return pd.DataFrame(hhi_records)

def main():
    print("=" * 60)
    print("Day 9: COMTRADE Ingestion + HHI Concentration Analysis")
    print("=" * 60)

    csv_file = RAW_DIR / "TradeData_5_1_2026_22_54_19.csv"
    if not csv_file.exists():
        # Fallback for local testing if directory structure differs
        csv_file = Path("TradeData_5_1_2026_22_54_19.csv")

    comtrade_df = parse_comtrade_csv(csv_file)

    # Save cleaned data
    clean_file = OUTPUT_DIR / f"comtrade_hs_{datetime.now().strftime('%Y%m%d')}.parquet"
    comtrade_df.to_parquet(clean_file, index=False)

    hhi_df = compute_hhi(comtrade_df)
    
    # DuckDB Persistence
    db_file = PROCESSED_DIR / "mro.duckdb"
    con = duckdb.connect(str(db_file))
    
    if not comtrade_df.empty:
        con.execute("CREATE OR REPLACE TABLE comtrade_raw AS SELECT * FROM comtrade_df")
    if not hhi_df.empty:
        con.execute("CREATE OR REPLACE TABLE sku_hhi_concentration AS SELECT * FROM hhi_df")
        print("\nTop Concentrated Markets:")
        print(con.execute("SELECT reporter, hs_code, hhi, hhi_risk FROM sku_hhi_concentration ORDER BY hhi DESC LIMIT 5").df())

    con.close()
    print("\n✅ Day 9 Process Complete")

if __name__ == "__main__":
    main()
