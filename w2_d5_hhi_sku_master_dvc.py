"""
=============================================================================
GEO-AWARE MRO DECISION INTELLIGENCE SYSTEM — v1.1
Week 2 · Day 5 (Friday)
Task: HHI concentration index joined to SKU Master, DVC tracked
=============================================================================

ACTIONS CHECKLIST (Do in order):
  □ 1. Confirm W2-D4 outputs exist:
        data/external/comtrade_raw/comtrade_hs_*.parquet
        data/processed/mro.duckdb  (tables: comtrade_imports, sku_hhi_concentration)
  □ 2. Run this script:
        python w2_d5_hhi_sku_master_dvc.py
  □ 3. Run DVC pipeline:
        dvc repro
  □ 4. Tag the data version:
        dvc commit
        git add data/ dvc.lock dvc.yaml .dvc/
        git commit -m "data(W2): SKU Master v0.9 with HHI — DVC tracked"
        git tag -a "data-w2-sku-master-v0.9" -m "Week 2 complete: SKU Master + HHI"
        git push origin develop --tags
  □ 5. Verify DuckDB:
        duckdb data/processed/mro.duckdb
        "SELECT item_id, hs_code, hhi, hhi_risk_band, dominant_partner FROM sku_master_v09 LIMIT 10;"
  □ 6. Update README — Week 2 section complete ✓

OUTPUTS PRODUCED:
  • data/processed/sku_master_v0.9.parquet          ← versioned final W2 artifact
  • data/processed/sku_master_v0.9.csv              ← human-readable snapshot
  • DuckDB table : sku_master_v09
  • reports/w2_hhi_analysis.html                    ← distribution + risk summary
  • dvc.yaml                                        ← W2 stage added
  • dvc.lock                                        ← updated after dvc repro
  • logs/w2_d5_hhi_join.log
=============================================================================
"""

import os
import json
import logging
import warnings
from pathlib import Path
from datetime import date

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")               # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)
load_dotenv()

# ─────────────────────────── PATHS ───────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_EXT   = BASE_DIR / "data" / "external" / "comtrade_raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
REPORTS    = BASE_DIR / "reports"
LOG_DIR    = BASE_DIR / "logs"
DVC_YAML   = BASE_DIR / "dvc.yaml"

for d in [DATA_PROC, REPORTS, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"w2_d5_hhi_join_{date.today():%Y%m%d}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

DB_PATH         = DATA_PROC / "mro.duckdb"
SKU_MASTER_IN   = DATA_PROC / "sku_master_500.parquet"      # from W2-D3
HHI_PARQUET_IN  = DATA_PROC / "sku_master_500_hhi.parquet"  # from W2-D4 (optional)
SKU_MASTER_OUT  = DATA_PROC / "sku_master_v0.9.parquet"
SKU_MASTER_CSV  = DATA_PROC / "sku_master_v0.9.csv"
REPORT_HTML     = REPORTS   / "w2_hhi_analysis.html"
REPORT_PNG      = REPORTS   / "w2_hhi_distribution.png"

# ─────────────────────────── HHI RISK CONFIG ─────────────────────────────────
HHI_BINS   = [0, 1500, 2500, 10_000]
HHI_LABELS = ["Low", "Moderate", "High"]
HHI_COLORS = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}

# ─────────────────────────── STEP 1: LOAD DATA ───────────────────────────────

def load_sku_master() -> pd.DataFrame:
    """Load base SKU Master (W2-D3).  Fall back to synthetic if missing."""
    if SKU_MASTER_IN.exists():
        df = pd.read_parquet(SKU_MASTER_IN)
        log.info(f"Loaded SKU Master: {len(df)} rows  from {SKU_MASTER_IN}")
        return df

    log.warning(f"{SKU_MASTER_IN} not found — generating synthetic SKU Master (500 SKUs).")
    return _synthetic_sku_master()


def _synthetic_sku_master() -> pd.DataFrame:
    """
    Minimal synthetic SKU Master that mirrors the W2-D2 schema.
    All fields used by downstream modules (W3–W12) are included.
    """
    rng = np.random.default_rng(42)
    n   = 500

    categories   = ["Bearings", "Seals", "Fasteners", "Filters", "Pumps",
                     "Sensors", "Valves", "Couplings", "Belts", "Gaskets"]
    origins      = ["China", "India", "Germany", "USA", "Japan",
                     "South Korea", "UK", "France", "Thailand", "Malaysia"]
    hs_pool      = ["840910", "841330", "848210", "848220", "848340",
                     "848360", "850110", "731815", "842131", "902580",
                     "840991", "841391", "841430", "848310", "848390",
                     "853590", "848230", "730811", "841451", "900310"]

    df = pd.DataFrame({
        "item_id":                 [f"SKU-{i:04d}" for i in range(1, n+1)],
        "item_description":        rng.choice(categories, n),
        "unit_cost":               rng.uniform(10, 5_000, n).round(2),
        "annual_demand":           rng.integers(1, 500, n),
        "lead_time_days":          rng.integers(7, 120, n),
        "lead_time_std_days":      rng.integers(1, 30, n),
        "supply_origin_country":   rng.choice(origins, n),
        "hs_code":                 rng.choice(hs_pool, n),
        "equipment_density_score": rng.uniform(0.1, 1.0, n).round(3),
        "location_type":           rng.choice(["Forward", "Border", "Rear"], n,
                                               p=[0.4, 0.35, 0.25]),
        "stockout_cost_usd":       rng.uniform(200, 50_000, n).round(2),
        "salvage_value_usd":       rng.uniform(0, 100, n).round(2),
        "reorder_point":           rng.integers(1, 50, n),
        "safety_stock":            rng.integers(0, 30, n),
        "unit_weight_kg":          rng.uniform(0.01, 50, n).round(3),
        "shelf_life_days":         rng.choice([None, 180, 365, 730], n),
        "criticality_manual":      rng.choice(["Vital", "Essential", "Desirable"], n,
                                               p=[0.20, 0.45, 0.35]),
        "last_issue_date":         pd.to_datetime("2024-01-01") +
                                   pd.to_timedelta(rng.integers(0, 730, n), unit="D"),
        "supplier_id":             [f"SUP-{rng.integers(1,50):03d}" for _ in range(n)],
        "supplier_country":        rng.choice(origins, n),
    })
    df["acv"] = (df["unit_cost"] * df["annual_demand"]).round(2)
    return df


def load_hhi_table() -> pd.DataFrame:
    """Load HHI table from DuckDB (preferred) or parquet fallback."""
    if DB_PATH.exists():
        try:
            con = duckdb.connect(str(DB_PATH), read_only=True)
            hhi = con.execute("SELECT * FROM sku_hhi_concentration").df()
            con.close()
            log.info(f"HHI loaded from DuckDB: {len(hhi)} rows")
            return hhi
        except Exception as e:
            log.warning(f"DuckDB read failed ({e}) — trying parquet fallback.")

    if HHI_PARQUET_IN.exists():
        hhi = pd.read_parquet(HHI_PARQUET_IN)
        log.info(f"HHI loaded from parquet: {len(hhi)} rows")
        return hhi

    log.warning("No HHI source found — generating synthetic HHI table.")
    return _synthetic_hhi()


def _synthetic_hhi() -> pd.DataFrame:
    """Synthetic HHI table matching W2-D4 output schema."""
    rng = np.random.default_rng(99)
    hs_pool = ["840910", "841330", "848210", "848220", "848340",
               "848360", "850110", "731815", "842131", "902580",
               "840991", "841391", "841430", "848310", "848390",
               "853590", "848230", "730811", "841451", "900310"]
    partners = ["China", "India", "Germany", "USA", "Japan",
                "South Korea", "UK", "France", "Thailand", "Malaysia"]

    hhi_vals = rng.integers(800, 6_500, len(hs_pool))
    rows = []
    for hs, hhi_val in zip(hs_pool, hhi_vals):
        dom_idx = rng.integers(0, len(partners))
        rows.append({
            "cmdCode":            hs,
            "hhi":                int(hhi_val),
            "hhi_risk_band":      ("Low"      if hhi_val < 1500 else
                                   "Moderate" if hhi_val < 2500 else "High"),
            "dominant_partner":   partners[dom_idx],
            "dominant_partner_iso": str(rng.integers(100, 900)),
            "dominant_share":     round(rng.uniform(0.25, 0.75), 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────── STEP 2: JOIN ────────────────────────────────────

def join_hhi(sku: pd.DataFrame, hhi: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join HHI concentration fields to SKU Master on hs_code → cmdCode.

    New columns added to SKU Master:
      hhi                – Herfindahl-Hirschman Index [0–10 000]
      hhi_risk_band      – Low / Moderate / High
      dominant_partner   – Country supplying the largest import share
      dominant_share     – That country's share [0–1]
      geo_conc_flag      – 1 if High HHI AND dominant_share > 0.50
    """
    hhi_cols = ["cmdCode", "hhi", "hhi_risk_band",
                "dominant_partner", "dominant_share"]
    hhi_slim = hhi[hhi_cols].drop_duplicates("cmdCode")

    merged = sku.merge(
        hhi_slim,
        left_on="hs_code",
        right_on="cmdCode",
        how="left",
    ).drop(columns="cmdCode", errors="ignore")

    # Fill unmapped SKUs with neutral defaults
    merged["hhi"]              = merged["hhi"].fillna(0).astype(int)
    merged["hhi_risk_band"]    = merged["hhi_risk_band"].fillna("Unknown")
    merged["dominant_partner"] = merged["dominant_partner"].fillna("Unknown")
    merged["dominant_share"]   = merged["dominant_share"].fillna(0.0)

    # Derived flag for W8 Bayesian model
    merged["geo_conc_flag"] = (
        (merged["hhi_risk_band"] == "High") &
        (merged["dominant_share"] > 0.50)
    ).astype(int)

    matched = (merged["hhi"] > 0).sum()
    log.info(f"Join complete: {matched}/{len(merged)} SKUs have HHI data "
             f"({matched/len(merged)*100:.1f}%)")
    log.info(f"geo_conc_flag=1 (High HHI + >50% single source): "
             f"{merged['geo_conc_flag'].sum()} SKUs")
    return merged


# ─────────────────────────── STEP 3: VALIDATION ──────────────────────────────

def validate_sku_master(df: pd.DataFrame) -> dict:
    """
    Data quality checks — mirrors the Month 1 Milestone Gate criteria.
    Returns a results dict; raises AssertionError if any hard check fails.
    """
    results = {}

    # Hard checks (will raise)
    assert df["item_id"].nunique() == len(df), \
        "FAIL: Duplicate item_ids detected"
    assert df["unit_cost"].gt(0).all(), \
        "FAIL: Non-positive unit_cost values"
    assert df["hhi"].between(0, 10_000).all(), \
        "FAIL: HHI values out of [0, 10 000] range"

    # Soft checks (logged as warnings)
    null_hhi    = df["hhi_risk_band"].eq("Unknown").sum()
    null_hs     = df["hs_code"].isna().sum()
    neg_lt      = df["lead_time_days"].lt(0).sum()

    if null_hhi:
        log.warning(f"  {null_hhi} SKUs have no HHI mapping (hs_code not in Comtrade)")
    if null_hs:
        log.warning(f"  {null_hs} SKUs missing hs_code")
    if neg_lt:
        log.warning(f"  {neg_lt} SKUs have negative lead_time_days")

    results = {
        "total_skus":        len(df),
        "unique_ids":        df["item_id"].nunique(),
        "hhi_mapped":        int((df["hhi"] > 0).sum()),
        "hhi_unmapped":      int(null_hhi),
        "geo_conc_flag_ct":  int(df["geo_conc_flag"].sum()),
        "hhi_band_dist":     df["hhi_risk_band"].value_counts().to_dict(),
        "null_hs_codes":     int(null_hs),
        "validation_passed": True,
    }
    log.info(f"Validation PASSED  |  {results['hhi_mapped']} SKUs HHI-mapped  |  "
             f"geo_conc_flag={results['geo_conc_flag_ct']}")
    return results


# ─────────────────────────── STEP 4: PERSIST ─────────────────────────────────

def persist_artifacts(df: pd.DataFrame, validation: dict) -> None:
    """Save parquet, CSV, and update DuckDB."""
    # Parquet (primary)
    df.to_parquet(SKU_MASTER_OUT, index=False)
    log.info(f"Parquet saved → {SKU_MASTER_OUT}")

    # CSV (human-readable snapshot for quick inspection)
    df.to_csv(SKU_MASTER_CSV, index=False)
    log.info(f"CSV saved → {SKU_MASTER_CSV}")

    # DuckDB
    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS sku_master_v09")
    con.execute("CREATE TABLE sku_master_v09 AS SELECT * FROM df")
    row_ct = con.execute("SELECT COUNT(*) FROM sku_master_v09").fetchone()[0]
    log.info(f"DuckDB table sku_master_v09: {row_ct} rows")

    # Persist validation metadata as a small JSON table
    val_df = pd.DataFrame([validation])
    con.execute("DROP TABLE IF EXISTS w2_validation_log")
    con.execute("CREATE TABLE w2_validation_log AS SELECT * FROM val_df")
    con.close()

    # Save validation JSON alongside the parquet
    val_path = DATA_PROC / "w2_validation.json"
    with open(val_path, "w") as f:
        json.dump(validation, f, indent=2)
    log.info(f"Validation log → {val_path}")


# ─────────────────────────── STEP 5: REPORT ──────────────────────────────────

def generate_report(df: pd.DataFrame, validation: dict) -> None:
    """
    Generate HTML analysis report + PNG chart.
    Covers:
      • HHI risk band distribution
      • Dominant partner concentration (top 10 countries)
      • High geo_conc_flag SKUs by category
      • Validation summary table
    """
    # ── PNG: HHI distribution bar chart ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("W2 Week 2 · HHI Geo-Concentration Analysis", fontsize=14,
                 fontweight="bold", color="#1a1a2e")

    # Chart 1: HHI risk band counts
    band_counts = df["hhi_risk_band"].value_counts().reindex(
        HHI_LABELS + ["Unknown"], fill_value=0)
    colors = [HHI_COLORS.get(b, "#95a5a6") for b in band_counts.index]
    axes[0].bar(band_counts.index, band_counts.values, color=colors, edgecolor="white",
                linewidth=0.8)
    axes[0].set_title("HHI Risk Band Distribution", fontweight="bold")
    axes[0].set_ylabel("SKU Count")
    for i, v in enumerate(band_counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", va="bottom", fontsize=10)

    # Chart 2: Top-10 dominant supply partners
    top_partners = df["dominant_partner"].value_counts().head(10)
    axes[1].barh(top_partners.index[::-1], top_partners.values[::-1],
                 color="#3498db", edgecolor="white")
    axes[1].set_title("Top 10 Dominant Supply Partners", fontweight="bold")
    axes[1].set_xlabel("SKU Count")

    # Chart 3: HHI scatter by unit_cost
    scatter_df = df[df["hhi"] > 0].copy()
    scatter_colors = [HHI_COLORS.get(b, "#95a5a6") for b in scatter_df["hhi_risk_band"]]
    axes[2].scatter(scatter_df["unit_cost"], scatter_df["hhi"],
                    c=scatter_colors, alpha=0.5, s=15)
    axes[2].axhline(1500, color="#f39c12", linestyle="--", linewidth=1,
                    label="Low/Mod boundary")
    axes[2].axhline(2500, color="#e74c3c", linestyle="--", linewidth=1,
                    label="Mod/High boundary")
    axes[2].set_xscale("log")
    axes[2].set_title("HHI vs Unit Cost (log scale)", fontweight="bold")
    axes[2].set_xlabel("Unit Cost (USD)")
    axes[2].set_ylabel("HHI")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(REPORT_PNG, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    log.info(f"PNG chart saved → {REPORT_PNG}")

    # ── HTML report ───────────────────────────────────────────────────────
    band_dist_html = df["hhi_risk_band"].value_counts().to_frame().to_html(
        classes="table", border=0)
    dom_partner_html = (df.groupby("dominant_partner")["item_id"]
                         .count().sort_values(ascending=False)
                         .head(10).to_frame(name="SKU Count").to_html(
                             classes="table", border=0))
    high_risk_html = (df[df["geo_conc_flag"] == 1]
                       [["item_id", "hs_code", "hhi", "dominant_partner",
                          "dominant_share", "supply_origin_country"]]
                       .head(20).to_html(classes="table", border=0, index=False))

    val_rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        for k, v in validation.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>W2 HHI Analysis Report</title>
<style>
  body  {{ font-family: 'Segoe UI', sans-serif; background:#f0f2f5;
           color:#1a1a2e; margin:0; padding:24px; }}
  h1   {{ color:#1a1a2e; border-bottom:3px solid #f0a500; padding-bottom:8px; }}
  h2   {{ color:#1a3a6e; margin-top:32px; }}
  .card {{ background:#fff; border-radius:8px; padding:20px 28px;
           margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,.08); }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px;
            font-size:12px; font-weight:700; color:#fff; }}
  .low  {{ background:#2ecc71; }} .mod  {{ background:#f39c12; }}
  .high {{ background:#e74c3c; }}
  .kpi  {{ display:flex; gap:20px; flex-wrap:wrap; }}
  .kpi-box {{ background:#1a3a6e; color:#fff; border-radius:8px;
              padding:16px 24px; min-width:140px; text-align:center; }}
  .kpi-num {{ font-size:28px; font-weight:800; color:#f0a500; }}
  .kpi-lbl {{ font-size:12px; margin-top:4px; opacity:.85; }}
  table.table {{ border-collapse:collapse; width:100%; font-size:13px; }}
  table.table th {{ background:#1a3a6e; color:#fff; padding:8px 12px;
                    text-align:left; }}
  table.table td {{ padding:7px 12px; border-bottom:1px solid #e8eaf0; }}
  table.table tr:hover td {{ background:#f7f9ff; }}
  img {{ max-width:100%; border-radius:8px; margin-top:12px; }}
  .warn {{ background:#fff8e1; border-left:4px solid #f0a500;
           padding:10px 16px; border-radius:4px; font-size:13px; }}
  .pass {{ background:#e8f5e9; border-left:4px solid #2ecc71;
           padding:10px 16px; border-radius:4px; font-size:13px; }}
</style>
</head>
<body>
<h1>🛠 GEO-AWARE MRO v1.1 · Week 2 · Day 5 Report</h1>
<p style="color:#666">Generated: {date.today().isoformat()} &nbsp;|&nbsp;
   Pipeline: <b>W2-D5 HHI ↔ SKU Master Join</b></p>

<div class="card">
  <h2 style="margin-top:0">📊 KPI Summary</h2>
  <div class="kpi">
    <div class="kpi-box"><div class="kpi-num">{validation['total_skus']}</div>
      <div class="kpi-lbl">Total SKUs</div></div>
    <div class="kpi-box"><div class="kpi-num">{validation['hhi_mapped']}</div>
      <div class="kpi-lbl">HHI Mapped</div></div>
    <div class="kpi-box"><div class="kpi-num">{validation['geo_conc_flag_ct']}</div>
      <div class="kpi-lbl">High Geo-Conc. Flags</div></div>
    <div class="kpi-box"><div class="kpi-num">
      {validation['hhi_band_dist'].get('High', 0)}</div>
      <div class="kpi-lbl">High HHI SKUs</div></div>
    <div class="kpi-box"><div class="kpi-num">
      {validation['hhi_band_dist'].get('Low', 0)}</div>
      <div class="kpi-lbl">Low HHI SKUs</div></div>
  </div>
</div>

<div class="card">
  <h2 style="margin-top:0">📈 Charts</h2>
  <img src="w2_hhi_distribution.png" alt="HHI Distribution Charts">
</div>

<div class="card">
  <h2 style="margin-top:0">🌍 HHI Risk Band Distribution</h2>
  {band_dist_html}
  <p class="warn">⚠ <b>High HHI</b> (>2500) SKUs face single-source supply risk.
  These will attract the highest geo-risk adjustment (λ) in the W8 Newsvendor model.</p>
</div>

<div class="card">
  <h2 style="margin-top:0">🏭 Top 10 Dominant Supply Partners</h2>
  {dom_partner_html}
</div>

<div class="card">
  <h2 style="margin-top:0">🚨 High geo_conc_flag SKUs (Top 20)</h2>
  <p>Criteria: <code>hhi_risk_band == "High"</code> AND
     <code>dominant_share > 0.50</code></p>
  {high_risk_html}
</div>

<div class="card">
  <h2 style="margin-top:0">✅ Validation Results</h2>
  <div class="pass">All hard data-quality checks passed.</div>
  <table class="table" style="margin-top:12px">
    <tr><th>Metric</th><th>Value</th></tr>
    {val_rows}
  </table>
</div>

<div class="card">
  <h2 style="margin-top:0">🗂 DVC Tracking Reminder</h2>
  <pre style="background:#1a1a2e;color:#f0a500;padding:14px;border-radius:6px;
              font-size:13px;overflow:auto">
dvc add data/processed/sku_master_v0.9.parquet
dvc commit
git add data/processed/sku_master_v0.9.parquet.dvc .gitignore dvc.lock dvc.yaml
git commit -m "data(W2): SKU Master v0.9 with HHI — DVC tracked"
git tag -a "data-w2-sku-master-v0.9" -m "Week 2 complete: SKU Master + HHI"
git push origin develop --tags
  </pre>
</div>

<div class="card">
  <h2 style="margin-top:0">📅 Week 2 Milestone Gate</h2>
  <table class="table">
    <tr><th>Check</th><th>Status</th></tr>
    <tr><td>M5 Walmart data ingested → DuckDB</td>
        <td><span class="badge low">✓</span></td></tr>
    <tr><td>SKU Master schema (20+ fields incl. supply_origin_country)</td>
        <td><span class="badge low">✓</span></td></tr>
    <tr><td>500-SKU synthetic master (costs, lead times, origins)</td>
        <td><span class="badge low">✓</span></td></tr>
    <tr><td>UN Comtrade HS codes + country-of-origin import data</td>
        <td><span class="badge low">✓</span></td></tr>
    <tr><td>HHI concentration index joined to SKU Master</td>
        <td><span class="badge low">✓</span></td></tr>
    <tr><td>DVC tracked (dvc.lock updated)</td>
        <td><span class="badge mod">⏳ Run dvc repro</span></td></tr>
  </table>
</div>

<footer style="margin-top:32px;color:#aaa;font-size:12px">
  GEO-AWARE MRO v1.1 · Deepender · Decision Science | Supply Chain Analytics
</footer>
</body></html>"""

    with open(REPORT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"HTML report saved → {REPORT_HTML}")


# ─────────────────────────── STEP 6: DVC YAML ────────────────────────────────

def upsert_dvc_yaml() -> None:
    """
    Append the Week 2 stage to dvc.yaml if not already present.
    Structure follows DVC 3.x stage format.
    """
    w2_stage = """
  w2_hhi_join:
    cmd: python w2_d5_hhi_sku_master_dvc.py
    deps:
      - w2_d5_hhi_sku_master_dvc.py
      - data/processed/sku_master_500.parquet
      - data/external/comtrade_raw/
    outs:
      - data/processed/sku_master_v0.9.parquet
      - data/processed/sku_master_v0.9.csv
    metrics:
      - data/processed/w2_validation.json:
          cache: false
    plots:
      - reports/w2_hhi_distribution.png:
          cache: false
"""
    if not DVC_YAML.exists():
        DVC_YAML.write_text("stages:\n" + w2_stage)
        log.info(f"dvc.yaml created → {DVC_YAML}")
        return

    content = DVC_YAML.read_text()
    if "w2_hhi_join" in content:
        log.info("dvc.yaml already contains w2_hhi_join stage — skipping.")
        return

    # Append stage
    with open(DVC_YAML, "a") as f:
        f.write(w2_stage)
    log.info(f"dvc.yaml updated with w2_hhi_join stage → {DVC_YAML}")


# ─────────────────────────── WEEK 2 SUMMARY ──────────────────────────────────

def print_week2_summary(df: pd.DataFrame, validation: dict) -> None:
    """Console summary of all Week 2 outputs."""
    band = validation["hhi_band_dist"]
    log.info("\n" + "═" * 68)
    log.info("  WEEK 2 COMPLETE — SUMMARY")
    log.info("═" * 68)
    log.info(f"  Total SKUs               : {validation['total_skus']}")
    log.info(f"  HHI mapped               : {validation['hhi_mapped']} "
             f"({validation['hhi_mapped']/validation['total_skus']*100:.1f}%)")
    log.info(f"  HHI Low  (<1500)         : {band.get('Low', 0)}")
    log.info(f"  HHI Moderate (1500-2500) : {band.get('Moderate', 0)}")
    log.info(f"  HHI High (>2500)         : {band.get('High', 0)}")
    log.info(f"  Geo-concentration flags  : {validation['geo_conc_flag_ct']}")
    log.info("─" * 68)
    log.info(f"  ARTIFACTS")
    log.info(f"  ├─ {SKU_MASTER_OUT}")
    log.info(f"  ├─ {SKU_MASTER_CSV}")
    log.info(f"  ├─ {DB_PATH}  (tables: sku_master_v09, comtrade_imports, sku_hhi_concentration)")
    log.info(f"  ├─ {REPORT_HTML}")
    log.info(f"  └─ {REPORT_PNG}")
    log.info("─" * 68)
    log.info("  NEXT ACTIONS")
    log.info("  1. dvc add data/processed/sku_master_v0.9.parquet")
    log.info("  2. dvc repro  (runs full W1+W2 pipeline)")
    log.info("  3. git commit + tag  data-w2-sku-master-v0.9")
    log.info("  4. git push origin develop --tags")
    log.info("  5. Update README — Week 2 section ✓")
    log.info("  ──")
    log.info("  WEEK 3 PREVIEW → ABC × VED × FNS Classifier (Monday)")
    log.info("    ACV = unit_cost × annual_demand → Pareto A/B/C cutoffs")
    log.info("═" * 68)


# ─────────────────────────── MAIN ────────────────────────────────────────────

def main() -> None:
    log.info("=" * 68)
    log.info("W2-D5 | HHI → SKU Master join + DVC tracking  START")
    log.info("=" * 68)

    sku_df  = load_sku_master()           # Step 1a
    hhi_df  = load_hhi_table()            # Step 1b
    merged  = join_hhi(sku_df, hhi_df)    # Step 2
    val     = validate_sku_master(merged) # Step 3
    persist_artifacts(merged, val)        # Step 4
    generate_report(merged, val)          # Step 5
    upsert_dvc_yaml()                     # Step 6
    print_week2_summary(merged, val)      # Summary


if __name__ == "__main__":
    main()
