from __future__ import annotations

import json
import logging
import warnings
from datetime import date
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ───────────────────────── PATHS ─────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_PROC = BASE_DIR / "data" / "processed"
REPORTS = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

for d in [DATA_PROC, REPORTS, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"w3_d5_{date.today():%Y%m%d}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SKU_IN = DATA_PROC / "skumaster500.parquet"
SKU_OUT = DATA_PROC / "sku_master_w3.parquet"

REPORT_HTML = REPORTS / "w3_classification_report.html"
LORENZ_PNG = REPORTS / "w3_lorenz_curve.png"
HEATMAP_PNG = REPORTS / "w3_27class_heatmap.png"
SENSITIVITY_PNG = REPORTS / "w3_ci_sensitivity.png"
QUADRANT_PNG = REPORTS / "w3_demand_quadrant.png"
GATE_JSON = DATA_PROC / "w3_milestone_gate.json"


# ─────────────────────── CONSTANTS ───────────────────────
ABC_A_CUM: Final[float] = 0.70
ABC_B_CUM: Final[float] = 0.90

ADI_BOUNDARY: Final[float] = 1.32
CV2_BOUNDARY: Final[float] = 0.49

ALL_27_CELLS: Final[list[str]] = [
    a + v + f
    for a in ("A", "B", "C")
    for v in ("V", "E", "D")
    for f in ("F", "N", "S")
]

CI_WEIGHT_SCENARIOS: Final[dict[str, dict[str, float]]] = {
    "Baseline\n(0.35/0.30/0.20/0.15)": {"abc": 0.35, "ved": 0.30, "fns": 0.20, "loc": 0.15},
    "Value-heavy": {"abc": 0.50, "ved": 0.25, "fns": 0.15, "loc": 0.10},
    "Criticality-heavy": {"abc": 0.20, "ved": 0.50, "fns": 0.20, "loc": 0.10},
    "Demand-heavy": {"abc": 0.25, "ved": 0.25, "fns": 0.40, "loc": 0.10},
    "Location-heavy": {"abc": 0.25, "ved": 0.25, "fns": 0.15, "loc": 0.35},
}

ABC_SCORE = {"A": 1.00, "B": 0.50, "C": 0.00}
VED_SCORE = {"V": 1.00, "E": 0.50, "D": 0.00}
FNS_SCORE = {"F": 1.00, "N": 0.50, "S": 0.00}
LOC_SCORE = {"Forward": 1.00, "Border": 0.60, "Rear": 0.00}

PALETTE = {
    "navy": "#1a3a6e",
    "gold": "#f0a500",
    "teal": "#00b4d8",
    "red": "#e74c3c",
    "green": "#2ecc71",
    "amber": "#f39c12",
    "grey": "#95a5a6",
}


# ───────────────────── DATA PREPARATION ─────────────────────

def load_sku_master() -> pd.DataFrame:
    if SKU_IN.exists():
        df = pd.read_parquet(SKU_IN)
        log.info(f"Loaded SKU Master ({len(df)} rows) from {SKU_IN}")
    else:
        raise FileNotFoundError(f"Input file not found: {SKU_IN}")

    # adapt actual parquet schema -> report schema
    rename_map = {
        "item_id": "itemid",
        "abc_class": "abcclass",
        "ved_class": "vedclass",
        "fns_class": "fnsclass",
        "sku_class": "taxonomycell",
        "criticality_index": "ci",
        "location": "locationtype",
        "equipment_density_score": "equipmentdensityscore",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # derive missing report-only fields from available columns
    if "cv" in df.columns and "cv2" not in df.columns:
        df["cv2"] = pd.to_numeric(df["cv"], errors="coerce").fillna(0).pow(2)

    if "fnsclass" in df.columns and "adi" not in df.columns:
        df["adi"] = df["fnsclass"].map({"F": 1.0, "N": 1.6, "S": 2.4}).fillna(1.6)

    if "demandclass" not in df.columns and {"adi", "cv2"}.issubset(df.columns):
        cond = [
            (df["adi"] < ADI_BOUNDARY) & (df["cv2"] < CV2_BOUNDARY),
            (df["adi"] < ADI_BOUNDARY) & (df["cv2"] >= CV2_BOUNDARY),
            (df["adi"] >= ADI_BOUNDARY) & (df["cv2"] < CV2_BOUNDARY),
            (df["adi"] >= ADI_BOUNDARY) & (df["cv2"] >= CV2_BOUNDARY),
        ]
        df["demandclass"] = np.select(cond, ["Smooth", "Erratic", "Intermittent", "Lumpy"], default="Unknown")

    if "forecastmethod" not in df.columns and "demandclass" in df.columns:
        df["forecastmethod"] = df["demandclass"].map({
            "Smooth": "Holt-Winters",
            "Erratic": "ARIMA",
            "Intermittent": "Croston",
            "Lumpy": "SBA",
            "Unknown": "Review",
        })

    if "abcscore" not in df.columns and "abcclass" in df.columns:
        df["abcscore"] = df["abcclass"].map(ABC_SCORE)
    if "vedscore" not in df.columns and "vedclass" in df.columns:
        df["vedscore"] = df["vedclass"].map(VED_SCORE)
    if "fnsscore" not in df.columns and "fnsclass" in df.columns:
        df["fnsscore"] = df["fnsclass"].map(FNS_SCORE)

    required = [
        "itemid", "abcclass", "vedclass", "fnsclass", "taxonomycell",
        "ci", "acv", "adi", "cv2", "locationtype",
        "equipmentdensityscore", "forecastmethod"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Still missing required columns after schema adaptation: {missing}")

    return df


def adapt_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["taxonomy_cell"] = df["sku_class"].astype(str).str.upper()
    df["ci"] = pd.to_numeric(df["criticality_index"], errors="coerce")
    df["location_type"] = df["location"].astype(str)

    df["abc_class"] = df["abc_class"].astype(str).str.upper()
    df["ved_class"] = df["ved_class"].astype(str).str.upper()
    df["fns_class"] = df["fns_class"].astype(str).str.upper()

    return df


def ensure_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["abc_score"] = df["abc_class"].map(ABC_SCORE)
    df["ved_score"] = df["ved_class"].map(VED_SCORE)
    df["fns_score"] = df["fns_class"].map(FNS_SCORE)
    df["loc_score"] = df["location_type"].map(LOC_SCORE).fillna(0.0)

    return df


def ensure_demand_quadrant(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fallback approximation:
    # - no weekly history is present, so ADI cannot be truly reconstructed
    # - use FNS class to create a stable proxy for quadrant reporting
    fns_to_adi = {"F": 1.00, "N": 1.60, "S": 2.40}
    df["adi"] = df["fns_class"].map(fns_to_adi).astype(float)

    # cv column is treated as CV; Day-5 chart expects CV²
    df["cv2"] = pd.to_numeric(df["cv"], errors="coerce").fillna(0).pow(2)

    cond = [
        (df["adi"] < ADI_BOUNDARY) & (df["cv2"] < CV2_BOUNDARY),
        (df["adi"] < ADI_BOUNDARY) & (df["cv2"] >= CV2_BOUNDARY),
        (df["adi"] >= ADI_BOUNDARY) & (df["cv2"] < CV2_BOUNDARY),
        (df["adi"] >= ADI_BOUNDARY) & (df["cv2"] >= CV2_BOUNDARY),
    ]
    labels = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    df["demand_class"] = np.select(cond, labels, default="Unknown")

    return df


def ensure_forecast_routing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    forecast_map = {
        "Smooth": "Holt-Winters",
        "Erratic": "ARIMA",
        "Intermittent": "Croston",
        "Lumpy": "SBA",
        "Unknown": "Review",
    }
    df["forecast_method"] = df["demand_class"].map(forecast_map)

    return df


def validate_core(df: pd.DataFrame) -> None:
    assert df["taxonomy_cell"].nunique() <= 27, "Taxonomy exceeded 27 cells"
    assert df["ci"].between(0, 1).all(), "Ci out of [0,1]"
    assert not df[["abc_class", "ved_class", "fns_class", "taxonomy_cell", "ci"]].isna().any().any(), \
        "Null classification values detected"


# ───────────────────── LORENZ / GINI ─────────────────────
def compute_lorenz(df: pd.DataFrame) -> dict:
    acv = pd.to_numeric(df["acv"], errors="coerce").fillna(0).values
    acv = np.sort(acv)

    n = len(acv)
    cum_acv = np.cumsum(acv) / acv.sum()
    cum_sku = np.arange(1, n + 1) / n

    gini = float(1 - 2 * np.trapz(cum_acv, cum_sku))

    acv_desc = np.sort(df["acv"].values)[::-1]
    cum_desc = np.cumsum(acv_desc) / acv_desc.sum()

    a_idx = np.searchsorted(cum_desc, ABC_A_CUM, side="left") + 1
    ab_idx = np.searchsorted(cum_desc, ABC_B_CUM, side="left") + 1

    a_sku_share = a_idx / n
    ab_sku_share = ab_idx / n

    return {
        "cum_sku": cum_sku,
        "cum_acv": cum_acv,
        "gini": gini,
        "a_sku_share": a_sku_share,
        "ab_sku_share": ab_sku_share,
    }


# ─────────────────────── CHARTS ───────────────────────
def plot_lorenz(df: pd.DataFrame, lorenz: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    x = lorenz["cum_sku"]
    y = lorenz["cum_acv"]

    ax.plot([0, 1], [0, 1], color=PALETTE["grey"], linestyle="--", linewidth=1.4, label="Line of equality")
    ax.plot(x, y, color=PALETTE["navy"], linewidth=2.5, label="Lorenz curve")
    ax.fill_between(x, y, x, alpha=0.12, color=PALETTE["navy"])

    a_x = lorenz["a_sku_share"]
    ab_x = lorenz["ab_sku_share"]

    ax.axvspan(0, a_x, alpha=0.10, color=PALETTE["red"], label=f"A-class: {a_x*100:.1f}% SKUs")
    ax.axvspan(a_x, ab_x, alpha=0.08, color=PALETTE["amber"], label=f"B-band: {(ab_x-a_x)*100:.1f}% SKUs")
    ax.axvspan(ab_x, 1.0, alpha=0.07, color=PALETTE["green"], label=f"C-band: {(1-ab_x)*100:.1f}% SKUs")

    ax.axvline(a_x, color=PALETTE["red"], linestyle=":", linewidth=1.2)
    ax.axvline(ab_x, color=PALETTE["amber"], linestyle=":", linewidth=1.2)
    ax.axhline(0.70, color=PALETTE["red"], linestyle=":", linewidth=1.0, alpha=0.6)
    ax.axhline(0.90, color=PALETTE["amber"], linestyle=":", linewidth=1.0, alpha=0.6)

    ax.text(
        0.37, 0.08,
        f"Gini coefficient\nG = {lorenz['gini']:.4f}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=PALETTE["gold"], linewidth=2),
    )

    ax.set_xlabel("Cumulative SKU share (sorted by ACV ascending)")
    ax.set_ylabel("Cumulative ACV share")
    ax.set_title("Lorenz Curve — ABC Pareto Analysis", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.savefig(LORENZ_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_27class_heatmap(df: pd.DataFrame) -> None:
    agg = (
        df.groupby("taxonomy_cell")
        .agg(count=("item_id", "count"), mean_ci=("ci", "mean"))
        .reindex(ALL_27_CELLS, fill_value=0)
        .reset_index()
    )
    agg.columns = ["cell", "count", "mean_ci"]

    count_mat = np.zeros((3, 9))
    meanci_mat = np.full((3, 9), np.nan)
    cell_labels = np.empty((3, 9), dtype=object)

    ved_idx = {"V": 0, "E": 1, "D": 2}
    abc_idx = {"A": 0, "B": 1, "C": 2}
    fns_idx = {"F": 0, "N": 1, "S": 2}

    for _, row in agg.iterrows():
        c = row["cell"]
        r = ved_idx[c[1]]
        col = abc_idx[c[0]] * 3 + fns_idx[c[2]]
        count_mat[r, col] = row["count"]
        meanci_mat[r, col] = row["mean_ci"] if row["count"] > 0 else np.nan
        cell_labels[r, col] = c

    fig, ax = plt.subplots(figsize=(14, 5))

    cmap = plt.cm.get_cmap("YlOrRd").copy()
    cmap.set_under("#f0f0f0")
    vmax = max(count_mat.max(), 1)
    im = ax.imshow(count_mat, cmap=cmap, vmin=0.5, vmax=vmax, aspect="auto")

    for r in range(3):
        for c in range(9):
            cnt = int(count_mat[r, c])
            ci = meanci_mat[r, c]
            lbl = cell_labels[r, c]

            if cnt == 0:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="#f0f0f0", edgecolor="#dddddd"))
                ax.text(c, r, f"{lbl}\n—", ha="center", va="center", fontsize=8, color="#aaaaaa")
            else:
                text_color = "white" if cnt > vmax * 0.55 else "#1a1a2e"
                ax.text(c, r - 0.16, lbl, ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)
                ax.text(c, r + 0.10, f"n={cnt}", ha="center", va="center", fontsize=8, color=text_color)
                ax.text(c, r + 0.34, f"Ci={ci:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["V (Vital)", "E (Essential)", "D (Desirable)"], fontsize=11)
    ax.set_xticks(range(9))
    ax.set_xticklabels([f"{a}\n{f}" for a in "ABC" for f in "FNS"], fontsize=10)

    for x in [2.5, 5.5]:
        ax.axvline(x, color=PALETTE["navy"], linewidth=2)

    for i, (a, x_center) in enumerate(zip("ABC", [1, 4, 7])):
        ax.text(
            x_center, -0.75, f"CLASS {a}", ha="center",
            fontsize=11, fontweight="bold",
            color=[PALETTE["red"], PALETTE["amber"], PALETTE["green"]][i]
        )

    plt.colorbar(im, ax=ax, label="SKU count", shrink=0.85)
    ax.set_title("27-Class ABC × VED × FNS Taxonomy Heatmap", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    plt.tight_layout()
    plt.savefig(HEATMAP_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_ci_sensitivity(df: pd.DataFrame) -> None:
    scenarios = list(CI_WEIGHT_SCENARIOS.keys())
    means = []

    for _, w in CI_WEIGHT_SCENARIOS.items():
        ci_s = (
            w["abc"] * df["abc_score"]
            + w["ved"] * df["ved_score"]
            + w["fns"] * df["fns_score"]
            + w["loc"] * df["loc_score"]
        ).clip(0, 1)
        means.append(float(ci_s.mean()))

    n = len(scenarios)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    theta += theta[:1]
    means_plot = means + means[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(theta, means_plot, color=PALETTE["navy"], linewidth=2.5)
    ax.fill(theta, means_plot, alpha=0.20, color=PALETTE["teal"])
    ax.fill_between(theta, [0.35] * len(theta), [0.65] * len(theta), alpha=0.10, color=PALETTE["green"])

    ax.scatter([theta[0]], [means[0]], color=PALETTE["gold"], s=120, zorder=5)

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(scenarios, fontsize=8.5, color=PALETTE["navy"])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.35, 0.50, 0.65, 0.80])
    ax.set_yticklabels(["0.2", "0.35", "0.5", "0.65", "0.8"], fontsize=8)
    ax.grid(color=PALETTE["grey"], alpha=0.35)
    ax.set_title("Ci Weight Sensitivity Analysis", fontsize=12, fontweight="bold", color=PALETTE["navy"], pad=20)

    for t, m in zip(theta[:-1], means):
        ax.annotate(f"{m:.3f}", xy=(t, m), xytext=(t, m + 0.06), fontsize=8.5, ha="center", color=PALETTE["navy"])

    plt.tight_layout()
    plt.savefig(SENSITIVITY_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_demand_quadrant(df: pd.DataFrame) -> None:
    color_map = {
        "Smooth": PALETTE["green"],
        "Erratic": PALETTE["amber"],
        "Intermittent": PALETTE["teal"],
        "Lumpy": PALETTE["red"],
        "Unknown": PALETTE["grey"],
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    for label, grp in df.groupby("demand_class"):
        ax.scatter(grp["adi"], grp["cv2"], color=color_map[label], alpha=0.55, s=20, label=label, zorder=3)

    ax.axvline(ADI_BOUNDARY, color="#1a1a2e", linestyle="--", linewidth=1.5)
    ax.axhline(CV2_BOUNDARY, color="#1a1a2e", linestyle=":", linewidth=1.5)

    quad_info = [
        (0.25, 0.95, "Smooth → F\nHolt-Winters", PALETTE["green"]),
        (1.55, 0.95, "Erratic → N\nARIMA", PALETTE["amber"]),
        (0.25, 0.12, "Intermittent → N\nCroston", PALETTE["teal"]),
        (1.55, 0.12, "Lumpy → S\nSBA", PALETTE["red"]),
    ]
    for qx, qy, qtxt, qcol in quad_info:
        ax.text(
            qx, qy, qtxt, fontsize=9, color=qcol, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=qcol, alpha=0.90, linewidth=1.5),
        )

    for label, grp in df.groupby("demand_class"):
        pct = len(grp) / len(df) * 100
        med_adi = grp["adi"].median()
        med_cv2 = grp["cv2"].median()
        ax.text(
            med_adi, med_cv2, f"n={len(grp)}\n({pct:.0f}%)",
            ha="center", va="center", fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color_map[label], alpha=0.80),
        )

    ax.set_xlabel("ADI — Average Demand Interval")
    ax.set_ylabel("CV² — Squared Coefficient of Variation")
    ax.set_title("Demand Pattern Quadrant (Proxy from available schema)", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-0.1, 3.2)
    ax.set_ylim(-0.05, max(1.2, min(df["cv2"].max() * 1.05, 12)))
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(QUADRANT_PNG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()


# ───────────────────── MILESTONE GATE ─────────────────────
def run_milestone_gate(df: pd.DataFrame, lorenz: dict) -> dict:
    gate = {}

    gate["all_skus_classified"] = (
        df[["abc_class", "ved_class", "fns_class"]].notna().all(axis=1).all(),
        f"Total SKUs: {len(df)}"
    )

    n_cells = df["taxonomy_cell"].nunique()
    gate["taxonomy_27_cells"] = (
        n_cells <= 27,
        f"Unique cells: {n_cells}/27"
    )

    gate["ci_bounds"] = (
        df["ci"].between(0, 1).all(),
        f"Ci min={df['ci'].min():.4f}, max={df['ci'].max():.4f}"
    )

    ci_mean = float(df["ci"].mean())
    gate["ci_mean_target"] = (
        0.35 <= ci_mean <= 0.65,
        f"Ci mean={ci_mean:.4f}, target=[0.35,0.65]"
    )

    acv_a_pct = df.loc[df["abc_class"] == "A", "acv"].sum() / df["acv"].sum()
    gate["abc_pareto"] = (
        0.60 <= acv_a_pct <= 0.80,
        f"A-class ACV share={acv_a_pct*100:.1f}%"
    )

    gate["gini_mro_range"] = (
        0.45 <= lorenz["gini"] <= 0.95,
        f"Gini={lorenz['gini']:.4f}"
    )

    methods = set(df["forecast_method"].dropna().unique())
    required_methods = {"Holt-Winters", "ARIMA", "Croston", "SBA"}
    gate["forecast_routing_complete"] = (
        required_methods.issubset(methods),
        f"Methods found={sorted(methods)}"
    )

    valid_abc = set(df["abc_class"].unique()).issubset({"A", "B", "C"})
    valid_ved = set(df["ved_class"].unique()).issubset({"V", "E", "D"})
    valid_fns = set(df["fns_class"].unique()).issubset({"F", "N", "S"})
    gate["no_label_drift"] = (
        valid_abc and valid_ved and valid_fns,
        f"ABC={sorted(df['abc_class'].unique())}, VED={sorted(df['ved_class'].unique())}, FNS={sorted(df['fns_class'].unique())}"
    )

    passed = sum(int(v[0]) for v in gate.values())
    total = len(gate)

    gate_json = {k: {"passed": v[0], "detail": v[1]} for k, v in gate.items()}
    gate_json["summary"] = {"passed": passed, "total": total, "all_green": passed == total}

    with open(GATE_JSON, "w", encoding="utf-8") as f:
        json.dump(gate_json, f, indent=2)

    for check, (ok, detail) in gate.items():
        log.info("%s %s | %s", "PASS" if ok else "FAIL", check, detail)
    log.info("Gate result: %s/%s", passed, total)

    return gate_json


# ───────────────────── HTML REPORT ─────────────────────
def build_html_report(df: pd.DataFrame, lorenz: dict, gate: dict) -> None:
    abc_cnt = df["abc_class"].value_counts().reindex(["A", "B", "C"], fill_value=0)
    ved_cnt = df["ved_class"].value_counts().reindex(["V", "E", "D"], fill_value=0)
    fns_cnt = df["fns_class"].value_counts().reindex(["F", "N", "S"], fill_value=0)
    dem_cnt = df["demand_class"].value_counts()
    fore_cnt = df["forecast_method"].value_counts()

    total_acv = df["acv"].sum()
    acv_a_pct = df.loc[df["abc_class"] == "A", "acv"].sum() / total_acv * 100
    acv_b_pct = df.loc[df["abc_class"] == "B", "acv"].sum() / total_acv * 100
    acv_c_pct = df.loc[df["abc_class"] == "C", "acv"].sum() / total_acv * 100

    cell_counts = df["taxonomy_cell"].value_counts().reindex(ALL_27_CELLS, fill_value=0)

    cell_rows = []
    for cell, cnt in cell_counts.items():
        pct = cnt / len(df) * 100
        ci_mean = df.loc[df["taxonomy_cell"] == cell, "ci"].mean() if cnt > 0 else np.nan
        ci_std = df.loc[df["taxonomy_cell"] == cell, "ci"].std() if cnt > 1 else 0.0
        bar = "█" * int(round(pct / 2))
        cell_rows.append(
            f"<tr><td>{cell}</td><td>{cnt}</td><td>{pct:.1f}%</td><td style='font-family:monospace'>{bar}</td>"
            f"<td>{'' if np.isnan(ci_mean) else f'{ci_mean:.3f}'}</td><td>{ci_std:.3f}</td></tr>"
        )
    cell_rows_html = "\n".join(cell_rows)

    gate_rows = []
    for check, payload in gate.items():
        if check == "summary":
            continue
        status = "✅ PASS" if payload["passed"] else "❌ FAIL"
        gate_rows.append(f"<tr><td>{check}</td><td>{status}</td><td>{payload['detail']}</td></tr>")
    gate_rows_html = "\n".join(gate_rows)

    forecast_items = "".join(f"<li>{k}: {v} SKUs</li>" for k, v in fore_cnt.items())

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>W3 Classification Report</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 30px;
    color: #222;
    line-height: 1.5;
}}
h1, h2, h3 {{ color: #1a3a6e; }}
.kpis {{
    display: grid;
    grid-template-columns: repeat(4, minmax(180px, 1fr));
    gap: 12px;
    margin: 20px 0;
}}
.card {{
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 14px;
    background: #fafafa;
}}
img {{
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 8px;
    margin: 12px 0 24px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0 28px;
}}
th, td {{
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
    vertical-align: top;
}}
th {{ background: #f3f6fb; }}
.code {{
    background: #f6f8fa;
    padding: 12px;
    border-radius: 8px;
    font-family: monospace;
    white-space: pre-wrap;
}}
.small {{ color: #666; font-size: 0.95rem; }}
</style>
</head>
<body>
    <h1>Week 3 · Day 5 Classification Report</h1>
    <p class="small">Generated: {date.today().isoformat()}</p>

    <div class="kpis">
        <div class="card"><h3>SKUs</h3><p>{len(df)}</p></div>
        <div class="card"><h3>Gini</h3><p>{lorenz["gini"]:.4f}</p></div>
        <div class="card"><h3>Mean Ci</h3><p>{df["ci"].mean():.4f}</p></div>
        <div class="card"><h3>27-cell usage</h3><p>{df["taxonomy_cell"].nunique()} / 27</p></div>
    </div>

    <h2>ABC summary</h2>
    <ul>
        <li>A: {abc_cnt["A"]} SKUs → {acv_a_pct:.1f}% of ACV</li>
        <li>B: {abc_cnt["B"]} SKUs → {acv_b_pct:.1f}% of ACV</li>
        <li>C: {abc_cnt["C"]} SKUs → {acv_c_pct:.1f}% of ACV</li>
    </ul>

    <h2>VED summary</h2>
    <ul>
        <li>V: {ved_cnt["V"]} SKUs</li>
        <li>E: {ved_cnt["E"]} SKUs</li>
        <li>D: {ved_cnt["D"]} SKUs</li>
    </ul>

    <h2>FNS summary</h2>
    <ul>
        <li>F: {fns_cnt["F"]} SKUs</li>
        <li>N: {fns_cnt["N"]} SKUs</li>
        <li>S: {fns_cnt["S"]} SKUs</li>
    </ul>

    <h2>Lorenz curve</h2>
    <p>The Lorenz curve should bow below the diagonal if Pareto concentration is working properly.</p>
    <img src="{LORENZ_PNG.name}" alt="Lorenz curve">

    <h2>27-class heatmap</h2>
    <p>Heatmap shows populated ABC × VED × FNS cells and mean Ci per cell.</p>
    <img src="{HEATMAP_PNG.name}" alt="27 class heatmap">

    <h2>Demand quadrant</h2>
    <p>This view is built using a proxy ADI from FNS class and CV² from the available <code>cv</code> column because weekly demand history is not present in the current parquet.</p>
    <img src="{QUADRANT_PNG.name}" alt="Demand quadrant">

    <h2>Ci sensitivity</h2>
    <p>Baseline and alternate weighting scenarios compare mean Ci stability.</p>
    <img src="{SENSITIVITY_PNG.name}" alt="Ci sensitivity radar">

    <h2>Forecast routing</h2>
    <ul>{forecast_items}</ul>

    <h2>27-cell distribution table</h2>
    <table>
        <thead>
            <tr>
                <th>Cell</th><th>Count</th><th>% SKUs</th><th>ACV share bar</th><th>Mean Ci</th><th>Std Ci</th>
            </tr>
        </thead>
        <tbody>
            {cell_rows_html}
        </tbody>
    </table>

    <h2>Milestone gate</h2>
    <table>
        <thead>
            <tr><th>Check</th><th>Status</th><th>Detail</th></tr>
        </thead>
        <tbody>
            {gate_rows_html}
        </tbody>
    </table>

    <h2>Next steps</h2>
    <div class="code">dvc add data/processed/sku_master_w3.parquet
git add reports/ logs/ data/processed/sku_master_w3.parquet.dvc
git commit -m "feat(W3D5): classification report + Lorenz + 27-class heatmap"</div>
</body>
</html>
"""
    REPORT_HTML.write_text(html, encoding="utf-8")
    log.info("HTML report written to %s", REPORT_HTML)


# ───────────────────── MLFLOW ─────────────────────
def log_to_mlflow(df: pd.DataFrame, lorenz: dict, gate: dict) -> None:
    if not MLFLOW_AVAILABLE:
        log.info("MLflow not installed — skipping")
        return

    mlflow.set_experiment("mro-sku-classification")

    with mlflow.start_run(run_name="W3-D5-classification-report"):
        mlflow.log_params({
            "n_skus": len(df),
            "w_abc": 0.35,
            "w_ved": 0.30,
            "w_fns": 0.20,
            "w_loc": 0.15,
            "adi_boundary": ADI_BOUNDARY,
            "cv2_boundary": CV2_BOUNDARY,
        })

        mlflow.log_metrics({
            "ci_mean": round(float(df["ci"].mean()), 4),
            "ci_std": round(float(df["ci"].std()), 4),
            "gini": round(lorenz["gini"], 4),
            "a_class_count": int(df["abc_class"].eq("A").sum()),
            "vital_count": int(df["ved_class"].eq("V").sum()),
            "slow_count": int(df["fns_class"].eq("S").sum()),
            "unique_cells": int(df["taxonomy_cell"].nunique()),
            "gate_checks_passed": int(gate["summary"]["passed"]),
        })

        for artifact in [LORENZ_PNG, HEATMAP_PNG, SENSITIVITY_PNG, QUADRANT_PNG, REPORT_HTML, GATE_JSON]:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))

        log.info("MLflow run logged")


# ───────────────────── MAIN ─────────────────────
def main() -> None:
    df = load_sku_master()

    lorenz = compute_lorenz(df)

    plot_lorenz(df, lorenz)
    plot_27class_heatmap(df)
    plot_demand_quadrant(df)
    plot_ci_sensitivity(df)

    gate = run_milestone_gate(df, lorenz)

    df.to_parquet(SKU_OUT, index=False)
    log.info("Saved final Week 3 parquet to %s", SKU_OUT)

    build_html_report(df, lorenz, gate)
    log_to_mlflow(df, lorenz, gate)

    log.info("Week 3 Day 5 complete")


if __name__ == "__main__":
    main()