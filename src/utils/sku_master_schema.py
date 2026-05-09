"""
SKU Master field registry — single source of truth for all classifiers.
Add new fields here first; classifiers read REQUIRED_COLS from this module.
"""

# ── Core identity ─────────────────────────────────────────────────────────────
IDENTITY_COLS = ["item_id"]

# ── ABC inputs ───────────────────────────────────────────────────────────────
ABC_COLS = ["unit_cost", "demand"]

# ── VED inputs ───────────────────────────────────────────────────────────────
VED_COLS = ["equipment_density_score"]

# ── FNS inputs ───────────────────────────────────────────────────────────────
FNS_COLS = ["adi", "cv_squared"]

# ── Location inputs ──────────────────────────────────────────────────────────
LOCATION_COLS = [
    "depot_tier",
    "environment_multiplier",
    "location_score_adj",
]

# ── LTR inputs ───────────────────────────────────────────────────────────────
LTR_COLS = [
    "lead_time_days",
    "supply_origin_country",
    "geo_risk_score",
    "ltr_score",
]

# ── Newsvendor inputs ────────────────────────────────────────────────────────
NEWSVENDOR_COLS = [
    "mean_demand",
    "std_demand",
    "mean_lead_time",
    "std_lead_time",
    "stockout_cost_usd",
]

# ── Classification outputs ──────────────────────────────────────────────────
CLASSIFICATION_OUTPUT_COLS = [
    "acv",
    "acv_for_abc",
    "abc_class",
    "ved_class",
    "fns_class",
    "abc_score",
    "ved_score",
    "fns_score",
    "ci_score",
    "ci_tier",
    "cvs_flag",
    "tsl",
    "critical_ratio",
    "q_star",
    "rop",
    "z_score",
    "sigma_rop",
    "demand_dist_used",
]

# ── Full schema ──────────────────────────────────────────────────────────────
V1_1_SCHEMA = (
    IDENTITY_COLS
    + ABC_COLS
    + VED_COLS
    + FNS_COLS
    + LOCATION_COLS
    + LTR_COLS
    + NEWSVENDOR_COLS
)

# ── Validation helper ────────────────────────────────────────────────────────
def validate_columns(df, required: list, caller: str = "unknown") -> None:
    """Raise ValueError listing missing columns with caller context."""
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"[{caller}] Missing required columns: {sorted(missing)}\n"
            f"Available: {sorted(df.columns.tolist())}"
        )
