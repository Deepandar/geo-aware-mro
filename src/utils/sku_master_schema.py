# src/utils/sku_master_schema.py

SKU_MASTER_FIELDS = {
    # --- EXISTING FIELDS ---
    "item_id": {
        "dtype": "str",
        "description": "Primary key"
    },
    "unit_cost": {
        "dtype": "float",
        "description": "Unit cost (INR/USD)"
    },
    "demand": {
        "dtype": "float",
        "description": "Annual demand quantity"
    },
    "lead_time_days": {
        "dtype": "int",
        "description": "Baseline lead time"
    },
    "supply_origin_country": {
        "dtype": "str",
        "description": "ISO country code"
    },
    "equipment_density_score": {
        "dtype": "float",
        "description": "Asset density at depot [0,1]"
    },
    "stockout_cost_usd": {
        "dtype": "float",
        "description": "Penalty for stockout"
    },

    # --- NEW FIELDS: Concept 1 additions (W3D5) ---
    "environment_multiplier": {
        "dtype": "float",
        "default": 1.0,
        "range": [0.8, 1.5],
        "description": (
            "v1.1: Static default=1.0. "
            "v1.2 extension point: populated by NASA CMAPSS operating context "
            "(climate zone, sortie rate, asset health index). "
            "Multiplies base_position_score to yield location_score_adj."
        ),
        "version_introduced": "v1.1",
        "populated_by": "v1.2 — NASA CMAPSS + operational context layer"
    },
    "location_score_adj": {
        "dtype": "float",
        "default": None,
        "description": (
            "Adjusted location score = base_position_score × environment_multiplier. "
            "In v1.1 this equals base_position_score (multiplier=1.0). "
            "Drives Ci weighting via w4 dimension."
        ),
        "version_introduced": "v1.1",
        "computed_by": "location_scorer.compute_location_score()"
    },
    #  Concept 2 additions (W3D5) 
    "ltr_score": {
        "dtype":       "float",
        "default":     0.0,
        "range":       [0.0, 1.0],
        "description": (
            "Lead Time Risk score normalized [0,1]. "
            "Formula: normalize(lead_time_days_i / mu_LT  (1 + geo_risk_score_i)). "
            "v1.1: geo_risk_score=0.0 (placeholder until W7-W8 Bayesian layer). "
            "v1.2: live geo_risk_score from Bayesian posterior replaces placeholder."
        ),
        "version_introduced": "v1.1",
        "populated_by":       "src/classifiers/ltr_scorer.py"
    },
}
