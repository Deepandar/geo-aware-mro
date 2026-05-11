from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
import json
import html
import mlflow
from pydantic import BaseModel, Field, field_validator, model_validator

FIELD_CATALOG = {
    "sku_id": {"dtype": "VARCHAR", "nullable": False, "source": "master", "used_by": ["forecasting", "risk", "inventory"], "validation": "non-empty unique SKU"},
    "item_name": {"dtype": "VARCHAR", "nullable": False, "source": "master", "used_by": ["forecasting", "inventory"], "validation": "non-empty"},
    "item_description": {"dtype": "VARCHAR", "nullable": True, "source": "master", "used_by": ["inventory"], "validation": "free text"},
    "category": {"dtype": "VARCHAR", "nullable": False, "source": "master", "used_by": ["forecasting", "inventory"], "validation": "non-empty"},
    "subcategory": {"dtype": "VARCHAR", "nullable": True, "source": "master", "used_by": ["inventory"], "validation": "free text"},
    "uom": {"dtype": "VARCHAR", "nullable": False, "source": "master", "used_by": ["inventory"], "validation": "non-empty"},
    "abc_class": {"dtype": "VARCHAR", "nullable": False, "source": "derived", "used_by": ["inventory"], "validation": "A/B/C"},
    "ved_class": {"dtype": "VARCHAR", "nullable": False, "source": "derived", "used_by": ["inventory"], "validation": "V/E/D"},
    "fns_class": {"dtype": "VARCHAR", "nullable": False, "source": "derived", "used_by": ["forecasting", "inventory"], "validation": "F/N/S"},
    "class_code": {"dtype": "VARCHAR", "nullable": False, "source": "derived", "used_by": ["forecasting", "risk", "inventory"], "validation": "len == 3, ABC×VED×FNS"},
    "unit_cost_usd": {"dtype": "DOUBLE", "nullable": False, "source": "procurement", "used_by": ["inventory"], "validation": "> 0"},
    "lead_time_days": {"dtype": "INTEGER", "nullable": False, "source": "procurement", "used_by": ["forecasting", "inventory"], "validation": ">= 0"},
    "moq": {"dtype": "INTEGER", "nullable": False, "source": "procurement", "used_by": ["inventory"], "validation": ">= 1"},
    "supplier_id": {"dtype": "VARCHAR", "nullable": False, "source": "supplier", "used_by": ["risk", "inventory"], "validation": "non-empty"},
    "supplier_name": {"dtype": "VARCHAR", "nullable": False, "source": "supplier", "used_by": ["risk", "inventory"], "validation": "non-empty"},
    "supply_origin_country": {"dtype": "VARCHAR", "nullable": False, "source": "supplier", "used_by": ["risk"], "validation": "ISO-3 country code"},
    "geo_risk_score": {"dtype": "DOUBLE", "nullable": False, "source": "risk", "used_by": ["risk", "inventory"], "validation": "0 <= x <= 1"},
    "criticality_score": {"dtype": "DOUBLE", "nullable": False, "source": "derived", "used_by": ["inventory"], "validation": "0 <= x <= 1"},
    "annual_demand_qty": {"dtype": "DOUBLE", "nullable": False, "source": "demand", "used_by": ["forecasting", "inventory"], "validation": ">= 0"},
    "avg_monthly_demand_qty": {"dtype": "DOUBLE", "nullable": False, "source": "demand", "used_by": ["forecasting"], "validation": ">= 0"},
    "demand_cv": {"dtype": "DOUBLE", "nullable": True, "source": "derived", "used_by": ["forecasting"], "validation": ">= 0"},
    "intermittency_index": {"dtype": "DOUBLE", "nullable": True, "source": "derived", "used_by": ["forecasting"], "validation": ">= 0"},
    "service_level_target": {"dtype": "DOUBLE", "nullable": False, "source": "policy", "used_by": ["inventory"], "validation": "0 < x < 1"},
    "stockout_cost_usd": {"dtype": "DOUBLE", "nullable": False, "source": "policy", "used_by": ["inventory"], "validation": "> 0"},
    "salvage_value_usd": {"dtype": "DOUBLE", "nullable": False, "source": "policy", "used_by": ["inventory"], "validation": ">= 0 and < stockout_cost_usd"},
    "current_on_hand_qty": {"dtype": "DOUBLE", "nullable": False, "source": "inventory", "used_by": ["inventory"], "validation": ">= 0"},
    "reorder_point_qty": {"dtype": "DOUBLE", "nullable": True, "source": "derived", "used_by": ["inventory"], "validation": ">= 0"},
    "safety_stock_qty": {"dtype": "DOUBLE", "nullable": True, "source": "derived", "used_by": ["inventory"], "validation": ">= 0"},
}

PIPELINE_REQUIREMENTS = {
    "forecasting": ["sku_id", "fns_class", "class_code", "lead_time_days", "annual_demand_qty", "avg_monthly_demand_qty", "demand_cv", "intermittency_index"],
    "risk": ["sku_id", "supplier_id", "supplier_name", "supply_origin_country", "geo_risk_score", "class_code"],
    "inventory": ["sku_id", "abc_class", "ved_class", "class_code", "unit_cost_usd", "moq", "service_level_target", "stockout_cost_usd", "salvage_value_usd", "current_on_hand_qty"],
}

class SKUMaster(BaseModel):
    sku_id: str
    item_name: str
    item_description: Optional[str] = None
    category: str
    subcategory: Optional[str] = None
    uom: str
    abc_class: Literal["A", "B", "C"]
    ved_class: Literal["V", "E", "D"]
    fns_class: Literal["F", "N", "S"]
    class_code: str
    unit_cost_usd: float = Field(gt=0)
    lead_time_days: int = Field(ge=0)
    moq: int = Field(ge=1)
    supplier_id: str
    supplier_name: str
    supply_origin_country: str
    geo_risk_score: float = Field(ge=0, le=1)
    criticality_score: float = Field(ge=0, le=1)
    annual_demand_qty: float = Field(ge=0)
    avg_monthly_demand_qty: float = Field(ge=0)
    demand_cv: Optional[float] = Field(default=None, ge=0)
    intermittency_index: Optional[float] = Field(default=None, ge=0)
    service_level_target: float = Field(gt=0, lt=1)
    stockout_cost_usd: float = Field(gt=0)
    salvage_value_usd: float = Field(ge=0)
    current_on_hand_qty: float = Field(ge=0)
    reorder_point_qty: Optional[float] = Field(default=None, ge=0)
    safety_stock_qty: Optional[float] = Field(default=None, ge=0)

    @field_validator("supply_origin_country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 3 or not v.isalpha():
            raise ValueError("supply_origin_country must be ISO-3 code")
        return v

    @field_validator("class_code")
    @classmethod
    def validate_class_code(cls, v: str) -> str:
        v = v.strip().upper()
        if len(v) != 3:
            raise ValueError("class_code must have length 3")
        if v[0] not in {"A", "B", "C"} or v[1] not in {"V", "E", "D"} or v[2] not in {"F", "N", "S"}:
            raise ValueError("class_code must match ABC×VED×FNS pattern")
        return v

    @model_validator(mode="after")
    def validate_model_rules(self):
        expected = f"{self.abc_class}{self.ved_class}{self.fns_class}"
        if self.class_code != expected:
            raise ValueError(f"class_code must equal {expected}")
        if self.salvage_value_usd >= self.stockout_cost_usd:
            raise ValueError("salvage_value_usd must be < stockout_cost_usd")
        return self

def generate_ddl() -> str:
    cols = []
    for name, meta in FIELD_CATALOG.items():
        null_sql = "" if meta["nullable"] else " NOT NULL"
        cols.append(f"    {name} {meta['dtype']}{null_sql}")
    return "CREATE TABLE sku_master (\n" + ",\n".join(cols) + "\n);"

def validate_pipeline_requirements() -> None:
    catalog_fields = set(FIELD_CATALOG.keys())
    for module, fields in PIPELINE_REQUIREMENTS.items():
        missing = [f for f in fields if f not in catalog_fields]
        if missing:
            raise AssertionError(f"{module} missing fields: {missing}")

def generate_html_doc() -> str:
    rows = []
    for name, meta in FIELD_CATALOG.items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(meta['dtype'])}</td>"
            f"<td>{html.escape(str(meta['nullable']))}</td>"
            f"<td>{html.escape(meta['source'])}</td>"
            f"<td>{html.escape(', '.join(meta['used_by']))}</td>"
            f"<td>{html.escape(meta['validation'])}</td>"
            "</tr>"
        )
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>SKU Master Schema</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background: #f4f4f4; }}
    h1, h2 {{ margin-bottom: 12px; }}
  </style>
</head>
<body>
  <h1>SKU Master Schema</h1>
  <p>Field count: {len(FIELD_CATALOG)}</p>
  <h2>Fields</h2>
  <table>
    <thead>
      <tr>
        <th>Field</th><th>Type</th><th>Nullable</th><th>Source</th><th>Used By</th><th>Validation</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>"""

def write_artifacts() -> None:
    Path("src/schema").mkdir(parents=True, exist_ok=True)
    Path("docs").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    validate_pipeline_requirements()

    ddl = generate_ddl()
    json_schema = SKUMaster.model_json_schema()
    html_doc = generate_html_doc()

    Path("src/schema/sku_master_ddl.sql").write_text(ddl, encoding="utf-8")
    Path("src/schema/sku_master_schema.json").write_text(json.dumps(json_schema, indent=2), encoding="utf-8")
    Path("data/processed/sku_master_schema.json").write_text(json.dumps(json_schema, indent=2), encoding="utf-8")
    Path("docs/schema_sku_master.html").write_text(html_doc, encoding="utf-8")

    mlflow.set_experiment("geo-aware-mro")
    with mlflow.start_run(run_name="W2D2_sku_master_schema"):
        mlflow.log_param("field_count", len(FIELD_CATALOG))
        mlflow.log_param("pydantic_version", "v2")
        mlflow.log_text(json.dumps(FIELD_CATALOG, indent=2), "field_catalog.json")
        mlflow.log_artifact("src/schema/sku_master_ddl.sql")
        mlflow.log_artifact("src/schema/sku_master_schema.json")
        mlflow.log_artifact("docs/schema_sku_master.html")

    print("Artifacts written:")
    print("- src/schema/sku_master_ddl.sql")
    print("- src/schema/sku_master_schema.json")
    print("- data/processed/sku_master_schema.json")
    print("- docs/schema_sku_master.html")
    print("- MLflow run: W2D2_sku_master_schema")
    print("Commit commands:")
    print("git add src/schema/__init__.py src/schema/sku_master.py src/schema/sku_master_schema.json src/schema/sku_master_ddl.sql docs/schema_sku_master.html data/processed/sku_master_schema.json")
    print('git commit -m "feat: W2D2 -- SKU Master schema, 28 fields, Pydantic v2, DuckDB DDL"')

if __name__ == "__main__":
    write_artifacts()
