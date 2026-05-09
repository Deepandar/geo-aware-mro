import pandas as pd


def test_supplier_columns_exist():

    df = pd.read_parquet(
        "data/processed/sku_master_v1.3.parquet"
    )

    required = [

        "supplier_risk_class",

        "supplier_risk_score",

        "procurement_flag",

        "strategic_risk_score",

        "supplier_strategy",
    ]

    for col in required:

        assert col in df.columns
