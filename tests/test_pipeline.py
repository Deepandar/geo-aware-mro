import pandas as pd

from src.pipelines.sku_pipeline import run_pipeline


def test_pipeline_runs(tmp_path):
    df = pd.DataFrame({
        "unit_cost": [10],
        "demand": [100],
        "stockout_cost_usd": [10000],
        "depot_tier": ["Forward"],
        "lead_time_days": [30]
    })

    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"

    df.to_csv(input_file, index=False)

    result = run_pipeline(str(input_file), str(output_file))

    assert "ci_score" in result.columns
    assert "q_star" in result.columns
    assert "rop" in result.columns
