import pandas as pd

from src.optimization.network_optimizer import MILPNetworkOptimizer


def sample_df():

    return pd.DataFrame(
        {
            "item_id": ["SKU1", "SKU2"],
            "unit_cost": [1000, 2000],
            "q_star": [10, 20],
            "geo_risk_score": [0.2, 0.5],
            "volume_unit": [50, 100],
            "weight_unit": [100, 200],
        }
    )


def test_optimizer_runs():

    opt = MILPNetworkOptimizer()

    result = opt.solve(sample_df())

    assert result.status == "Optimal"


def test_sensitivity_report():

    opt = MILPNetworkOptimizer()

    opt.solve(sample_df())

    report = opt.get_sensitivity_report()

    assert isinstance(report, dict)

    assert len(report) > 0


def test_mlflow_tracking():

    opt = MILPNetworkOptimizer()

    result = opt.solve_with_tracking(sample_df(), experiment_name="pytest_tracking")

    assert result.status == "Optimal"
