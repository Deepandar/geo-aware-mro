import numpy as np
import pandas as pd

from src.simulation.depot_sim import (
    DepotSimulator,
)


def build_df():

    np.random.seed(42)

    return pd.DataFrame({

        "item_id": [
            f"SKU{i:03d}"
            for i in range(20)
        ],

        "fns_class":
            np.random.choice(
                ["F", "N", "S"],
                20,
            ),

        "mean_demand":
            np.random.uniform(
                5,
                50,
                20,
            ),

        "std_demand":
            np.random.uniform(
                1,
                10,
                20,
            ),

        "q_star":
            np.random.uniform(
                10,
                40,
                20,
            ),

        "rop":
            np.random.uniform(
                5,
                20,
                20,
            ),

        "ci_tier":
            np.random.choice(
                ["High", "Medium", "Low"],
                20,
            ),
    })


def test_sim_runs():

    df = build_df()

    sim = DepotSimulator(
        sim_periods=12,
        n_trials=5,
        fast_mode=True,
    )

    out = sim.run_monte_carlo(df)

    assert len(out) == 5


def test_fill_rate_bounded():

    df = build_df()

    sim = DepotSimulator(
        sim_periods=10,
        n_trials=3,
        fast_mode=True,
    )

    out = sim.run_monte_carlo(df)

    assert (
        out["mean_fill_rate"]
        .between(0, 1)
        .all()
    )


def test_total_cost_positive():

    df = build_df()

    sim = DepotSimulator(
        sim_periods=10,
        n_trials=3,
        fast_mode=True,
    )

    out = sim.run_monte_carlo(df)

    assert (
        out["total_cost"] >= 0
    ).all()

