import pandas as pd

from src.suppliers.nash_equilibrium import (
    NashEquilibriumEngine,
)


def test_nash_runs():

    df = pd.DataFrame({

        "supplier_risk_score": [
            0.1,
            0.5,
            0.9,
        ],

        "ci_score": [
            0.2,
            0.6,
            0.9,
        ],
    })

    engine = NashEquilibriumEngine()

    out = engine.compute(df)

    assert (
        "strategic_risk_score"
        in out.columns
    )

    assert (
        "supplier_strategy"
        in out.columns
    )
