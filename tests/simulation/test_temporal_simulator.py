# tests/simulation/test_temporal_simulator.py

from src.classifiers.abc_classifier import (
    classify_abc,
)

from src.classifiers.fns_classifier import (
    classify_fns,
)

from src.classifiers.ved_classifier import (
    classify_ved,
)

from src.data_ingestion.synthetic_sku_master import (
    generate_sku_master,
)

from src.simulation.temporal_simulator import (
    TemporalSimulator,
)


def build_simulation_df(
    n_skus: int = 100,
):

    df = generate_sku_master(
        n_skus=n_skus
    )

    # -----------------------------------------------------
    # Required classification stages
    # -----------------------------------------------------

    df = classify_abc(df)

    df = classify_ved(df)

    df = classify_fns(df)

    return df


def test_temporal_simulator_runs():

    df = build_simulation_df(
        n_skus=100
    )

    sim = TemporalSimulator(
        simulation_horizon=5
    )

    result = sim.run(df)

    assert not result.empty

    required_cols = [
        "simulation_step",
        "active_disruptions",
        "mean_ltr",
        "mean_ci",
    ]

    for col in required_cols:

        assert col in result.columns


def test_temporal_simulator_generates_steps():

    df = build_simulation_df(
        n_skus=50
    )

    sim = TemporalSimulator(
        simulation_horizon=4
    )

    result = sim.run(df)

    assert (
        result["simulation_step"]
        .nunique()
        == 4
    )


def test_temporal_simulator_metrics_bounded():

    df = build_simulation_df(
        n_skus=50
    )

    sim = TemporalSimulator(
        simulation_horizon=3
    )

    result = sim.run(df)

    assert (
        result["mean_ltr"]
        .between(0, 1)
        .all()
    )

    assert (
        result["mean_ci"]
        .between(0, 1)
        .all()
    )


def test_temporal_simulator_disruptions_nonnegative():

    df = build_simulation_df(
        n_skus=50
    )

    sim = TemporalSimulator(
        simulation_horizon=3
    )

    result = sim.run(df)

    assert (
        result["active_disruptions"]
        >= 0
    ).all()