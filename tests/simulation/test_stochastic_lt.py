import pytest
import numpy as np
import pandas as pd

from src.simulation.lead_time_fitter import (
    StochasticLeadTimeFitter,
    CLUSTER_GAMMA_PARAMS,
)


@pytest.fixture
def fitter():

    return StochasticLeadTimeFitter(
        penalty_factor=0.5,
        scenario="baseline",
    )


@pytest.fixture
def rng():

    return np.random.default_rng(42)


def test_sample_lead_time_positive(
    fitter,
    rng,
):

    for country in [
        "IN",
        "CN",
        "RU",
        "UA",
        "US",
    ]:

        lt = fitter.sample_lead_time(
            country,
            geo_risk=0.3,
            baseline_lt=45,
            rng=rng,
        )

        assert lt > 0


def test_high_risk_country_longer_lt_than_low_risk(
    rng,
):

    fitter = StochasticLeadTimeFitter()

    n = 500

    ru_lts = fitter.sample_batch(
        "RU",
        0.8,
        45,
        n,
        rng,
    )

    in_lts = fitter.sample_batch(
        "IN",
        0.1,
        45,
        n,
        rng,
    )

    assert ru_lts.mean() > in_lts.mean()


def test_sanctions_scenario_increases_high_risk_lt(
    rng,
):

    base_fitter = StochasticLeadTimeFitter(scenario="baseline")

    sanc_fitter = StochasticLeadTimeFitter(scenario="sanctions")

    n = 300

    base_lts = base_fitter.sample_batch(
        "RU",
        0.8,
        45,
        n,
        rng,
    )

    sanc_lts = sanc_fitter.sample_batch(
        "RU",
        0.8,
        45,
        n,
        rng,
    )

    assert sanc_lts.mean() > base_lts.mean()


def test_geo_risk_penalty_increases_lt(
    fitter,
    rng,
):

    low_geo = np.mean(
        [
            fitter.sample_lead_time(
                "CN",
                0.1,
                45,
                rng,
            )
            for _ in range(300)
        ]
    )

    high_geo = np.mean(
        [
            fitter.sample_lead_time(
                "CN",
                0.9,
                45,
                rng,
            )
            for _ in range(300)
        ]
    )

    assert high_geo > low_geo


def test_unknown_country_uses_unknown_cluster(
    fitter,
    rng,
):

    lt = fitter.sample_lead_time(
        "ZZ",
        geo_risk=0.3,
        baseline_lt=30,
        rng=rng,
    )

    assert lt > 0


def test_distribution_summary_has_all_clusters(
    fitter,
):

    df = fitter.distribution_summary()

    clusters = set(df["cluster"].tolist())

    expected = set(CLUSTER_GAMMA_PARAMS.keys())

    assert expected.issubset(clusters)


def test_generate_overrides_returns_dicts(
    fitter,
):

    df = pd.DataFrame(
        {
            "item_id": [
                "A",
                "B",
                "C",
            ],
            "supply_origin_country": [
                "IN",
                "RU",
                "CN",
            ],
            "geo_risk_score": [
                0.1,
                0.8,
                0.5,
            ],
            "mean_lead_time": [
                14,
                90,
                45,
            ],
        }
    )

    mu_ov, sig_ov = fitter.generate_overrides(df)

    assert set(mu_ov.keys()) == {
        "A",
        "B",
        "C",
    }

    assert set(sig_ov.keys()) == {
        "A",
        "B",
        "C",
    }


def test_mle_fit_updates_distribution(
    fitter,
):

    obs = np.random.gamma(
        shape=3.0,
        scale=20.0,
        size=100,
    )

    fitted = fitter.fit_to_observed(
        obs,
        country="IN",
    )

    assert abs(fitted.mean_lt - 60.0) / 60.0 < 0.40
