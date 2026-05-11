# tests/optimization/test_bellman_engine.py


from src.optimization.bellman_engine import (
    BellmanEngine,
)

from src.data_ingestion.synthetic_sku_master import (
    generate_sku_master,
)

from src.classifiers.abc_classifier import (
    classify_abc,
)

from src.classifiers.ved_classifier import (
    classify_ved,
)

from src.classifiers.fns_classifier import (
    classify_fns,
)

from src.classifiers.location_scorer import (
    LocationScorer,
)

from src.geo.risk_scorer import (
    BayesianRiskScorer,
)

from src.classifiers.ltr_scorer import (
    LTRScorer,
)

from src.classifiers.criticality_index import (
    CriticalityIndexer,
)


def build_df():

    df = generate_sku_master(n_skus=100)

    df = classify_abc(df)

    df = classify_ved(df)

    df = classify_fns(df)

    scorer = LocationScorer()

    df = scorer.score(df)

    risk = BayesianRiskScorer()

    df = risk.score(df)

    ltr = LTRScorer()

    df = ltr.compute(df)

    ci = CriticalityIndexer()

    df = ci.compute(df)

    return df


def test_bellman_runs():

    df = build_df()

    engine = BellmanEngine()

    result = engine.compute(df)

    assert not result.empty


def test_bellman_columns_exist():

    df = build_df()

    engine = BellmanEngine()

    result = engine.compute(df)

    required = [
        "bellman_q_star",
        "bellman_rop",
        "expected_future_cost",
        "state_value",
    ]

    for col in required:

        assert col in result.columns


def test_bellman_qstar_positive():

    df = build_df()

    engine = BellmanEngine()

    result = engine.compute(df)

    assert (result["bellman_q_star"] >= 0).all()


def test_bellman_rop_positive():

    df = build_df()

    engine = BellmanEngine()

    result = engine.compute(df)

    assert (result["bellman_rop"] > 0).all()


def test_bellman_rop_bounded():

    df = build_df()

    engine = BellmanEngine()

    result = engine.compute(df)

    assert (result["bellman_rop"] <= 50000).all()


def test_future_cost_finite():

    df = build_df()

    engine = BellmanEngine()

    result = engine.compute(df)

    assert result["expected_future_cost"].notna().all()


def test_state_value_finite():

    df = build_df()

    engine = BellmanEngine()

    result = engine.compute(df)

    assert result["state_value"].notna().all()
