from src.optimization.nlp_lifecycle import KKTLifecycleOptimizer


def test_single_sku():

    opt = KKTLifecycleOptimizer()

    row = {
        "item_id": "SKU_TEST",
        "unit_cost": 1000.0,
        "q_star": 10.0,
        "dp_q_star": 12.0,
        "geo_risk_score": 0.2,
        "item_age_years": 2.0,
    }

    result = opt.optimize_sku(row)

    assert result.optimal_holding_qty > 0

    assert result.optimal_total_cost > 0

    assert result.kkt_stationarity < 1.0


def test_softplus_smoothing():

    opt = KKTLifecycleOptimizer()

    v1 = opt._softplus(-1.0)

    v2 = opt._softplus(1.0)

    assert v2 > v1


def test_obsolescence_probability():

    opt = KKTLifecycleOptimizer()

    p1 = opt._obsolescence_probability(1.0)

    p2 = opt._obsolescence_probability(10.0)

    assert p2 > p1
