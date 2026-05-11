import pandas as pd

from week3_day4_final_scorer import compute_composite_index


def test_environment_fields_created_and_consistent(tmp_path):
    # Minimal synthetic input covering ABC/VED/FNS + locationtype
    df = pd.DataFrame(
        {
            "item_id": ["X1", "X2"],
            "abcclass": ["A", "C"],
            "vedclass": ["V", "D"],
            "fnsclass": ["F", "S"],
            "locationtype": ["Forward", "Rear"],
        }
    )

    out = compute_composite_index(df)

    # Columns must exist
    required = {
        "locscore",
        "environment_multiplier",
        "base_position_score",
        "location_score_adj",
        "ci",
    }
    missing = required - set(out.columns)
    assert not missing, f"Missing columns: {missing}"

    # Neutral environment in v1.1
    assert (out["environment_multiplier"] == 1.0).all()

    # base_position_score mirrors locscore
    pd.testing.assert_series_equal(
        out["base_position_score"],
        out["locscore"],
        check_names=False,
    )

    # Adjusted location score matches base × env
    expected_adj = out["base_position_score"] * out["environment_multiplier"]
    pd.testing.assert_series_equal(
        out["location_score_adj"],
        expected_adj,
        check_names=False,
    )

    # Composite index uses adjusted location score, not raw locscore
    # Recompute a manual ci and compare
    from week3_day4_final_scorer import WEIGHTS

    manual_ci = (
        WEIGHTS["abc"] * out["abcscore"]
        + WEIGHTS["ved"] * out["vedscore"]
        + WEIGHTS["fns"] * out["fnsscore"]
        + WEIGHTS["loc"] * out["location_score_adj"]
    ).clip(0, 1).round(4)

    pd.testing.assert_series_equal(
        out["ci"],
        manual_ci,
        check_names=False,
    )
