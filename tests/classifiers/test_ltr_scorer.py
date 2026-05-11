import pandas as pd

from src.classifiers.ltr_scorer import LTRScorer


def test_ltr_score_exists():
    df = pd.DataFrame({"lead_time_days": [10, 30, 90]})

    out = LTRScorer().compute(df)

    assert "ltr_score" in out.columns
    assert len(out) == 3


def test_ltr_is_monotonic():
    df = pd.DataFrame({"lead_time_days": [10, 30, 90]})

    out = LTRScorer().compute(df)

    assert out.loc[0, "ltr_score"] < out.loc[1, "ltr_score"]
    assert out.loc[1, "ltr_score"] < out.loc[2, "ltr_score"]


def test_ltr_score_is_bounded():
    df = pd.DataFrame({"lead_time_days": [10, 30, 90]})

    out = LTRScorer().compute(df)

    assert out["ltr_score"].between(0, 1).all()
