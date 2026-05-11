import pandas as pd

from src.classifiers.location_scorer import LocationScorer


def test_location_scores_exist():
    df = pd.DataFrame({"depot_tier": ["Forward", "Border", "Rear"]})

    out = LocationScorer().score(df)

    assert "location_score" in out.columns
    assert len(out) == 3


def test_forward_has_higher_score_than_rear():
    df = pd.DataFrame({"depot_tier": ["Forward", "Rear"]})

    out = LocationScorer().score(df)

    forward = out.loc[0, "location_score"]
    rear = out.loc[1, "location_score"]

    assert forward > rear


def test_location_score_is_normalized():
    df = pd.DataFrame({"depot_tier": ["Forward", "Border", "Rear"]})

    out = LocationScorer().score(df)

    assert out["location_score"].between(0, 1).all()
