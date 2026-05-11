import pandas as pd
from src.classifiers.newsvendor import NewsvendorEngine


def test_newsvendor_outputs_positive():
    df = pd.DataFrame(
        {
            "item_id": ["SKU1"],
            "abc_class": ["A"],
            "ved_class": ["V"],
            "fns_class": ["F"],
            "demand": [50],
            "lead_time_days": [30],
        }
    )

    result = NewsvendorEngine().compute(df)

    assert result["q_star"].iloc[0] > 0
    assert result["rop"].iloc[0] > 0
