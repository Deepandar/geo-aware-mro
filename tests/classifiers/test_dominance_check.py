import pandas as pd

from src.classifiers.dominance_check import DominanceChecker


def test_detects_dominance():
    df = pd.DataFrame(
        {
            "annual_consumption_value": [
                100000,
                1000,
                800,
                500,
                200,
            ]
        }
    )

    _, result = DominanceChecker().check_and_remediate(df)

    assert result["bias_detected"] is True


def test_creates_acv_for_abc():
    df = pd.DataFrame(
        {
            "annual_consumption_value": [
                100000,
                1000,
                800,
                500,
                200,
            ]
        }
    )

    out, _ = DominanceChecker().check_and_remediate(df)

    assert "acv_for_abc" in out.columns


def test_remediation_changes_distribution():
    df = pd.DataFrame(
        {
            "annual_consumption_value": [
                100000,
                1000,
                800,
                500,
                200,
            ]
        }
    )

    out, _ = DominanceChecker().check_and_remediate(df)

    original_max = df["annual_consumption_value"].max()
    remediated_max = out["acv_for_abc"].max()

    assert remediated_max < original_max
