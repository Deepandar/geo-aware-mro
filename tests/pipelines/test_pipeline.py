import pytest

from src.pipelines.sku_pipeline import run_pipeline


@pytest.fixture(scope="module")
def pipeline_output():
    """
    Run pipeline once for all tests.
    """
    return run_pipeline(n_skus=25)


def test_pipeline_runs(pipeline_output):
    assert len(pipeline_output) == 25


def test_pipeline_outputs_exist(pipeline_output):
    required_cols = [
        "abc_class",
        "ved_class",
        "fns_class",
        "location_score",
        "ltr_score",
        "ci_score",
        "ci_tier",
        "tsl",
        "q_star",
        "rop",
    ]

    for col in required_cols:
        assert col in pipeline_output.columns


def test_ci_score_is_bounded(pipeline_output):
    assert pipeline_output["ci_score"].between(0, 1).all()


def test_tsl_is_bounded(pipeline_output):
    assert pipeline_output["tsl"].between(0, 1).all()


def test_qstar_positive(pipeline_output):
    assert (pipeline_output["q_star"] > 0).all()


def test_rop_positive(pipeline_output):
    assert (pipeline_output["rop"] > 0).all()
