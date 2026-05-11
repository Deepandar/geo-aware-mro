import numpy as np
import pandas as pd
import pytest

from src.suppliers.shapley_allocation import (
    ShapleyAllocator,
    ShapleyAllocation,
    cooperative_reward,
)


@pytest.fixture
def sku_df():

    np.random.seed(42)

    n = 90

    depot_assignment = ["Forward"] * 30 + ["Border"] * 35 + ["Rear"] * 25

    return pd.DataFrame(
        {
            "item_id": [f"SKU{i:04d}" for i in range(n)],
            "depot_tier": depot_assignment,
            "unit_cost": np.random.uniform(
                100,
                5000,
                n,
            ),
            "q_star": np.random.uniform(
                5,
                30,
                n,
            ),
            "dp_q_star": np.random.uniform(
                8,
                35,
                n,
            ),
            "ci_score": np.random.uniform(
                0.2,
                0.95,
                n,
            ),
        }
    )


@pytest.fixture
def allocator():

    return ShapleyAllocator()


def test_allocation_returns_result(
    allocator,
    sku_df,
):

    alloc = allocator.allocate(sku_df)

    assert isinstance(
        alloc,
        ShapleyAllocation,
    )


def test_efficiency_property(
    allocator,
    sku_df,
):

    alloc = allocator.allocate(sku_df)

    shapley_sum = sum(r.shapley_value for r in alloc.depot_results)

    assert abs(shapley_sum - alloc.coalition_cost) < alloc.coalition_cost * 0.02


def test_pooling_reduces_cost(
    allocator,
    sku_df,
):

    alloc = allocator.allocate(sku_df)

    assert alloc.coalition_cost <= alloc.sum_solo_costs


def test_coalition_rows(
    allocator,
    sku_df,
):

    alloc = allocator.allocate(sku_df)

    assert len(alloc.coalition_table) == 8


def test_rl_state_vector(
    allocator,
):

    state = allocator.build_rl_state_vector(
        inventory_level=100.0,
        failure_risk=0.2,
        geo_risk=0.4,
        coalition_saving=5000.0,
        supplier_rep=0.9,
    )

    assert len(state) == 5


def test_reward_function():

    reward = cooperative_reward(
        shapley_saving=10000.0,
        shortage_penalty=2000.0,
        geo_risk=0.3,
        coalition_health=0.9,
    )

    assert isinstance(
        reward,
        float,
    )


def test_missing_columns_raises(
    allocator,
):

    bad_df = pd.DataFrame(
        {
            "item_id": ["X"],
            "depot_tier": ["Forward"],
        }
    )

    with pytest.raises(ValueError):

        allocator.allocate(bad_df)
