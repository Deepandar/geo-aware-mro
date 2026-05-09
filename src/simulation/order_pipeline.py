"""
Order Pipeline — v1.2
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OrderRecord:

    period: float
    trigger: str
    order_qty: float
    lead_time: float
    arrival_period: float
    geo_risk_at_order: float


@dataclass
class SKUInventoryState:

    item_id: str

    reorder_point: float
    order_qty: float

    inventory: float

    push_qty: float = 0.0
    pull_trigger_active: bool = False
    sourcing_strategy: str = "Single"

    on_order: float = 0.0

    total_demand: float = 0.0
    total_filled: float = 0.0
    total_stockout: float = 0.0

    total_holding_cost: float = 0.0
    total_stockout_cost: float = 0.0
    total_order_cost: float = 0.0

    stockout_events: int = 0

    order_records: list = field(
        default_factory=list
    )

    unit_cost: float = 100.0
    stockout_cost_usd: float = 500.0
    holding_cost_rate: float = 0.20

    @property
    def fill_rate(self):

        if self.total_demand <= 0:
            return 1.0

        return (
            self.total_filled
            / self.total_demand
        )

