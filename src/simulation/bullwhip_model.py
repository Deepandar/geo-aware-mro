from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EchelonData:

    echelon_id: int
    echelon_name: str

    demands: np.ndarray
    orders: np.ndarray

    @property
    def demand_variance(self) -> float:

        if len(self.demands) <= 1:
            return 0.0

        return float(
            np.var(
                self.demands,
                ddof=1,
            )
        )

    @property
    def order_variance(self) -> float:

        if len(self.orders) <= 1:
            return 0.0

        return float(
            np.var(
                self.orders,
                ddof=1,
            )
        )

    @property
    def bullwhip_ratio(self) -> float:

        if self.demand_variance <= 0:
            return 1.0

        return self.order_variance / self.demand_variance


@dataclass
class BullwhipResult:

    item_id: str

    ci_tier: str

    fns_class: str

    echelons: list[EchelonData]

    amplification_ratios: list[float]

    total_amplification: float

    policy_type: str

    lead_times: list[float]

    n_periods: int


@dataclass
class BullwhipSummary:

    n_skus: int

    policy_type: str

    mean_bwr_echelon_1: float

    mean_bwr_echelon_2: float

    mean_bwr_echelon_3: float

    mean_total_amplification: float

    std_total_amplification: float

    pct_bwr_gt_1: float

    high_tier_bwr: float

    medium_tier_bwr: float

    low_tier_bwr: float

    sku_results: list[BullwhipResult] = field(default_factory=list)


class BullwhipModel:

    ECHELON_NAMES = [
        "End-User",
        "Forward Depot",
        "Border Depot",
        "Central Supply",
    ]

    def __init__(
        self,
        n_periods: int = 52,
        n_echelons: int = 4,
        lead_times: list[float] | None = None,
        smoothing_alpha: float = 0.20,
        seed: int = 42,
    ):

        self.n_periods = n_periods

        self.n_echelons = n_echelons

        self.lead_times = lead_times or [1, 2, 3, 4]

        self.smoothing_alpha = smoothing_alpha

        self.seed = seed

        if len(self.lead_times) != self.n_echelons:

            raise ValueError("lead_times length must equal n_echelons")

    # =====================================================
    # DEMAND GENERATION
    # =====================================================

    def _generate_end_demand(
        self,
        mean_demand: float,
        std_demand: float,
        rng: np.random.Generator,
    ) -> np.ndarray:

        demand = rng.normal(
            loc=mean_demand,
            scale=max(
                std_demand,
                0.1,
            ),
            size=self.n_periods,
        )

        return np.maximum(
            demand,
            0,
        )

    # =====================================================
    # POLICY AMPLIFICATION
    # =====================================================

    def _policy_multiplier(
        self,
        policy_type: str,
        echelon: int,
    ) -> float:

        if echelon == 0:
            return 1.0

        if policy_type == "standard":
            return 1.25 + echelon * 0.45

        if policy_type == "dp_optimized":
            return 1.05 + echelon * 0.20

        # CODP optimized

        return 1.02 + echelon * 0.12

    # =====================================================
    # SINGLE SKU
    # =====================================================

    def analyze_sku(
        self,
        row: dict,
        policy_type: str = "dp_optimized",
        rng: Optional[np.random.Generator] = None,
    ) -> BullwhipResult:

        if rng is None:

            rng = np.random.default_rng(self.seed)

        mean_demand = float(
            row.get(
                "mean_demand",
                10.0,
            )
        )

        std_demand = float(
            row.get(
                "std_demand",
                3.0,
            )
        )

        ci_tier = str(
            row.get(
                "ci_tier",
                "Medium",
            )
        )

        fns_class = str(
            row.get(
                "fns_class",
                "N",
            )
        )

        end_demand = self._generate_end_demand(
            mean_demand,
            std_demand,
            rng,
        )

        echelons = []

        amplification_ratios = []

        prev_orders = end_demand.copy()

        for echelon in range(self.n_echelons):

            multiplier = self._policy_multiplier(
                policy_type,
                echelon,
            )

            orders = prev_orders * multiplier

            orders += rng.normal(
                0,
                std_demand * 0.2,
                self.n_periods,
            )

            orders = np.maximum(
                orders,
                0,
            )

            if echelon == 0:

                orders = end_demand.copy()

            echelon_data = EchelonData(
                echelon_id=echelon,
                echelon_name=self.ECHELON_NAMES[echelon],
                demands=prev_orders.copy(),
                orders=orders.copy(),
            )

            echelons.append(echelon_data)

            amplification_ratios.append(echelon_data.bullwhip_ratio)

            prev_orders = orders.copy()

        total_amp = amplification_ratios[-1]

        return BullwhipResult(
            item_id=str(
                row.get(
                    "item_id",
                    "UNKNOWN",
                )
            ),
            ci_tier=ci_tier,
            fns_class=fns_class,
            echelons=echelons,
            amplification_ratios=amplification_ratios,
            total_amplification=total_amp,
            policy_type=policy_type,
            lead_times=self.lead_times,
            n_periods=self.n_periods,
        )

    # =====================================================
    # ALL SKUS
    # =====================================================

    def analyze_all(
        self,
        df: pd.DataFrame,
        policy_type: str = "dp_optimized",
    ) -> BullwhipSummary:

        rng = np.random.default_rng(self.seed)

        results = []

        for _, row in df.iterrows():

            results.append(
                self.analyze_sku(
                    row.to_dict(),
                    policy_type,
                    rng,
                )
            )

        total_amps = [r.total_amplification for r in results]

        def tier_mean(tier: str) -> float:

            vals = [r.total_amplification for r in results if r.ci_tier == tier]

            if not vals:
                return 1.0

            return float(np.mean(vals))

        return BullwhipSummary(
            n_skus=len(results),
            policy_type=policy_type,
            mean_bwr_echelon_1=float(
                np.mean([r.amplification_ratios[1] for r in results])
            ),
            mean_bwr_echelon_2=float(
                np.mean([r.amplification_ratios[2] for r in results])
            ),
            mean_bwr_echelon_3=float(
                np.mean([r.amplification_ratios[3] for r in results])
            ),
            mean_total_amplification=float(np.mean(total_amps)),
            std_total_amplification=float(np.std(total_amps)),
            pct_bwr_gt_1=float(np.mean([a > 1.0 for a in total_amps])),
            high_tier_bwr=tier_mean("High"),
            medium_tier_bwr=tier_mean("Medium"),
            low_tier_bwr=tier_mean("Low"),
            sku_results=results,
        )

    # =====================================================
    # POLICY COMPARISON
    # =====================================================

    def compare_policies(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        rows = []

        for policy in [
            "standard",
            "dp_optimized",
            "codp",
        ]:

            summary = self.analyze_all(
                df,
                policy,
            )

            rows.append(
                {
                    "policy": policy,
                    "mean_total_amp": summary.mean_total_amplification,
                }
            )

        return pd.DataFrame(rows)

    # =====================================================
    # EXPORT
    # =====================================================

    def to_dataframe(
        self,
        summary: BullwhipSummary,
    ) -> pd.DataFrame:

        rows = []

        for r in summary.sku_results:

            row = {
                "item_id": r.item_id,
                "ci_tier": r.ci_tier,
                "fns_class": r.fns_class,
                "policy_type": r.policy_type,
                "total_amplification": r.total_amplification,
            }

            for i, bwr in enumerate(r.amplification_ratios):

                row[f"bwr_e{i}"] = bwr

            rows.append(row)

        return pd.DataFrame(rows)
