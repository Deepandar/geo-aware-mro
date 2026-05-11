from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import mlflow
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class AuctionResult:

    item_id: str

    auction_type: str

    winner_id: str

    winning_bid: float

    reserve_price: float

    n_rounds: int

    n_bidders: int

    procurement_cost_per_unit: float

    baseline_cost_per_unit: float

    cost_reduction_pct: float

    winner_geo_risk: float

    winner_reputation: float

    all_bids: list = field(default_factory=list)


@dataclass
class AuctionPortfolio:

    n_auctions: int

    n_sealed_bid: int

    n_english: int

    total_baseline_cost: float

    total_auction_cost: float

    total_saving: float

    mean_cost_reduction_pct: float

    mean_winner_geo_risk: float

    mean_winner_reputation: float

    results: List[AuctionResult] = field(default_factory=list)


# ============================================================
# ENGINE
# ============================================================

class ProcurementAuctionEngine:

    def __init__(

        self,

        n_bidders_range=(3, 7),

        reserve_price_pct=0.70,

        english_increment=0.03,

        max_rounds=25,

        n_jobs=-1,

        seed=42,

    ):

        self.n_bidders_range = n_bidders_range

        self.reserve_price_pct = reserve_price_pct

        self.english_increment = english_increment

        self.max_rounds = max_rounds

        self.n_jobs = n_jobs

        self.seed = seed

        self.rng = np.random.default_rng(seed)

    # ========================================================
    # VECTOR BIDDER GENERATION
    # ========================================================

    def _generate_bidders_batch(

        self,

        n,

        baseline_cost,

        ci_score,

        rng,

    ):

        geo = rng.uniform(
            0.05,
            0.95,
            n,
        )

        rep = rng.beta(
            4.0 + (1 - geo) * 3,
            2.0,
        )

        margin_factor = (
            1.0
            + ci_score * 0.25
            + geo * 0.15
        )

        cost_base = (
            baseline_cost
            * rng.uniform(
                0.55,
                0.90,
                n,
            )
        )

        vals = (
            cost_base
            * margin_factor
        )

        strat = rng.choice(

            ["nash", "agg", "cons"],

            size=n,

            p=[0.60, 0.25, 0.15],

        )

        return {

            "id": np.array([
                f"SUP_{chr(65+i)}"
                for i in range(n)
            ]),

            "geo": geo,

            "rep": rep,

            "vals": vals,

            "strat": strat,

        }

    # ========================================================
    # SEALED AUCTION
    # ========================================================

    def _sim_sealed(

        self,

        row,

        n,

        seed_offset=0,

    ):

        rng = np.random.default_rng(
            self.seed + seed_offset
        )

        baseline = float(
            row["unit_cost"]
        )

        reserve = (
            baseline
            * self.reserve_price_pct
        )

        floor_price = baseline * 0.20

        b = self._generate_bidders_batch(

            n,

            baseline,

            float(row["ci_score"]),

            rng,

        )

        shading = (
            (max(n, 2) - 1)
            / max(n, 2)
        )

        strat_map = {

            "nash": (0.98, 1.02),

            "agg": (0.95, 1.05),

            "cons": (0.85, 0.95),

        }

        strat = np.array([

            rng.uniform(*strat_map[s])

            for s in b["strat"]

        ])

        rep_adj = (
            1.0
            + (b["rep"] - 0.5)
            * 0.10
        )

        geo_adj = (
            1.0
            - b["geo"] * 0.15
        )

        bids = (

            b["vals"]

            * shading

            * rep_adj

            * geo_adj

            * strat

        )

        bids = np.clip(
            bids,
            floor_price,
            reserve,
        )

        valid_mask = (
            (bids >= floor_price)
            & (bids <= reserve)
        )

        if not np.any(valid_mask):

            return self._baseline_result(
                row,
                n,
                "sealed_bid",
            )

        valid_idx = np.where(
            valid_mask
        )[0]

        local = np.argmin(
            bids[valid_idx]
        )

        win_idx = valid_idx[local]

        win_bid = float(
            bids[win_idx]
        )

        reduction = (
            baseline - win_bid
        ) / baseline

        return AuctionResult(

            item_id=row["item_id"],

            auction_type="sealed_bid",

            winner_id=str(
                b["id"][win_idx]
            ),

            winning_bid=round(
                win_bid,
                2,
            ),

            reserve_price=round(
                reserve,
                2,
            ),

            n_rounds=1,

            n_bidders=n,

            procurement_cost_per_unit=round(
                win_bid,
                2,
            ),

            baseline_cost_per_unit=baseline,

            cost_reduction_pct=round(
                reduction * 100,
                2,
            ),

            winner_geo_risk=round(
                float(b["geo"][win_idx]),
                3,
            ),

            winner_reputation=round(
                float(b["rep"][win_idx]),
                3,
            ),

        )

    # ========================================================
    # ENGLISH AUCTION
    # ========================================================

    def _sim_english(

        self,

        row,

        n,

        seed_offset=0,

    ):

        rng = np.random.default_rng(
            self.seed + seed_offset
        )

        baseline = float(
            row["unit_cost"]
        )

        reserve = (
            baseline
            * self.reserve_price_pct
        )

        b = self._generate_bidders_batch(

            n,

            baseline,

            float(row["ci_score"]),

            rng,

        )

        current_price = (
            baseline * 0.25
        )

        active_mask = np.ones(
            n,
            dtype=bool,
        )

        round_num = 0

        while (

            np.sum(active_mask) > 1

            and round_num < self.max_rounds

        ):

            round_num += 1

            threshold = (
                current_price
                * (
                    1
                    + self.english_increment
                )
            )

            active_mask &= (

                b["vals"] >= threshold

            ) & (

                threshold <= reserve

            )

            current_price = threshold

        if np.sum(active_mask) == 0:

            return self._baseline_result(
                row,
                n,
                "english",
            )

        active_idx = np.where(
            active_mask
        )[0]

        local = np.argmin(
            b["vals"][active_idx]
        )

        win_idx = active_idx[local]

        final_price = min(
            current_price,
            reserve,
        )

        reduction = (
            baseline - final_price
        ) / baseline

        return AuctionResult(

            item_id=row["item_id"],

            auction_type="english",

            winner_id=str(
                b["id"][win_idx]
            ),

            winning_bid=round(
                final_price,
                2,
            ),

            reserve_price=round(
                reserve,
                2,
            ),

            n_rounds=round_num,

            n_bidders=n,

            procurement_cost_per_unit=round(
                final_price,
                2,
            ),

            baseline_cost_per_unit=baseline,

            cost_reduction_pct=round(
                reduction * 100,
                2,
            ),

            winner_geo_risk=round(
                float(b["geo"][win_idx]),
                3,
            ),

            winner_reputation=round(
                float(b["rep"][win_idx]),
                3,
            ),

        )

    # ========================================================
    # BASELINE RESULT
    # ========================================================

    def _baseline_result(

        self,

        row,

        n,

        auction_type,

    ):

        baseline = float(
            row["unit_cost"]
        )

        return AuctionResult(

            item_id=row["item_id"],

            auction_type=auction_type,

            winner_id="BASELINE",

            winning_bid=baseline,

            reserve_price=baseline
            * self.reserve_price_pct,

            n_rounds=1,

            n_bidders=n,

            procurement_cost_per_unit=baseline,

            baseline_cost_per_unit=baseline,

            cost_reduction_pct=0.0,

            winner_geo_risk=float(
                row.get(
                    "geo_risk_score",
                    0.5,
                )
            ),

            winner_reputation=0.5,

        )

    # ========================================================
    # PARALLEL PORTFOLIO
    # ========================================================

    def run_portfolio_auctions(

        self,

        sku_df,

        target_tiers=None,

    ):

        target_tiers = target_tiers or ["High"]

        eligible = sku_df[
            sku_df["ci_tier"].isin(
                target_tiers
            )
        ].to_dict("records")

        def process(row_idx_row):

            idx, row = row_idx_row

            n = int(

                self.rng.integers(

                    self.n_bidders_range[0],

                    self.n_bidders_range[1] + 1,

                )

            )

            use_english = (

                row.get("ci_score", 0.0) > 0.75

                and row.get("ved_class") == "V"

            )

            if use_english:

                return self._sim_english(
                    row,
                    n,
                    seed_offset=idx,
                )

            return self._sim_sealed(
                row,
                n,
                seed_offset=idx,
            )

        results = Parallel(

            n_jobs=self.n_jobs,

            backend="loky",

        )(

            delayed(process)(x)

            for x in enumerate(eligible)

        )

        if not results:

            return AuctionPortfolio(

                n_auctions=0,

                n_sealed_bid=0,

                n_english=0,

                total_baseline_cost=0.0,

                total_auction_cost=0.0,

                total_saving=0.0,

                mean_cost_reduction_pct=0.0,

                mean_winner_geo_risk=0.0,

                mean_winner_reputation=0.0,

            )

        rdf = pd.DataFrame([

            {

                "base": r.baseline_cost_per_unit,

                "auction": r.procurement_cost_per_unit,

                "reduction": r.cost_reduction_pct,

                "geo": r.winner_geo_risk,

                "rep": r.winner_reputation,

                "type": r.auction_type,

            }

            for r in results

        ])

        baseline = float(rdf["base"].sum())

        auction = float(rdf["auction"].sum())

        savings = baseline - auction

        return AuctionPortfolio(

            n_auctions=len(results),

            n_sealed_bid=int(
                (
                    rdf["type"]
                    == "sealed_bid"
                ).sum()
            ),

            n_english=int(
                (
                    rdf["type"]
                    == "english"
                ).sum()
            ),

            total_baseline_cost=round(
                baseline,
                2,
            ),

            total_auction_cost=round(
                auction,
                2,
            ),

            total_saving=round(
                savings,
                2,
            ),

            mean_cost_reduction_pct=round(

                float(
                    rdf["reduction"].mean()
                ),

                2,

            ),

            mean_winner_geo_risk=round(

                float(
                    rdf["geo"].mean()
                ),

                3,

            ),

            mean_winner_reputation=round(

                float(
                    rdf["rep"].mean()
                ),

                3,

            ),

            results=results,

        )

    # ========================================================
    # MLFLOW
    # ========================================================

    def log_to_mlflow(

        self,

        portfolio,

        run_name="w27_parallel",

    ):

        with mlflow.start_run(
            run_name=run_name
        ):

            mlflow.log_metric(
                "n_auctions",
                portfolio.n_auctions,
            )

            mlflow.log_metric(
                "total_saving",
                portfolio.total_saving,
            )

            mlflow.log_metric(
                "mean_reduction",
                portfolio.mean_cost_reduction_pct,
            )


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    np.random.seed(42)

    df = pd.DataFrame({

        "item_id": [
            f"SKU{i:04d}"
            for i in range(500)
        ],

        "ci_tier": np.random.choice(
            ["High", "Medium", "Low"],
            500,
        ),

        "ci_score": np.random.uniform(
            0.3,
            0.95,
            500,
        ),

        "ved_class": np.random.choice(
            ["V", "E", "D"],
            500,
        ),

        "unit_cost": np.random.uniform(
            500,
            20000,
            500,
        ),

        "geo_risk_score": np.random.uniform(
            0,
            0.8,
            500,
        ),

    })

    engine = ProcurementAuctionEngine()

    p = engine.run_portfolio_auctions(df)

    print("\n==================================================")
    print("W27.2 PARALLEL PROCUREMENT RESULTS")
    print("==================================================")
    print(f"Auctions        : {p.n_auctions}")
    print(f"Sealed Bid      : {p.n_sealed_bid}")
    print(f"English         : {p.n_english}")
    print(f"Total Saving    : {p.total_saving:.2f}")
    print(f"Mean Reduction  : {p.mean_cost_reduction_pct:.2f}%")
    print("==================================================")
