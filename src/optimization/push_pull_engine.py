from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DecouplingPolicy:

    item_id: str

    decoupling_mode: str

    push_qty: float

    pull_qty: float

    total_position: float

    codp_tier: str

    rationale: str


class PushPullEngine:

    def __init__(
        self,
        push_density_threshold: float = 0.50,
        pull_rul_threshold: float = 20.0,
        push_weight: float = 0.60,
    ):

        self.push_density_threshold = push_density_threshold

        self.pull_rul_threshold = pull_rul_threshold

        self.push_weight = push_weight

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    def _validate(
        self,
        df: pd.DataFrame,
    ):

        required = {
            "item_id",
            "equipment_density_score",
            "rul_signal",
        }

        missing = required - set(df.columns)

        if missing:

            raise ValueError(f"Missing columns: {missing}")

    # -----------------------------------------------------
    # MAIN COMPUTE
    # -----------------------------------------------------

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        self._validate(df)

        out = df.copy()

        modes = []

        pushes = []

        pulls = []

        totals = []

        codps = []

        rationales = []

        for _, row in out.iterrows():

            density = float(row["equipment_density_score"])

            rul = float(row["rul_signal"])

            depot = str(
                row.get(
                    "depot_tier",
                    "Rear",
                )
            )

            base_stock = float(
                row.get(
                    "base_stock_level",
                    row.get(
                        "q_star",
                        10.0,
                    ),
                )
            )

            push = density > self.push_density_threshold

            pull = rul < self.pull_rul_threshold

            push_qty = base_stock * self.push_weight if push else 0.0

            if pull:

                urgency = max(
                    0.0,
                    1.0 - (rul / self.pull_rul_threshold),
                )

                pull_qty = base_stock * urgency

            else:

                pull_qty = 0.0

            total = push_qty + pull_qty

            # ---------------------------------------------
            # MODE LOGIC
            # ---------------------------------------------

            if push and pull:

                mode = "Push+Pull"

                codp = "Forward"

            elif push:

                mode = "Push"

                codp = "Border"

            elif pull:

                mode = "Pull"

                codp = depot

            else:

                mode = "Newsvendor"

                codp = "Rear"

            rationale = f"density={density:.2f} | " f"RUL={rul:.1f}"

            modes.append(mode)

            pushes.append(round(push_qty, 2))

            pulls.append(round(pull_qty, 2))

            totals.append(round(total, 2))

            codps.append(codp)

            rationales.append(rationale)

        out["decoupling_mode"] = modes

        out["push_qty"] = pushes

        out["pull_qty"] = pulls

        out["total_position"] = totals

        out["codp_tier"] = codps

        out["pp_rationale"] = rationales

        out["pull_justified"] = out["pull_qty"] > 0

        logger.info(
            "PushPullEngine complete | " "Push+Pull=%d",
            int((out["decoupling_mode"] == "Push+Pull").sum()),
        )

        return out

    # -----------------------------------------------------
    # DECOUPLING MAP
    # -----------------------------------------------------

    def decoupling_map(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        return (
            df.groupby(
                [
                    "depot_tier",
                    "decoupling_mode",
                ]
            )
            .agg(
                sku_count=(
                    "item_id",
                    "count",
                ),
                mean_push_qty=(
                    "push_qty",
                    "mean",
                ),
                mean_pull_qty=(
                    "pull_qty",
                    "mean",
                ),
            )
            .reset_index()
        )
