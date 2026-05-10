from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class SupplierHistory:

    item_id: str

    periods: int

    on_time_rate: float

    defection_prob: float

    late_threshold_days: float

    geo_risk_score: float = 0.0

    supplier_risk_class: str = "Medium"

    deliveries: list[float] = field(
        default_factory=list
    )

    defections: list[bool] = field(
        default_factory=list
    )

    def simulate(
        self,
        rng: np.random.Generator,
    ) -> None:

        self.deliveries = []
        self.defections = []

        for _ in range(self.periods):

            late_days = max(
                0.0,
                rng.normal(
                    loc=3.0,
                    scale=4.0,
                )
            )

            defected = (
                rng.random()
                <
                self.defection_prob
            )

            if defected:

                late_days += rng.uniform(
                    5.0,
                    15.0,
                )

            self.deliveries.append(
                late_days
            )

            self.defections.append(
                defected
            )


@dataclass
class ReputationResult:

    item_id: str

    reputation_score: float

    grim_trigger_fired: bool

    n_defections: int

    n_periods: int

    delta_satisfied: bool

    delta_value: float

    cooperation_surplus: float

    defection_gain: float

    recommended_action: str


# =========================================================
# REPEATED GAME MODEL
# =========================================================

class RepeatedGameModel:

    def __init__(
        self,
        T: int = 24,
        discount_factor: float = 0.92,
        late_threshold_days: float = 7.0,
        cooperation_surplus: float = 100.0,
        defection_gain: float = 20.0,
        grim_trigger_threshold: int = 3,
        reputation_decay: float = 0.90,
        seed: int = 42,
    ):

        self.T = T

        self.discount_factor = (
            discount_factor
        )

        self.late_threshold_days = (
            late_threshold_days
        )

        self.cooperation_surplus = (
            cooperation_surplus
        )

        self.defection_gain = (
            defection_gain
        )

        self.grim_trigger_threshold = (
            grim_trigger_threshold
        )

        self.reputation_decay = (
            reputation_decay
        )

        self.seed = seed

        self.rng = np.random.default_rng(
            seed
        )

        self.delta_required = (
            self.defection_gain
            /
            (
                self.defection_gain
                +
                self.cooperation_surplus
            )
        )

    # =====================================================
    # HISTORY GENERATION
    # =====================================================

    def _build_history(
        self,
        row: pd.Series,
    ) -> SupplierHistory:

        risk_class = str(
            row.get(
                "supplier_risk_class",
                "Medium"
            )
        )

        risk_map = {
            "Low": 0.05,
            "Medium": 0.12,
            "High": 0.22,
            "Critical": 0.35,
        }

        defect_prob = risk_map.get(
            risk_class,
            0.12,
        )

        return SupplierHistory(
            item_id=str(
                row.get(
                    "item_id",
                    "UNKNOWN"
                )
            ),
            periods=self.T,
            on_time_rate=float(
                1.0 - defect_prob
            ),
            defection_prob=defect_prob,
            late_threshold_days=self.late_threshold_days,
            geo_risk_score=float(
                row.get(
                    "geo_risk_score",
                    0.0
                )
            ),
            supplier_risk_class=risk_class,
        )

    # =====================================================
    # REPUTATION ENGINE
    # =====================================================

    def _compute_reputation(
        self,
        history: SupplierHistory,
    ) -> ReputationResult:

        rep = 1.0

        n_defections = 0

        trigger = False

        for defected in history.defections:

            if defected:

                n_defections += 1

                rep = (
                    self.reputation_decay
                    *
                    rep
                )

                adaptive_threshold = (
                    self.grim_trigger_threshold
                    +
                    int(
                        history.geo_risk_score
                        * 3
                    )
                )

                if (
                    n_defections >=
                    adaptive_threshold
                ):

                    trigger = True

            else:

                rep = (
                    self.reputation_decay
                    *
                    rep
                    +
                    (
                        1
                        -
                        self.reputation_decay
                    )
                )

        # -------------------------------------------------
        # CONTEXT-AWARE TRIGGER PENALTY
        # -------------------------------------------------

        if trigger:

            punishment_floor = (
                0.30
                +
                (
                    0.15
                    *
                    history.geo_risk_score
                )
            )

            rep = min(
                rep,
                punishment_floor
            )

        rep = float(
            np.clip(rep, 0.0, 1.0)
        )

        delta_ok = (
            self.discount_factor
            >
            self.delta_required
        )

        # -------------------------------------------------
        # ENTERPRISE ESCALATION LADDER
        # -------------------------------------------------

        supplier_risk = (
            history.supplier_risk_class
        )

        if (
            trigger
            and
            (
                rep <= 0.10
                or
                (
                    rep <= 0.25
                    and supplier_risk in (
                        "High",
                        "Critical",
                    )
                )
            )
        ):

            action = "Mandatory Switch"

        elif (
            supplier_risk == "Critical"
            or rep <= 0.40
        ):

            action = "Renegotiate"

        elif rep <= 0.60:

            action = "Warning"

        elif rep <= 0.80:

            action = "Monitor"

        else:

            action = "Continue"

        return ReputationResult(

            item_id=history.item_id,

            reputation_score=round(
                rep,
                4,
            ),

            grim_trigger_fired=trigger,

            n_defections=n_defections,

            n_periods=len(
                history.defections
            ),

            delta_satisfied=delta_ok,

            delta_value=self.delta_required,

            cooperation_surplus=(
                self.cooperation_surplus
            ),

            defection_gain=(
                self.defection_gain
            ),

            recommended_action=action,
        )

    # =====================================================
    # PUBLIC API
    # =====================================================

    def score(
        self,
        df: pd.DataFrame,
    ):

        results = []

        for _, row in df.iterrows():

            history = self._build_history(
                row
            )

            history.simulate(
                self.rng
            )

            result = (
                self._compute_reputation(
                    history
                )
            )

            results.append({
                "item_id":
                    result.item_id,

                "reputation_score":
                    result.reputation_score,

                "grim_trigger_fired":
                    result.grim_trigger_fired,

                "n_defections":
                    result.n_defections,

                "delta_satisfied":
                    result.delta_satisfied,

                "recommended_action":
                    result.recommended_action,

                "supplier_risk_class":
                    row.get(
                        "supplier_risk_class",
                        "Medium"
                    ),
            })

        out = pd.DataFrame(
            results
        )

        return out, out.copy()

    def folk_theorem_summary(
        self,
    ):

        return {
            "discount_factor":
                self.discount_factor,

            "delta_required":
                self.delta_required,

            "folk_theorem_satisfied":
                (
                    self.discount_factor
                    >
                    self.delta_required
                ),
        }
