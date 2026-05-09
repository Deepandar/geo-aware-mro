from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# OPTIONAL MLFLOW IMPORT
# ---------------------------------------------------------

try:

    import mlflow

    MLFLOW_AVAILABLE = True

except Exception:

    MLFLOW_AVAILABLE = False


# ---------------------------------------------------------
# DELIVERY HISTORY
# ---------------------------------------------------------

@dataclass
class SupplierHistory:

    item_id: str

    periods: int

    on_time_rate: float

    defection_prob: float

    late_threshold_days: float

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

            if rng.random() < self.defection_prob:

                delay = rng.gamma(
                    shape=2.0,
                    scale=self.late_threshold_days,
                )

            else:

                delay = rng.exponential(
                    scale=1.5
                )

            self.deliveries.append(delay)

            self.defections.append(
                delay >
                self.late_threshold_days
            )


# ---------------------------------------------------------
# RESULT OBJECT
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# MAIN MODEL
# ---------------------------------------------------------

class RepeatedGameModel:

    def __init__(
        self,
        T: int = 24,
        discount_factor: float = 0.92,
        late_threshold_days: float = 7.0,
        reputation_decay: float = 0.85,
        cooperation_surplus: float = 100.0,
        defection_gain: float = 20.0,
        grim_trigger_threshold: int = 1,
    ):

        self.T = T

        self.discount_factor = (
            discount_factor
        )

        self.late_threshold_days = (
            late_threshold_days
        )

        self.reputation_decay = (
            reputation_decay
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

        self.delta_required = (
            self.defection_gain /
            (
                self.defection_gain +
                self.cooperation_surplus
            )
        )

        logger.info(
            "RepeatedGameModel initialised | "
            "δ=%.3f | δ_required=%.3f",
            self.discount_factor,
            self.delta_required,
        )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    def _validate(
        self,
        df: pd.DataFrame,
    ):

        if "item_id" not in df.columns:

            raise ValueError(
                "RepeatedGameModel: "
                "'item_id' required"
            )

    # -----------------------------------------------------
    # REPUTATION UPDATE
    # -----------------------------------------------------

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
                    self.reputation_decay *
                    rep
                )

                if (
                    n_defections >=
                    self.grim_trigger_threshold
                ):

                    trigger = True

            else:

                rep = (
                    self.reputation_decay *
                    rep
                    +
                    (
                        1 -
                        self.reputation_decay
                    )
                )

        if trigger:

            rep = min(rep, 0.20)

        rep = float(
            np.clip(rep, 0.0, 1.0)
        )

        delta_ok = (
            self.discount_factor >
            self.delta_required
        )

        # -------------------------------------------------
        # ACTION POLICY
        # -------------------------------------------------

        if trigger:

            action = "Mandatory Switch"

        elif rep < 0.40:

            action = "Warning"

        elif rep < 0.65:

            action = "Monitor"

        elif not delta_ok:

            action = "Renegotiate"

        else:

            action = "Continue"

        return ReputationResult(

            item_id=history.item_id,

            reputation_score=round(rep, 4),

            grim_trigger_fired=trigger,

            n_defections=n_defections,

            n_periods=len(history.defections),

            delta_satisfied=delta_ok,

            delta_value=self.delta_required,

            cooperation_surplus=
            self.cooperation_surplus,

            defection_gain=
            self.defection_gain,

            recommended_action=action,
        )

    # -----------------------------------------------------
    # MAIN SCORE
    # -----------------------------------------------------

    def score(
        self,
        df: pd.DataFrame,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        self._validate(df)

        out = df.copy()

        rng = np.random.default_rng(seed)

        results = []

        for _, row in out.iterrows():

            geo = float(
                row.get(
                    "geo_risk_score",
                    0.30,
                )
            )

            risk = str(
                row.get(
                    "supplier_risk_class",
                    "Medium",
                )
            )

            defect_base = {

                "Low": 0.05,

                "Medium": 0.15,

                "High": 0.30,

                "Critical": 0.50,

            }.get(risk, 0.15)

            defect_prob = float(
                np.clip(
                    0.6 * defect_base
                    +
                    0.4 * geo * 0.5,
                    0.01,
                    0.80,
                )
            )

            history = SupplierHistory(

                item_id=str(
                    row["item_id"]
                ),

                periods=self.T,

                on_time_rate=
                1.0 - defect_prob,

                defection_prob=
                defect_prob,

                late_threshold_days=
                self.late_threshold_days,
            )

            history.simulate(rng)

            result = (
                self._compute_reputation(
                    history
                )
            )

            results.append(result)

        # -------------------------------------------------
        # OUTPUT COLUMNS
        # -------------------------------------------------

        out["reputation_score"] = [
            r.reputation_score
            for r in results
        ]

        out["grim_trigger_fired"] = [
            r.grim_trigger_fired
            for r in results
        ]

        out["n_defections"] = [
            r.n_defections
            for r in results
        ]

        out["delta_satisfied"] = [
            r.delta_satisfied
            for r in results
        ]

        out["recommended_action"] = [
            r.recommended_action
            for r in results
        ]

        rep_matrix = pd.DataFrame([{

            "item_id":
            r.item_id,

            "reputation_score":
            r.reputation_score,

            "grim_trigger_fired":
            r.grim_trigger_fired,

            "n_defections":
            r.n_defections,

            "n_periods":
            r.n_periods,

            "delta_satisfied":
            r.delta_satisfied,

            "delta_required":
            r.delta_value,

            "recommended_action":
            r.recommended_action,

        } for r in results])

        logger.info(
            "Repeated game complete | "
            "mean_rep=%.3f | "
            "grim_trigger=%d",
            out[
                "reputation_score"
            ].mean(),
            int(
                out[
                    "grim_trigger_fired"
                ].sum()
            ),
        )

        return out, rep_matrix

    # -----------------------------------------------------
    # FOLK THEOREM
    # -----------------------------------------------------

    def folk_theorem_summary(
        self,
    ) -> dict:

        ok = (
            self.discount_factor >
            self.delta_required
        )

        return {

            "discount_factor":
            self.discount_factor,

            "defection_gain":
            self.defection_gain,

            "cooperation_surplus":
            self.cooperation_surplus,

            "delta_required":
            round(
                self.delta_required,
                4,
            ),

            "folk_theorem_satisfied":
            ok,
        }

    # -----------------------------------------------------
    # OPTIONAL MLFLOW
    # -----------------------------------------------------

    def log_to_mlflow(
        self,
        df: pd.DataFrame,
        run_name: str =
        "repeated_game_v1.2",
    ) -> None:

        if not MLFLOW_AVAILABLE:

            logger.warning(
                "MLflow unavailable"
            )

            return

        with mlflow.start_run(
            run_name=run_name,
            nested=True,
        ):

            mlflow.log_param(
                "T",
                self.T,
            )

            mlflow.log_param(
                "discount_factor",
                self.discount_factor,
            )

            mlflow.log_metric(
                "mean_reputation",
                float(
                    df[
                        "reputation_score"
                    ].mean()
                ),
            )
