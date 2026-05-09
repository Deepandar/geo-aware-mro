from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import nashpy as nash
    NASHPY_AVAILABLE = True
except ImportError:
    NASHPY_AVAILABLE = False
    logger.warning(
        "nashpy unavailable — fallback mode active"
    )


# =========================================================
# PAYOFF MATRICES
# =========================================================

PAYOFF_BUYER_2x2 = np.array([
    [3.0, 1.0],
    [4.0, 2.0],
])

PAYOFF_SUPPLIER_2x2 = np.array([
    [2.0, 4.0],
    [1.0, 3.0],
]).T


# =========================================================
# RESULT DATACLASSES
# =========================================================

@dataclass
class NashModelResult:

    n_skus: int
    n_dual_source: int
    n_dual_source_mandatory: int
    n_single_source: int
    mean_strategic_risk: float
    pure_ne_count: int
    mixed_ne_count: int


# =========================================================
# MAIN MODEL
# =========================================================

class NashEquilibriumModel:

    def __init__(
        self,
        dual_source_threshold: float = 0.35,
        mandatory_threshold: float = 0.65,
        buffer_stock_multiplier: float = 0.25,
    ):

        self.dual_source_threshold = dual_source_threshold
        self.mandatory_threshold = mandatory_threshold
        self.buffer_stock_multiplier = buffer_stock_multiplier

        logger.info(
            (
                "NashEquilibriumModel initialised | "
                "dual=%.2f | mandatory=%.2f"
            ),
            dual_source_threshold,
            mandatory_threshold,
        )

    # =====================================================
    # SRS
    # =====================================================

    def _compute_srs(
        self,
        geo_risk: float,
        hhi: float,
    ) -> float:

        return float(
            np.clip(
                geo_risk * hhi,
                0.0,
                1.0,
            )
        )

    # =====================================================
    # NASH SOLVER
    # =====================================================

    def _compute_nash(
        self,
        srs: float,
    ):

        risk_factor = 1.0 + (srs * 2.0)

        buyer_payoff = PAYOFF_BUYER_2x2.copy()
        buyer_payoff[1, :] *= risk_factor

        supplier_payoff = PAYOFF_SUPPLIER_2x2.copy()

        if NASHPY_AVAILABLE:

            try:

                game = nash.Game(
                    buyer_payoff,
                    supplier_payoff,
                )

                equilibria = list(
                    game.support_enumeration()
                )

                if equilibria:

                    buyer_mix, supp_mix = equilibria[0]

                    ne_type = (
                        "Pure"
                        if (
                            np.max(buyer_mix) > 0.99
                            or np.max(supp_mix) > 0.99
                        )
                        else "Mixed"
                    )

                    return (
                        ne_type,
                        buyer_mix,
                        supp_mix,
                    )

            except Exception as e:

                logger.debug(
                    "nashpy failed: %s",
                    e,
                )

        # fallback

        buyer_mix = np.array([
            max(0.0, 1.0 - srs),
            min(1.0, srs),
        ])

        supp_mix = np.array([
            min(1.0, 0.4 + srs),
            max(0.0, 0.6 - srs),
        ])

        return (
            "Mixed",
            buyer_mix,
            supp_mix,
        )

    # =====================================================
    # MAIN SCORING
    # =====================================================

    def score(
        self,
        df: pd.DataFrame,
    ):

        self._validate(df)

        out = df.copy()

        srs_list = []
        strategy_list = []
        ne_type_list = []
        buffer_signal_list = []
        ne_price_list = []

        for _, row in out.iterrows():

            geo = float(
                row.get(
                    "geo_risk_score",
                    0.0,
                )
            )

            hhi = float(
                row.get(
                    "hhi_score",
                    0.3,
                )
            )

            ved = str(
                row.get(
                    "ved_class",
                    "E",
                )
            )

            is_vital = (ved == "V")

            srs = self._compute_srs(
                geo,
                hhi,
            )

            srs_list.append(srs)

            # sourcing strategy

            if (
                srs >= self.mandatory_threshold
                and is_vital
            ):

                strategy = (
                    "Dual-Source (Mandatory)"
                )

            elif srs >= self.dual_source_threshold:

                strategy = "Dual-Source"

            else:

                strategy = "Single-Source"

            strategy_list.append(strategy)

            # nash equilibrium

            ne_type, buyer_mix, supp_mix = (
                self._compute_nash(
                    srs
                )
            )

            ne_type_list.append(ne_type)

            buffer_signal = float(
                supp_mix[1]
            ) * self.buffer_stock_multiplier

            buffer_signal_list.append(
                round(
                    buffer_signal,
                    3,
                )
            )

            ne_price = (
                "H"
                if supp_mix[0] >= 0.5
                else "L"
            )

            ne_price_list.append(
                ne_price
            )

        # =================================================
        # OUTPUT COLUMNS
        # =================================================

        out[
            "strategic_risk_score"
        ] = srs_list

        out[
            "sourcing_strategy"
        ] = strategy_list

        out[
            "ne_type"
        ] = ne_type_list

        out[
            "buffer_stock_signal"
        ] = buffer_signal_list

        out[
            "ne_price_equilibrium"
        ] = ne_price_list

        out[
            "dual_source_justified"
        ] = out[
            "sourcing_strategy"
        ].str.startswith(
            "Dual"
        )

        # backward compatibility

        out[
            "supplier_strategy"
        ] = out[
            "sourcing_strategy"
        ]

        # =================================================
        # AGGREGATE RESULTS
        # =================================================

        dist = out[
            "sourcing_strategy"
        ].value_counts()

        result = NashModelResult(

            n_skus=len(out),

            n_dual_source=int(
                dist.get(
                    "Dual-Source",
                    0,
                )
            ),

            n_dual_source_mandatory=int(
                dist.get(
                    "Dual-Source (Mandatory)",
                    0,
                )
            ),

            n_single_source=int(
                dist.get(
                    "Single-Source",
                    0,
                )
            ),

            mean_strategic_risk=float(
                out[
                    "strategic_risk_score"
                ].mean()
            ),

            pure_ne_count=int(
                (
                    out["ne_type"]
                    == "Pure"
                ).sum()
            ),

            mixed_ne_count=int(
                (
                    out["ne_type"]
                    == "Mixed"
                ).sum()
            ),
        )

        logger.info(
            (
                "Nash scoring complete | "
                "Single=%d | "
                "Dual=%d | "
                "Mandatory=%d"
            ),
            result.n_single_source,
            result.n_dual_source,
            result.n_dual_source_mandatory,
        )

        return out, result

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate(
        self,
        df: pd.DataFrame,
    ):

        required = {
            "item_id",
            "ved_class",
        }

        missing = required - set(
            df.columns
        )

        if missing:

            raise ValueError(
                (
                    "NashEquilibriumModel "
                    f"missing columns: {missing}"
                )
            )


# =========================================================
# BACKWARD COMPATIBILITY WRAPPER
# =========================================================

class NashEquilibriumEngine:

    def __init__(self):

        self.model = (
            NashEquilibriumModel()
        )

    def compute(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        out, _ = self.model.score(df)

        return out
