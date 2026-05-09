# src/simulation/temporal_simulator.py

from __future__ import annotations

import logging

import pandas as pd

from src.risk.scenario_manager import (
    ScenarioManager,
)

from src.risk.resilience_engine import (
    ResilienceEngine,
)

from src.classifiers.ltr_scorer import (
    LTRScorer,
)

from src.classifiers.criticality_index import (
    CriticalityIndexer,
)


logger = logging.getLogger(__name__)


class TemporalSimulator:

    """
    Stateful temporal disruption simulator.

    Simulates:
        - rolling disruptions
        - resilience recovery
        - persistent operational risk
        - evolving criticality

    This is the pre-SimPy DES orchestration layer.
    """

    def __init__(
        self,
        simulation_horizon: int = 12,
    ):

        self.simulation_horizon = (
            simulation_horizon
        )

        self.scenario_mgr = (
            ScenarioManager()
        )

        self.resilience = (
            ResilienceEngine()
        )

        self.ltr = LTRScorer()

        self.ci = (
            CriticalityIndexer()
        )

        logger.info(
            (
                "TemporalSimulator initialised | "
                "horizon=%d"
            ),
            self.simulation_horizon,
        )

    # -------------------------------------------------------------
    # Run simulation
    # -------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        current_state = df.copy()

        history = []

        # ---------------------------------------------------------
        # Rolling simulation loop
        # ---------------------------------------------------------

        for t in range(
            1,
            self.simulation_horizon + 1,
        ):

            logger.info(
                (
                    "Simulation step %d/%d"
                ),
                t,
                self.simulation_horizon,
            )

            # -----------------------------------------------------
            # Inject new disruptions
            # -----------------------------------------------------

            current_state = (
                self.scenario_mgr.inject(
                    current_state,
                    sim_time=t,
                )
            )

            # -----------------------------------------------------
            # Apply resilience decay
            # -----------------------------------------------------

            current_state = (
                self.resilience.apply_decay(
                    current_state,
                    sim_time=t,
                )
            )

            # -----------------------------------------------------
            # Recompute LTR
            # -----------------------------------------------------

            current_state = (
                self.ltr.compute(
                    current_state
                )
            )

            # -----------------------------------------------------
            # Recompute CI
            # -----------------------------------------------------

            current_state = (
                self.ci.compute(
                    current_state
                )
            )

            # -----------------------------------------------------
            # Persist temporal metadata
            # -----------------------------------------------------

            current_state[
                "simulation_step"
            ] = t

            current_state[
                "active_disruptions"
            ] = int(
                current_state[
                    "scenario_active"
                ].sum()
            )

            current_state[
                "mean_ltr"
            ] = float(
                current_state[
                    "ltr_score"
                ].mean()
            )

            current_state[
                "mean_ci"
            ] = float(
                current_state[
                    "ci_score"
                ].mean()
            )

            history.append(
                current_state.copy()
            )

            logger.info(
                (
                    "Step complete | "
                    "active=%d | "
                    "mean_ltr=%.3f | "
                    "mean_ci=%.3f"
                ),
                int(
                    current_state[
                        "scenario_active"
                    ].sum()
                ),
                float(
                    current_state[
                        "ltr_score"
                    ].mean()
                ),
                float(
                    current_state[
                        "ci_score"
                    ].mean()
                ),
            )

        # ---------------------------------------------------------
        # Combine temporal history
        # ---------------------------------------------------------

        result = pd.concat(
            history,
            ignore_index=True,
        )

        logger.info(
            (
                "Temporal simulation complete | "
                "rows=%d"
            ),
            len(result),
        )

        return result