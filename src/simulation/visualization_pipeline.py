from __future__ import annotations

import logging

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# Visualization Pipeline
# =========================================================


class VisualizationPipeline:

    def __init__(
        self,
        output_dir: str = "reports/figures",
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # -----------------------------------------------------
    # Edge of Chaos
    # -----------------------------------------------------

    def edge_of_chaos_plot(
        self,
        df: pd.DataFrame,
        save_name: str = ("edge_of_chaos.png"),
    ) -> str:

        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            df["geo_risk_score"],
            df["fill_rate"],
            s=df["ci_score"] * 400,
            c=df["ci_score"],
            alpha=0.7,
        )

        ax.set_title(
            ("Geo-Aware Digital Twin\n" "Risk vs Operational Availability"),
            fontsize=16,
        )

        ax.set_xlabel(
            "Geo Risk Score",
            fontsize=12,
        )

        ax.set_ylabel(
            "Realized Fill Rate",
            fontsize=12,
        )

        ax.grid(True)

        cbar = plt.colorbar(scatter)

        cbar.set_label("Criticality Index")

        # -------------------------------------------------
        # Quadrant annotations
        # -------------------------------------------------

        ax.axvline(
            0.5,
            linestyle="--",
            alpha=0.5,
        )

        ax.axhline(
            0.5,
            linestyle="--",
            alpha=0.5,
        )

        ax.text(
            0.1,
            0.9,
            "SAFE HAVEN",
            fontsize=12,
            weight="bold",
        )

        ax.text(
            0.75,
            0.2,
            "DEATH ZONE",
            fontsize=12,
            weight="bold",
        )

        output_path = self.output_dir / save_name

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            ("Edge of Chaos plot saved | %s"),
            output_path,
        )

        return str(output_path)

    # -----------------------------------------------------
    # Country failure chart
    # -----------------------------------------------------

    def country_failure_plot(
        self,
        failed_df: pd.DataFrame,
        save_name: str = ("country_failure.png"),
    ) -> str:

        counts = (
            failed_df["supply_origin_country"]
            .value_counts()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        counts.plot(
            kind="bar",
            ax=ax,
        )

        ax.set_title(
            ("Failed SKU Distribution\n" "by Supply Origin"),
            fontsize=15,
        )

        ax.set_xlabel("Country")

        ax.set_ylabel("Failed SKU Count")

        ax.grid(
            axis="y",
            alpha=0.3,
        )

        output_path = self.output_dir / save_name

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        logger.info(
            ("Country failure plot saved | %s"),
            output_path,
        )

        return str(output_path)
