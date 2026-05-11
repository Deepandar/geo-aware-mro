"""
Supplier Reputation Dashboard — W22 Deliverable
=================================================
Produces the W22 reputation matrix and analysis figures.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

BG, CARD, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT, MUTED = "#e6edf3", "#8b949e"

BLUE = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#e3b341"
RED = "#f85149"
PURPLE = "#8957e5"

ACTION_COLORS = {
    "Continue": GREEN,
    "Monitor": BLUE,
    "Warning": ORANGE,
    "Renegotiate": PURPLE,
    "Mandatory Switch": RED,
}

LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD,
    font=dict(
        family="Arial",
        color=TEXT,
        size=10,
    ),
    margin=dict(
        l=55,
        r=20,
        t=50,
        b=45,
    ),
)


# =========================================================
# REPUTATION MATRIX
# =========================================================


def fig_reputation_matrix(
    df: pd.DataFrame,
    n_display: int = 100,
) -> go.Figure:

    sample = df.nlargest(n_display, "ci_score") if len(df) > n_display else df.copy()

    sample = sample.sort_values(
        "ci_score",
        ascending=False,
    ).reset_index(drop=True)

    z = sample["reputation_score"].values.reshape(-1, 1)

    labels = [
        f"★ {row.item_id}" if row.grim_trigger_fired else str(row.item_id)
        for _, row in sample.iterrows()
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Reputation"],
            y=labels,
            colorscale=[
                [0.0, RED],
                [0.20, ORANGE],
                [0.60, BLUE],
                [1.0, GREEN],
            ],
            zmin=0,
            zmax=1,
            hovertemplate=("SKU: %{y}<br>" "Rep: %{z:.3f}<extra></extra>"),
        )
    )

    fig.update_layout(
        **LAYOUT,
        title=dict(
            text="Supplier Reputation Matrix",
            x=0.01,
        ),
        height=max(
            350,
            n_display * 12,
        ),
    )

    return fig


# =========================================================
# DELTA SENSITIVITY
# =========================================================


def fig_delta_sensitivity(
    gains: list[float],
    surpluses: list[float],
) -> go.Figure:

    G, S = np.meshgrid(
        gains,
        surpluses,
    )

    delta_req = G / (G + S)

    fig = go.Figure(
        go.Surface(
            z=delta_req,
            x=gains,
            y=surpluses,
            colorscale="Viridis",
            opacity=0.90,
        )
    )

    fig.update_layout(
        **LAYOUT,
        title=dict(
            text="Folk Theorem Stability Boundary",
            x=0.01,
        ),
        scene=dict(
            xaxis_title="Defection Gain",
            yaxis_title="Cooperation Surplus",
            zaxis_title="δ Required",
        ),
        height=480,
    )

    return fig


# =========================================================
# GRIM TRIGGER RATE
# =========================================================


def fig_trigger_firing_by_class(df: pd.DataFrame) -> go.Figure:

    classes = [
        "Low",
        "Medium",
        "High",
        "Critical",
    ]

    vals = []

    for cls in classes:

        sub = df[df.supplier_risk_class == cls]

        vals.append(float(sub["grim_trigger_fired"].mean()) if len(sub) > 0 else 0.0)

    fig = go.Figure(
        go.Bar(
            x=classes,
            y=[v * 100 for v in vals],
            marker_color=[
                GREEN,
                BLUE,
                ORANGE,
                RED,
            ],
        )
    )

    fig.update_layout(
        **LAYOUT,
        title=dict(
            text="Grim Trigger Firing Rate",
            x=0.01,
        ),
        yaxis_title="Trigger Rate (%)",
        height=340,
    )

    return fig


# =========================================================
# ACTION DISTRIBUTION
# =========================================================


def fig_action_distribution(df: pd.DataFrame) -> go.Figure:

    vc = df["recommended_action"].value_counts()

    labels = vc.index.tolist()

    colors = [
        ACTION_COLORS.get(
            label,
            BLUE,
        )
        for label in labels
    ]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=vc.values.tolist(),
            hole=0.40,
            marker=dict(
                colors=colors,
            ),
        )
    )

    fig.update_layout(
        **LAYOUT,
        title=dict(
            text="Recommended Actions",
            x=0.01,
        ),
        height=360,
    )

    return fig


# =========================================================
# GEO-RISK VS REPUTATION
# =========================================================


def fig_reputation_vs_geo_risk(df: pd.DataFrame) -> go.Figure:

    fig = go.Figure(
        go.Scatter(
            x=df["geo_risk_score"],
            y=df["reputation_score"],
            mode="markers",
            marker=dict(
                size=5,
                opacity=0.7,
            ),
            customdata=df[["item_id"]].values,
            hovertemplate=(
                "SKU: %{customdata[0]}<br>"
                "geo_risk: %{x:.3f}<br>"
                "rep: %{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_hline(
        y=0.40,
        line_color=RED,
        line_dash="dash",
    )

    fig.update_layout(
        **LAYOUT,
        title=dict(
            text="Reputation vs Geo-Risk",
            x=0.01,
        ),
        xaxis_title="Geo Risk",
        yaxis_title="Reputation",
        height=360,
    )

    return fig


# =========================================================
# FULL REPORT
# =========================================================


def generate_reputation_report(df: pd.DataFrame) -> dict:

    return {
        "reputation_matrix": fig_reputation_matrix(df),
        "delta_sensitivity": fig_delta_sensitivity(
            gains=list(range(5, 101, 5)),
            surpluses=list(range(10, 201, 10)),
        ),
        "trigger_by_class": fig_trigger_firing_by_class(df),
        "action_distribution": fig_action_distribution(df),
        "rep_vs_geo_risk": fig_reputation_vs_geo_risk(df),
    }
