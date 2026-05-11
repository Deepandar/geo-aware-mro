from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


# =========================================================
# THEME
# =========================================================

BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
GRID = "#30363d"

LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD,
    font=dict(color=TEXT),
)


# =========================================================
# Q* DISTRIBUTION
# =========================================================

def fig_qstar_distribution(
    df: pd.DataFrame,
):

    required = {
        "q_star",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=df["q_star"],
            nbinsx=40,
            name="q*",
        )
    )

    fig.update_layout(
        **LAYOUT,
        title="Bellman q* Distribution",
        xaxis_title="Optimal Order Quantity",
        yaxis_title="Frequency",
        height=450,
    )

    return fig


# =========================================================
# ROP VS GEO RISK
# =========================================================

def fig_rop_vs_geo_risk(
    df: pd.DataFrame,
):

    required = {
        "bellman_rop",
        "geo_risk_score",
        "ci_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(

            x=df["geo_risk_score"],

            y=df["bellman_rop"],

            mode="markers",

            marker=dict(
                size=10,
                color=df["ci_score"],
                colorscale="Viridis",
                showscale=True,
            ),

            text=df.get(
                "item_id",
                None,
            ),

            name="SKU",
        )
    )

    fig.update_layout(
        **LAYOUT,
        title="ROP vs Geo Risk",
        xaxis_title="Geo Risk Score",
        yaxis_title="Bellman ROP",
        height=500,
    )

    return fig


# =========================================================
# FUTURE COST SURFACE
# =========================================================

def fig_future_cost_surface(
    df: pd.DataFrame,
):

    required = {
        "expected_future_cost",
        "geo_risk_score",
        "ci_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(

            x=df["geo_risk_score"],

            y=df["ci_score"],

            z=df["expected_future_cost"],

            mode="markers",

            marker=dict(
                size=4,
                color=df[
                    "expected_future_cost"
                ],
                colorscale="Inferno",
            ),
        )
    )

    fig.update_layout(
        **LAYOUT,
        title="Future Cost Surface",
        scene=dict(
            xaxis_title="Geo Risk",
            yaxis_title="CI Score",
            zaxis_title="Future Cost",
        ),
        height=600,
    )

    return fig
