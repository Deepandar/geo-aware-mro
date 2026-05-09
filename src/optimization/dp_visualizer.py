from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.optimization.bellman_engine import (
    BellmanEngine,
    DPParams,
)

BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
GRID = "#30363d"

LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD,
    font=dict(color=TEXT),
)


def fig_value_function(
    engine: BellmanEngine,
    params: DPParams,
):

    result = engine.solve(params)

    x_vals = list(
        range(
            params.x_min,
            params.x_max + 1,
        )
    )

    y_vals = [
        result.value_function.get((0, x), np.nan)
        for x in x_vals
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="V0(x)",
        )
    )

    fig.update_layout(
        **LAYOUT,
        title="Bellman Value Function",
        xaxis=dict(
            title="Inventory State",
            gridcolor=GRID,
        ),
        yaxis=dict(
            title="Expected Cost",
            gridcolor=GRID,
        ),
        height=400,
    )

    return fig


def fig_policy_heatmap(
    engine: BellmanEngine,
    params: DPParams,
):

    result = engine.solve(params)

    T = params.T

    x_range = list(
        range(
            params.x_min,
            params.x_max + 1,
        )
    )

    Z = np.array([
        [
            result.optimal_policy.get((t, x), 0)
            for x in x_range
        ]
        for t in range(T + 1)
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=Z,
            x=x_range,
            y=list(range(T + 1)),
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        **LAYOUT,
        title="Optimal Policy Heatmap",
        xaxis_title="Inventory State",
        yaxis_title="Time Period",
        height=450,
    )

    return fig


def fig_dp_vs_static(df: pd.DataFrame):

    required = {
        "q_star",
        "dp_q_star",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["q_star"],
            y=df["dp_q_star"],
            mode="markers",
            marker=dict(size=7),
        )
    )

    max_val = max(
        df["q_star"].max(),
        df["dp_q_star"].max(),
    )

    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(dash="dash"),
            name="Equal",
        )
    )

    fig.update_layout(
        **LAYOUT,
        title="DP vs Static Policy",
        xaxis_title="Static q*",
        yaxis_title="DP q*",
        height=400,
    )

    return fig
