from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash
from dash import Input
from dash import Output
from dash import dcc
from dash import html


DATA_PATH = Path(
    "data/processed/sku_master_v1.1.parquet"
)

df = pd.read_parquet(DATA_PATH)

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            "Geo-Aware MRO Intelligence Dashboard"
        ),

        html.Div(
            [
                html.Label(
                    "ABC Class"
                ),

                dcc.Dropdown(
                    options=[
                        {
                            "label": value,
                            "value": value,
                        }
                        for value in sorted(
                            df["abc_class"].unique()
                        )
                    ],
                    multi=True,
                    id="abc-filter",
                ),

                html.Label(
                    "VED Class"
                ),

                dcc.Dropdown(
                    options=[
                        {
                            "label": value,
                            "value": value,
                        }
                        for value in sorted(
                            df["ved_class"].unique()
                        )
                    ],
                    multi=True,
                    id="ved-filter",
                ),

                html.Label(
                    "FNS Class"
                ),

                dcc.Dropdown(
                    options=[
                        {
                            "label": value,
                            "value": value,
                        }
                        for value in sorted(
                            df["fns_class"].unique()
                        )
                    ],
                    multi=True,
                    id="fns-filter",
                ),
            ],
            style={
                "width": "25%",
                "padding": "20px",
            },
        ),

        dcc.Graph(
            id="ci-histogram"
        ),

        dcc.Graph(
            id="ltr-scatter"
        ),

        dcc.Graph(
            id="pareto-chart"
        ),

        dcc.Graph(
            id="ci-heatmap"
        ),

        dcc.Graph(
            id="tsl-box"
        ),
    ]
)


@app.callback(
    [
        Output(
            "ci-histogram",
            "figure",
        ),

        Output(
            "ltr-scatter",
            "figure",
        ),

        Output(
            "pareto-chart",
            "figure",
        ),

        Output(
            "ci-heatmap",
            "figure",
        ),

        Output(
            "tsl-box",
            "figure",
        ),
    ],

    [
        Input(
            "abc-filter",
            "value",
        ),

        Input(
            "ved-filter",
            "value",
        ),

        Input(
            "fns-filter",
            "value",
        ),
    ],
)
def update_dashboard(
    abc_values,
    ved_values,
    fns_values,
):

    filtered_df = df.copy()

    if abc_values:
        filtered_df = filtered_df[
            filtered_df["abc_class"].isin(
                abc_values
            )
        ]

    if ved_values:
        filtered_df = filtered_df[
            filtered_df["ved_class"].isin(
                ved_values
            )
        ]

    if fns_values:
        filtered_df = filtered_df[
            filtered_df["fns_class"].isin(
                fns_values
            )
        ]

    ci_hist = px.histogram(
        filtered_df,
        x="ci_score",
        nbins=20,
        title="Criticality Index Distribution",
    )

    scatter = px.scatter(
        filtered_df,
        x="ltr_score",
        y="ci_score",
        color="abc_class",
        size="q_star",
        hover_data=["item_id"],
        title="LTR vs Criticality",
    )

    pareto_df = (
        filtered_df
        .sort_values(
            by="annual_consumption_value",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    pareto_df["cum_pct"] = (
        pareto_df[
            "annual_consumption_value"
        ].cumsum()
        /
        pareto_df[
            "annual_consumption_value"
        ].sum()
    ) * 100

    pareto_fig = go.Figure()

    pareto_fig.add_trace(
        go.Bar(
            x=pareto_df.index,
            y=pareto_df[
                "annual_consumption_value"
            ],
            name="ACV",
        )
    )

    pareto_fig.add_trace(
        go.Scatter(
            x=pareto_df.index,
            y=pareto_df["cum_pct"],
            mode="lines",
            name="Cumulative %",
            yaxis="y2",
        )
    )

    pareto_fig.update_layout(
        title="Pareto Concentration",

        yaxis2=dict(
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
    )

    heatmap_df = (
        filtered_df
        .groupby(
            [
                "abc_class",
                "ved_class",
            ]
        )["ci_score"]
        .mean()
        .reset_index()
    )

    heatmap = px.density_heatmap(
        heatmap_df,
        x="abc_class",
        y="ved_class",
        z="ci_score",
        title="CI Heatmap",
    )

    tsl_box = px.box(
        filtered_df,
        x="abc_class",
        y="tsl",
        title="Service Level Segmentation",
    )

    return (
        ci_hist,
        scatter,
        pareto_fig,
        heatmap,
        tsl_box,
    )


if __name__ == "__main__":
    app.run(debug=True)