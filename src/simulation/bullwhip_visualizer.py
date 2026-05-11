
import pandas as pd
import plotly.graph_objects as go

# -----------------------------------------------------
# Visualization theme
# -----------------------------------------------------

TEXT = "#EAEAEA"

LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "#111111",
    "plot_bgcolor": "#111111",
}

# -----------------------------------------------------
# Compatibility placeholder
# -----------------------------------------------------

class BullwhipSummary:
    pass

# -----------------------------------------------------
# Fallback visualization stubs
# -----------------------------------------------------

def fig_bwr_by_echelon(summary):

    return go.Figure()

def fig_policy_comparison(comparison_df):

    return go.Figure()

def fig_bwr_distribution_by_tier(bw_df):

    return go.Figure()

def fig_order_demand_timeseries(result):

    return go.Figure()



# ──────────────────────────────────────────────────────────
# Heatmap: amplification matrix
# ──────────────────────────────────────────────────────────

def fig_amplification_heatmap(
    bw_df: pd.DataFrame
) -> go.Figure:

    echelon_cols = [
        c for c in bw_df.columns
        if c.startswith("bwr_e")
    ]

    matrix = bw_df[echelon_cols].values

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=echelon_cols,
            y=bw_df["item_id"],
            colorscale="RdYlGn_r",
            colorbar=dict(
                title="BWR"
            ),
        )
    )

    fig.update_layout(
        **LAYOUT,
        title=dict(
            text="Bullwhip Amplification Heatmap",
            font=dict(size=12, color=TEXT),
            x=0.01,
        ),
        height=max(350, len(bw_df) * 12),
    )

    return fig


# ──────────────────────────────────────────────────────────
# EXPORT ALL FIGURES
# ──────────────────────────────────────────────────────────

def export_all_figures(
    summary: BullwhipSummary,
    comparison_df: pd.DataFrame,
    output_dir: str = "reports/figures/bullwhip",
) -> None:

    import os

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    bw_df = pd.DataFrame([
        {
            "item_id": r.item_id,
            "ci_tier": r.ci_tier,
            "total_amplification":
                r.total_amplification,
            **{
                f"bwr_e{i}": b
                for i, b in enumerate(
                    r.amplification_ratios
                )
            }
        }
        for r in summary.sku_results
    ])

    figs = {

        "01_bwr_by_echelon":
            fig_bwr_by_echelon(summary),

        "02_policy_comparison":
            fig_policy_comparison(comparison_df),

        "03_bwr_distribution":
            fig_bwr_distribution_by_tier(bw_df),

        "04_amplification_heatmap":
            fig_amplification_heatmap(bw_df),
    }

    # single sample SKU
    if summary.sku_results:

        figs["05_sample_timeseries"] = (
            fig_order_demand_timeseries(
                summary.sku_results[0]
            )
        )

    for name, fig in figs.items():

        html_path = (
            f"{output_dir}/{name}.html"
        )

        fig.write_html(html_path)

        print(
            f"Saved: {html_path}"
        )

