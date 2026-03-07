"""
Plotly chart helpers for the thesis dashboard.
"""

from __future__ import annotations

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    go = None
    px = None


def bar_reward_by_system(df: pd.DataFrame, env: str, title: str = "Mean reward by system") -> "go.Figure | None":
    """Bar chart: system vs mean_reward for one environment. df has System, env column with reward."""
    if go is None:
        return None
    col = env
    if col not in df.columns:
        return None
    df_sorted = df.sort_values(col, ascending=True)
    fig = go.Figure(
        data=[go.Bar(y=df_sorted["System"], x=df_sorted[col], orientation="h", marker_color="#16A34A")],
        layout=go.Layout(
            title=title,
            xaxis_title="Mean reward",
            yaxis_title="",
            height=400,
            margin=dict(l=120),
            showlegend=False,
        ),
    )
    return fig


def scatter_precision_reward(
    df: pd.DataFrame,
    precision_col: str = "precision",
    reward_col: str = "reward",
    system_col: str = "System",
    title: str = "Precision vs reward",
) -> "go.Figure | None":
    """Scatter: precision (x) vs reward (y), point label = system."""
    if px is None:
        return None
    fig = px.scatter(
        df,
        x=precision_col,
        y=reward_col,
        hover_name=system_col,
        title=title,
        labels={precision_col: "Retrieval precision", reward_col: "Mean reward"},
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(height=400)
    return fig


def heatmap_sensitivity(
    dim1_values: list[float],
    dim2_values: list[float],
    reward_grid: list[list[float]],
    learned_dim1: float | None = None,
    learned_dim2: float | None = None,
    dim1_name: str = "theta_novel",
    dim2_name: str = "w_recency",
) -> "go.Figure | None":
    """2D heatmap of reward over dim1 x dim2; optional learned point marker."""
    if go is None:
        return None
    import numpy as np
    z = np.array(reward_grid)
    if z.shape != (len(dim2_values), len(dim1_values)):
        z = np.array(reward_grid).T  # try transpose
    fig = go.Figure(
        data=go.Heatmap(
            x=dim1_values,
            y=dim2_values,
            z=z,
            colorscale="RdYlGn",
            zmin=0,
            zmax=z.max() if z.size else 1,
            colorbar=dict(title="Mean reward"),
        ),
        layout=go.Layout(
            title=f"Reward landscape — {dim1_name} × {dim2_name}",
            xaxis_title=dim1_name,
            yaxis_title=dim2_name,
            height=450,
        ),
    )
    if learned_dim1 is not None and learned_dim2 is not None:
        fig.add_trace(
            go.Scatter(
                x=[learned_dim1],
                y=[learned_dim2],
                mode="markers",
                marker=dict(symbol="star", size=16, color="black", line=dict(width=2, color="white")),
                name="Learned θ",
            )
        )
    return fig


def line_training_curve(
    generations: list[int],
    best_fitness: list[float],
    sigma: list[float] | None = None,
    v4_reward: float | None = None,
    title: str = "CMA-ES training curve",
) -> "go.Figure | None":
    """Line chart: generation vs best_fitness; optional second y-axis for sigma, optional hline for V4."""
    if go is None:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=generations, y=best_fitness, mode="lines+markers", name="Best fitness", line=dict(color="#7C3AED"))
    )
    if v4_reward is not None:
        fig.add_hline(y=v4_reward, line_dash="dash", line_color="#16A34A", annotation_text="V4 scalar")
    if sigma is not None:
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=sigma,
                mode="lines",
                name="Sigma",
                line=dict(color="#6B7280"),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="Sigma"),
            yaxis_title="Best fitness (mean reward)",
        )
    fig.update_layout(title=title, xaxis_title="Generation", height=400, showlegend=True)
    return fig


def bar_degradation(
    configs: list[str],
    degradation_pct: list[float],
    title: str = "Reward degradation when component removed",
) -> "go.Figure | None":
    """Horizontal bar chart: config vs degradation %."""
    if go is None:
        return None
    colors = ["#DC2626" if d > 50 else "#EA580C" if d > 10 else "#16A34A" for d in degradation_pct]
    fig = go.Figure(
        data=[go.Bar(x=degradation_pct, y=configs, orientation="h", marker_color=colors)],
        layout=go.Layout(
            title=title,
            xaxis_title="Degradation (%)",
            yaxis_title="",
            height=max(300, len(configs) * 28),
            margin=dict(l=150),
            showlegend=False,
        ),
    )
    return fig
