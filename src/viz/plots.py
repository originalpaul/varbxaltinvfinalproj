"""Plotting functions for VARBX analysis."""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from src.viz.styling import apply_style, get_colors, get_figure_size, setup_style


def plot_cumulative_returns(
    returns_dict: dict[str, pd.Series],
    title: str = "Cumulative Returns",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot cumulative returns for multiple series.

    Args:
        returns_dict: Dictionary mapping series names to return Series
        title: Plot title
        figsize: Figure size (width, height). If None, uses config default.

    Returns:
        Matplotlib figure
    """
    setup_style()

    if figsize is None:
        figsize = get_figure_size("default")

    fig, ax = plt.subplots(figsize=figsize)
    colors = get_colors()

    # Color mapping
    color_map = {
        "VARBX": colors["varbx"],
        "S&P 500": colors["sp500"],
        "SP500": colors["sp500"],
        "AGG": colors["agg"],
        "Bloomberg US Aggregate": colors["agg"],
    }

    for name, returns in returns_dict.items():
        # Calculate cumulative returns
        cumulative = (1 + returns).cumprod()

        # Get color
        color = color_map.get(name, None)
        if color is None:
            # Use default matplotlib color cycle
            color = None

        ax.plot(cumulative.index, cumulative.values, label=name, color=color, linewidth=2)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_yscale("linear")

    apply_style(fig, ax)

    return fig


def plot_rolling_metric(
    metric_series: pd.Series,
    title: str,
    ylabel: str,
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot rolling metric over time.

    Args:
        metric_series: Series of rolling metric values
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size. If None, uses config default.

    Returns:
        Matplotlib figure
    """
    setup_style()

    if figsize is None:
        figsize = get_figure_size("default")

    fig, ax = plt.subplots(figsize=figsize)
    colors = get_colors()

    ax.plot(metric_series.index, metric_series.values, linewidth=2, color=colors["varbx"])
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    apply_style(fig, ax)

    return fig


def plot_drawdown(
    drawdown_series: pd.Series,
    title: str = "Drawdown",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot drawdown series.

    Args:
        drawdown_series: Series of drawdown values
        title: Plot title
        figsize: Figure size. If None, uses config default.

    Returns:
        Matplotlib figure
    """
    setup_style()

    if figsize is None:
        figsize = get_figure_size("default")

    fig, ax = plt.subplots(figsize=figsize)
    colors = get_colors()

    # Fill area under curve
    ax.fill_between(
        drawdown_series.index,
        drawdown_series.values,
        0,
        color=colors["varbx"],
        alpha=0.3,
    )
    ax.plot(drawdown_series.index, drawdown_series.values, linewidth=2, color=colors["varbx"])

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.set_title(title)
    ax.set_ylim(bottom=min(drawdown_series.min() * 1.1, -0.01))

    apply_style(fig, ax)

    return fig


def plot_rolling_alpha_beta(
    alpha_beta_df: pd.DataFrame,
    title: str = "Rolling Alpha and Beta",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot rolling alpha and beta.

    Args:
        alpha_beta_df: DataFrame with 'alpha_annualized' and 'beta' columns
        title: Plot title
        figsize: Figure size. If None, uses config default.

    Returns:
        Matplotlib figure
    """
    setup_style()

    if figsize is None:
        figsize = get_figure_size("wide")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    colors = get_colors()

    # Plot alpha
    if "alpha_annualized" in alpha_beta_df.columns:
        ax1.plot(
            alpha_beta_df.index,
            alpha_beta_df["alpha_annualized"],
            linewidth=2,
            color=colors["varbx"],
            label="Alpha (Annualized)",
        )
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_ylabel("Alpha (Annualized)")
    ax1.set_title(title)
    ax1.legend(loc="best")

    # Plot beta
    if "beta" in alpha_beta_df.columns:
        ax2.plot(
            alpha_beta_df.index,
            alpha_beta_df["beta"],
            linewidth=2,
            color=colors["sp500"],
            label="Beta",
        )
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Beta")
    ax2.legend(loc="best")

    apply_style(fig, ax1)
    apply_style(fig, ax2)

    return fig


def plot_correlation_matrix(
    returns_df: pd.DataFrame,
    title: str = "Return Correlation Matrix",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot correlation matrix heatmap.

    Args:
        returns_df: DataFrame with return columns
        title: Plot title
        figsize: Figure size. If None, uses config default.

    Returns:
        Matplotlib figure
    """
    setup_style()

    if figsize is None:
        figsize = get_figure_size("default")

    # Calculate correlation
    corr = returns_df.corr()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

    # Add text annotations
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Correlation")

    apply_style(fig, ax)

    return fig


def plot_performance_attribution(
    metrics_dict: dict[str, dict[str, float]],
    title: str = "Performance Metrics Comparison",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot performance metrics comparison as bar chart.

    Args:
        metrics_dict: Dictionary mapping series names to metrics dictionaries
        title: Plot title
        figsize: Figure size. If None, uses config default.

    Returns:
        Matplotlib figure
    """
    setup_style()

    if figsize is None:
        figsize = get_figure_size("wide")

    # Extract metrics
    metric_names = ["cagr", "sharpe_ratio", "calmar_ratio"]
    series_names = list(metrics_dict.keys())

    # Create DataFrame
    data = {}
    for name in series_names:
        data[name] = [metrics_dict[name].get(metric, 0) for metric in metric_names]

    df = pd.DataFrame(data, index=metric_names)

    fig, ax = plt.subplots(figsize=figsize)
    colors = get_colors()

    # Color mapping
    color_list = [colors.get(name.lower().replace(" ", "_"), None) for name in series_names]
    color_list = [c if c else colors["varbx"] for c in color_list]

    df.plot(kind="bar", ax=ax, color=color_list, width=0.8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    apply_style(fig, ax)

    return fig

