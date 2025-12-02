"""Plotting functions for VARBX analysis."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
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

    color_map = {
        "VARBX": colors["varbx"],
        "VARBX (HFRI-benchmark)": colors["varbx"],
        "VARBX (SP500-benchmark)": colors["varbx"],
        "S&P 500": colors["sp500"],
        "SP500": colors["sp500"],
        "HFRI ED": colors["hfri_ed"],
        "HFRI": colors["hfri_ed"],
        "HFRI ED: Merger Arbitrage Index": colors["hfri_ed"],
    }

    for name, returns in returns_dict.items():
        cumulative = (1 + returns).cumprod()

        color = color_map.get(name, None)
        if color is None:
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

    if "alpha_annualized" in alpha_beta_df.columns and len(alpha_beta_df["alpha_annualized"].dropna()) > 0:
        ax1.plot(
            alpha_beta_df.index,
            alpha_beta_df["alpha_annualized"],
            linewidth=2,
            color=colors["varbx"],
            label="Alpha (Annualized)",
        )
        ax1.legend(loc="best")
    else:
        ax1.text(0.5, 0.5, "No alpha data available", ha="center", va="center", transform=ax1.transAxes)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax1.set_ylabel("Alpha (Annualized)")

    if "beta" in alpha_beta_df.columns and len(alpha_beta_df["beta"].dropna()) > 0:
        ax2.plot(
            alpha_beta_df.index,
            alpha_beta_df["beta"],
            linewidth=2,
            color=colors["sp500"],
            label="Beta",
        )
        ax2.legend(loc="best")
    else:
        ax2.text(0.5, 0.5, "No beta data available", ha="center", va="center", transform=ax2.transAxes)
    ax2.axhline(y=1, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Beta")

    fig.suptitle(title, y=1.0, fontsize=14)

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

    im = ax.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)

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

    metric_names = ["cagr", "sharpe_ratio", "calmar_ratio"]
    series_names = list(metrics_dict.keys())

    data = {}
    for name in series_names:
        data[name] = [metrics_dict[name].get(metric, 0) for metric in metric_names]

    df = pd.DataFrame(data, index=metric_names)

    fig, ax = plt.subplots(figsize=figsize)
    colors = get_colors()

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


def plot_rolling_metrics_comparison(
    metrics_dict: dict[str, pd.Series],
    title: str = "Rolling Metrics Comparison",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot multiple rolling metrics series for comparison.

    Args:
        metrics_dict: Dictionary mapping metric names to Series
        title: Plot title
        figsize: Figure size. If None, uses config default.

    Returns:
        Matplotlib figure
    """
    setup_style()

    if figsize is None:
        figsize = get_figure_size("wide")

    n_metrics = len(metrics_dict)
    if n_metrics == 0:
        raise ValueError("No metrics provided")

    fig, axes = plt.subplots(n_metrics, 1, figsize=(figsize[0], figsize[1] * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    colors = get_colors()
    color_cycle = [colors["varbx"], colors["sp500"], colors["hfri_ed"]]

    for idx, (metric_name, metric_series) in enumerate(metrics_dict.items()):
        ax = axes[idx]
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(metric_series.index, metric_series.values, linewidth=2, color=color, label=metric_name)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_ylabel(metric_name)
        ax.legend(loc="best")
        apply_style(fig, ax)

    axes[-1].set_xlabel("Date")
    fig.suptitle(title, y=1.0, fontsize=14)

    return fig


def plot_metrics_comparison_table(
    metrics_df: pd.DataFrame,
    title: str = "Performance Metrics Comparison",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot comprehensive metrics comparison as a table visualization.

    Args:
        metrics_df: DataFrame with metrics as rows and series as columns
        title: Plot title
        figsize: Figure size. If None, uses config default.

    Returns:
        Matplotlib figure
    """
    setup_style()

    if figsize is None:
        figsize = get_figure_size("wide")

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=metrics_df.round(4).values,
        rowLabels=metrics_df.index,
        colLabels=metrics_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    for i in range(len(metrics_df.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax.set_title(title, pad=20, fontsize=14, weight="bold")

    return fig


def plot_up_down_capture(
    up_capture: float,
    down_capture: float,
    series_name: str = "Asset",
    title: str = "Up/Down Capture",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot up and down capture ratios.

    Args:
        up_capture: Up capture ratio
        down_capture: Down capture ratio
        series_name: Name of the series
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

    categories = ["Up Capture", "Down Capture"]
    values = [up_capture, down_capture]
    bar_colors = [colors.get("sp500", "#2ca02c"), colors.get("varbx", "#1f77b4")]

    bars = ax.bar(categories, values, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=1.5)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.2f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontweight="bold",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="Benchmark (1.0)")
    ax.set_ylabel("Capture Ratio")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    apply_style(fig, ax)

    return fig


def plot_return_distribution(
    returns: pd.Series,
    title: str = "Return Distribution",
    figsize: Optional[tuple[float, float]] = None,
) -> plt.Figure:
    """Plot return distribution with skewness and kurtosis.

    Args:
        returns: Series of returns
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

    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    mean_return = returns.mean()

    ax.hist(returns, bins=30, alpha=0.7, color=colors["varbx"], edgecolor="black", density=True)

    ax.axvline(mean_return, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_return:.4f}")

    from scipy import stats

    x = np.linspace(returns.min(), returns.max(), 100)
    normal_dist = stats.norm.pdf(x, returns.mean(), returns.std())
    ax.plot(x, normal_dist, "k--", linewidth=2, label="Normal Distribution", alpha=0.7)

    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\nSkewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    apply_style(fig, ax)

    return fig

