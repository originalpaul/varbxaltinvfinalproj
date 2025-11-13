"""Matplotlib styling configuration."""

import matplotlib.pyplot as plt
import matplotlib

from src.config import get_config


def setup_style() -> None:
    """Set up matplotlib style for institutional-quality plots."""
    config = get_config()
    viz_config = config.viz

    # Font settings
    font_family = viz_config.get("font", {}).get("family", "sans-serif")
    font_size = viz_config.get("font", {}).get("size", 11)
    title_size = viz_config.get("font", {}).get("title_size", 14)

    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": font_size,
            "axes.titlesize": title_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.titlesize": title_size + 2,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def get_colors() -> dict[str, str]:
    """Get color palette from config.

    Returns:
        Dictionary with color names and hex values
    """
    config = get_config()
    colors = config.viz.get("colors", {})
    return {
        "varbx": colors.get("varbx", "#1f77b4"),
        "sp500": colors.get("sp500", "#ff7f0e"),
        "agg": colors.get("agg", "#2ca02c"),
        "grid": colors.get("grid", "#e0e0e0"),
        "text": colors.get("text", "#333333"),
    }


def get_figure_size(size_type: str = "default") -> tuple[float, float]:
    """Get figure size from config.

    Args:
        size_type: Size type ('default', 'wide', 'tall')

    Returns:
        Tuple of (width, height) in inches
    """
    config = get_config()
    figure_sizes = config.viz.get("figure_size", {})
    size = figure_sizes.get(size_type, [10, 6])
    return tuple(size)


def apply_style(fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes) -> None:
    """Apply styling to figure and axes.

    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
    """
    colors = get_colors()

    # Set grid color
    ax.grid(True, alpha=0.3, color=colors["grid"], linewidth=0.5)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set text color
    ax.xaxis.label.set_color(colors["text"])
    ax.yaxis.label.set_color(colors["text"])
    ax.title.set_color(colors["text"])

    # Set tick colors
    ax.tick_params(colors=colors["text"])

    # Tight layout
    fig.tight_layout()

