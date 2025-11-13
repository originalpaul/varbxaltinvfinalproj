"""File I/O utilities for data and output management."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.paths import ensure_dir, get_outputs_figures_path, get_outputs_tables_path


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    output_dir: Optional[Path] = None,
    formats: Optional[list[str]] = None,
) -> None:
    """Save DataFrame to multiple formats.

    Args:
        df: DataFrame to save
        filename: Base filename (without extension)
        output_dir: Output directory. If None, uses outputs/tables/
        formats: List of formats ('csv', 'latex', 'json'). If None, uses config defaults.
    """
    if output_dir is None:
        output_dir = get_outputs_tables_path()

    ensure_dir(output_dir)

    if formats is None:
        from src.config import get_config

        config = get_config()
        formats = config.export.get("table_formats", ["csv"])

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        if fmt == "csv":
            df.to_csv(filepath, index=True)
        elif fmt == "latex":
            try:
                df.to_latex(filepath, index=True, float_format="%.4f")
            except ImportError as e:
                if "jinja2" in str(e).lower():
                    raise ImportError(
                        "LaTeX export requires jinja2. Install with: pip install jinja2"
                    ) from e
                raise
        elif fmt == "json":
            df.to_json(filepath, orient="records", indent=2)
        else:
            raise ValueError(f"Unsupported format: {fmt}")


def save_figure(
    fig: Any,  # matplotlib.figure.Figure
    filename: str,
    output_dir: Optional[Path] = None,
    formats: Optional[list[str]] = None,
    dpi: Optional[int] = None,
    bbox_inches: str = "tight",
) -> None:
    """Save matplotlib figure to multiple formats.

    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Output directory. If None, uses outputs/figures/
        formats: List of formats ('png', 'pdf', 'svg'). If None, uses config defaults.
        dpi: DPI for raster formats. If None, uses config default.
        bbox_inches: Bounding box for figure saving
    """
    if output_dir is None:
        output_dir = get_outputs_figures_path()

    ensure_dir(output_dir)

    if formats is None:
        from src.config import get_config

        config = get_config()
        formats = config.export.get("figure_formats", ["png"])

    if dpi is None:
        from src.config import get_config

        config = get_config()
        dpi = config.viz.get("dpi", 300)

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(
            filepath,
            format=fmt,
            dpi=dpi,
            bbox_inches=bbox_inches,
            facecolor="white",
        )


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with JSON contents
    """
    with open(filepath, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to save JSON file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

