"""Path management utilities using config.yml."""

from pathlib import Path
from typing import Optional

from src.config import get_config


def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root
    """
    # Assume this file is in src/utils/, go up two levels
    return Path(__file__).parent.parent.parent


def get_data_raw_path() -> Path:
    """Get path to raw data directory.

    Returns:
        Path to data/raw/
    """
    config = get_config()
    path_str = config.get("paths.data_raw", "data/raw")
    return get_project_root() / path_str


def get_data_interim_path() -> Path:
    """Get path to interim data directory.

    Returns:
        Path to data/interim/
    """
    config = get_config()
    path_str = config.get("paths.data_interim", "data/interim")
    return get_project_root() / path_str


def get_outputs_figures_path() -> Path:
    """Get path to figures output directory.

    Returns:
        Path to outputs/figures/
    """
    config = get_config()
    path_str = config.get("paths.outputs_figures", "outputs/figures")
    return get_project_root() / path_str


def get_outputs_tables_path() -> Path:
    """Get path to tables output directory.

    Returns:
        Path to outputs/tables/
    """
    config = get_config()
    path_str = config.get("paths.outputs_tables", "outputs/tables")
    return get_project_root() / path_str


def get_notebooks_path() -> Path:
    """Get path to notebooks directory.

    Returns:
        Path to notebooks/
    """
    config = get_config()
    path_str = config.get("paths.notebooks", "notebooks")
    return get_project_root() / path_str


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path

