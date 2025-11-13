"""Data validation schemas."""

from typing import Any, Dict

import pandas as pd


def validate_returns_dataframe(df: pd.DataFrame, date_column: str = "date") -> None:
    """Validate returns DataFrame structure.

    Args:
        df: DataFrame to validate
        date_column: Name of date column

    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")

    # Check if date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        raise ValueError(f"Date column '{date_column}' must be datetime type")

    # Check for duplicate dates
    if df[date_column].duplicated().any():
        raise ValueError("Duplicate dates found in DataFrame")

    # Check for missing values in date column
    if df[date_column].isna().any():
        raise ValueError("Missing values found in date column")

    # Check that at least one return column exists
    return_cols = [col for col in df.columns if col != date_column]
    if not return_cols:
        raise ValueError("No return columns found in DataFrame")


def validate_returns_series(series: pd.Series, name: str = "returns") -> None:
    """Validate returns series.

    Args:
        series: Series to validate
        name: Name of series for error messages

    Raises:
        ValueError: If validation fails
    """
    if series.empty:
        raise ValueError(f"{name} series is empty")

    # Check for all NaN
    if series.isna().all():
        raise ValueError(f"{name} series contains only NaN values")

    # Check for infinite values
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"{name} series must be numeric")

    if (series == float("inf")).any() or (series == float("-inf")).any():
        raise ValueError(f"{name} series contains infinite values")


def get_expected_schema() -> Dict[str, Any]:
    """Get expected schema for returns data.

    Returns:
        Dictionary with expected column names and types
    """
    return {
        "date": "datetime64[ns]",
        "return": "float64",
    }

