"""Data preprocessing utilities."""

from typing import Optional

import numpy as np
import pandas as pd

from src.data.schema import validate_returns_series


def align_timeframes(
    *dfs: pd.DataFrame, date_column: str = "date"
) -> tuple[pd.DataFrame, ...]:
    """Align multiple DataFrames to common date range.

    Args:
        *dfs: Variable number of DataFrames to align
        date_column: Name of date column

    Returns:
        Tuple of aligned DataFrames
    """
    if not dfs:
        return tuple()

    # Find common date range
    date_series = [df[date_column] for df in dfs]
    common_start = max(s.min() for s in date_series)
    common_end = min(s.max() for s in date_series)

    # Filter each DataFrame to common range
    aligned_dfs = []
    for df in dfs:
        mask = (df[date_column] >= common_start) & (df[date_column] <= common_end)
        aligned_df = df[mask].copy()
        aligned_dfs.append(aligned_df)

    return tuple(aligned_dfs)


def handle_missing_values(
    series: pd.Series, method: str = "forward_fill"
) -> pd.Series:
    """Handle missing values in returns series.

    Args:
        series: Returns series
        method: Method to use ('forward_fill', 'backward_fill', 'drop', 'zero')

    Returns:
        Series with missing values handled
    """
    validate_returns_series(series)

    if method == "forward_fill":
        return series.ffill()
    elif method == "backward_fill":
        return series.bfill()
    elif method == "drop":
        return series.dropna()
    elif method == "zero":
        return series.fillna(0.0)
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'forward_fill', 'backward_fill', 'drop', or 'zero'"
        )


def calculate_returns_from_prices(
    prices: pd.Series, method: str = "simple"
) -> pd.Series:
    """Calculate returns from price series.

    Args:
        prices: Price series
        method: Return calculation method ('simple' or 'log')

    Returns:
        Returns series
    """
    if method == "simple":
        returns = prices.pct_change()
    elif method == "log":
        returns = pd.Series(
            np.log(prices / prices.shift(1)), index=prices.index
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'simple' or 'log'")

    return returns.dropna()


def merge_returns_dataframes(
    *dfs: pd.DataFrame,
    date_column: str = "date",
    suffixes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Merge multiple returns DataFrames on date.

    Args:
        *dfs: Variable number of DataFrames to merge
        date_column: Name of date column
        suffixes: Suffixes for return columns. If None, uses default numbering.

    Returns:
        Merged DataFrame
    """
    if not dfs:
        return pd.DataFrame()

    # Align timeframes first
    aligned_dfs = align_timeframes(*dfs, date_column=date_column)

    # Start with first DataFrame and rename its return column
    result = aligned_dfs[0].copy()
    if suffixes and len(suffixes) > 0:
        first_suffix = suffixes[0]
        result = result.rename(columns={"return": f"return_{first_suffix}"})
    else:
        result = result.rename(columns={"return": "return_0"})

    # Merge remaining DataFrames
    for i, df in enumerate(aligned_dfs[1:], start=1):
        if suffixes and i < len(suffixes):
            suffix = suffixes[i]
            return_col = f"return_{suffix}"
        else:
            return_col = f"return_{i}"

        # Rename return column
        df_renamed = df.rename(columns={"return": return_col})

        # Merge on date
        result = pd.merge(
            result, df_renamed[[date_column, return_col]], on=date_column, how="inner"
        )

    return result.sort_values(date_column).reset_index(drop=True)


def clean_returns_dataframe(
    df: pd.DataFrame,
    date_column: str = "date",
    return_column: str = "return",
    handle_missing: str = "drop",
) -> pd.DataFrame:
    """Clean returns DataFrame.

    Args:
        df: Returns DataFrame
        date_column: Name of date column
        return_column: Name of return column
        handle_missing: Method to handle missing values

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_clean[date_column]):
        df_clean[date_column] = pd.to_datetime(df_clean[date_column])

    # Sort by date
    df_clean = df_clean.sort_values(date_column).reset_index(drop=True)

    # Handle missing values in return column
    if return_column in df_clean.columns:
        df_clean[return_column] = handle_missing_values(
            df_clean[return_column], method=handle_missing
        )

    # Remove rows with missing dates
    df_clean = df_clean.dropna(subset=[date_column])

    return df_clean

