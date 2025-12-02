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

    non_empty_dfs = [df for df in dfs if len(df) > 0]
    empty_indices = [i for i, df in enumerate(dfs) if len(df) == 0]
    
    if not non_empty_dfs:
        return tuple(dfs)  # Return original if all are empty

    # Normalize all dates to timezone-naive and ensure datetime type
    normalized_dfs = []
    for df in non_empty_dfs:
        df_copy = df.copy()
        if date_column in df_copy.columns and len(df_copy) > 0:
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            if hasattr(df_copy[date_column].dtype, 'tz') and df_copy[date_column].dt.tz is not None:
                df_copy[date_column] = df_copy[date_column].dt.tz_localize(None)
            elif hasattr(df_copy[date_column], 'dt') and df_copy[date_column].dt.tz is not None:
                df_copy[date_column] = df_copy[date_column].dt.tz_localize(None)
        normalized_dfs.append(df_copy)

    if normalized_dfs and all(len(df) > 0 for df in normalized_dfs):
        date_series = [df[date_column] for df in normalized_dfs]
        common_start = max(s.min() for s in date_series)
        common_end = min(s.max() for s in date_series)

        aligned_dfs = []
        for df in normalized_dfs:
            if len(df) > 0:
                mask = (df[date_column] >= common_start) & (df[date_column] <= common_end)
                aligned_df = df[mask].copy()
                aligned_dfs.append(aligned_df)
            else:
                aligned_dfs.append(df)
        
        result = []
        aligned_idx = 0
        for i, original_df in enumerate(dfs):
            if i in empty_indices:
                result.append(original_df)
            else:
                result.append(aligned_dfs[aligned_idx])
                aligned_idx += 1
        return tuple(result)
    
    return tuple(dfs)


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

    non_empty_with_idx = [(i, df) for i, df in enumerate(dfs) if len(df) > 0]
    empty_indices = [i for i, df in enumerate(dfs) if len(df) == 0]

    if not non_empty_with_idx:
        if suffixes and len(suffixes) > 0:
            cols = [date_column] + [f"return_{s}" for s in suffixes]
        else:
            cols = [date_column] + [f"return_{i}" for i in range(len(dfs))]
        return pd.DataFrame(columns=cols)

    non_empty_dfs = [df for _, df in non_empty_with_idx]
    non_empty_indices = [i for i, _ in non_empty_with_idx]

    normalized_dfs = []
    for df in non_empty_dfs:
        df_copy = df.copy()
        if date_column in df_copy.columns and len(df_copy) > 0:
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
                df_copy[date_column] = pd.to_datetime(df_copy[date_column])
            if hasattr(df_copy[date_column], 'dt') and df_copy[date_column].dt.tz is not None:
                df_copy[date_column] = df_copy[date_column].dt.tz_localize(None)
        normalized_dfs.append(df_copy)

    result = normalized_dfs[0].copy()
    first_idx = non_empty_indices[0]
    
    if suffixes and first_idx < len(suffixes):
        first_suffix = suffixes[first_idx]
        result = result.rename(columns={"return": f"return_{first_suffix}"})
    else:
        result = result.rename(columns={"return": f"return_{first_idx}"})

    for i, df in enumerate(normalized_dfs[1:], start=1):
        if len(df) == 0:
            continue
        
        orig_idx = non_empty_indices[i]
        if suffixes and orig_idx < len(suffixes):
            suffix = suffixes[orig_idx]
            return_col = f"return_{suffix}"
        else:
            return_col = f"return_{orig_idx}"

        df_renamed = df.rename(columns={"return": return_col})

        result = pd.merge(
            result, df_renamed[[date_column, return_col]], on=date_column, how="outer"
        )

    for empty_idx in empty_indices:
        if suffixes and empty_idx < len(suffixes):
            suffix = suffixes[empty_idx]
        else:
            suffix = str(empty_idx)
        col_name = f"return_{suffix}"
        if col_name not in result.columns:
            result[col_name] = np.nan

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

    if not pd.api.types.is_datetime64_any_dtype(df_clean[date_column]):
        df_clean[date_column] = pd.to_datetime(df_clean[date_column])
    
    if len(df_clean) > 0:
        if hasattr(df_clean[date_column].dtype, 'tz') and df_clean[date_column].dt.tz is not None:
            df_clean[date_column] = df_clean[date_column].dt.tz_localize(None)
        elif hasattr(df_clean[date_column], 'dt') and df_clean[date_column].dt.tz is not None:
            df_clean[date_column] = df_clean[date_column].dt.tz_localize(None)

    df_clean = df_clean.sort_values(date_column).reset_index(drop=True)

    if return_column in df_clean.columns:
        df_clean[return_column] = handle_missing_values(
            df_clean[return_column], method=handle_missing
        )

    df_clean = df_clean.dropna(subset=[date_column])

    return df_clean

