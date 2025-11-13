"""Data loading utilities for VARBX and benchmark data."""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import get_config
from src.data.schema import validate_returns_dataframe
from src.utils.paths import get_data_raw_path


def load_csv_returns(
    filename: str, date_column: str = "date", return_column: str = "return"
) -> pd.DataFrame:
    """Load returns data from CSV file.

    Args:
        filename: CSV filename
        date_column: Name of date column
        return_column: Name of return column

    Returns:
        DataFrame with date and return columns
    """
    data_path = get_data_raw_path()
    filepath = data_path / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Parse date column
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])

    # Select only date and return columns
    columns_to_keep = [date_column, return_column]
    if return_column not in df.columns:
        # Try to find return column by common names
        possible_names = ["return", "returns", "ret", "monthly_return"]
        for name in possible_names:
            if name in df.columns:
                return_column = name
                columns_to_keep = [date_column, return_column]
                break
        else:
            raise ValueError(
                f"Return column '{return_column}' not found. Available columns: {df.columns.tolist()}"
            )

    df = df[columns_to_keep].copy()
    df = df.rename(columns={return_column: "return"})

    # Sort by date
    df = df.sort_values(date_column).reset_index(drop=True)

    # Validate
    validate_returns_dataframe(df, date_column=date_column)

    return df


def download_benchmark_data(
    ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> pd.DataFrame:
    """Download benchmark data using yfinance.

    Args:
        ticker: Stock ticker symbol (e.g., 'SPY', 'AGG')
        start_date: Start date in YYYY-MM-DD format. If None, downloads all available.
        end_date: End date in YYYY-MM-DD format. If None, uses today.

    Returns:
        DataFrame with date and monthly return columns
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for downloading data. Install with: pip install yfinance"
        )

    # Download data
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    if hist.empty:
        raise ValueError(f"No data downloaded for ticker {ticker}")

    # Calculate monthly returns
    hist = hist[["Close"]].copy()
    hist["return"] = hist["Close"].pct_change()
    hist = hist.dropna()

    # Resample to monthly (last trading day of month)
    hist_monthly = hist.resample("M").last()
    hist_monthly["return"] = hist_monthly["Close"].pct_change()
    hist_monthly = hist_monthly.dropna()

    # Reset index to get date as column
    hist_monthly = hist_monthly.reset_index()
    hist_monthly = hist_monthly.rename(columns={"Date": "date"})
    hist_monthly = hist_monthly[["date", "return"]].copy()

    return hist_monthly


def load_varbx_data() -> pd.DataFrame:
    """Load VARBX returns data.

    Returns:
        DataFrame with date and return columns
    """
    config = get_config()
    filename = config.data.get("varbx_file", "varbx_monthly_returns.csv")
    date_col = config.data.get("date_column", "date")
    return_col = config.data.get("return_columns", {}).get("varbx", "return")

    return load_csv_returns(filename, date_column=date_col, return_column=return_col)


def load_sp500_data() -> pd.DataFrame:
    """Load S&P 500 returns data.

    Returns:
        DataFrame with date and return columns
    """
    config = get_config()

    # Check if using yfinance
    if config.data.get("use_yfinance", False):
        ticker = config.data.get("benchmark_tickers", {}).get("sp500", "SPY")
        return download_benchmark_data(ticker)
    else:
        filename = config.data.get("sp500_file", "sp500_monthly.csv")
        date_col = config.data.get("date_column", "date")
        return_col = config.data.get("return_columns", {}).get("sp500", "return")
        return load_csv_returns(filename, date_column=date_col, return_column=return_col)


def load_agg_data() -> pd.DataFrame:
    """Load Bloomberg US Aggregate returns data.

    Returns:
        DataFrame with date and return columns
    """
    config = get_config()

    # Check if using yfinance
    if config.data.get("use_yfinance", False):
        ticker = config.data.get("benchmark_tickers", {}).get("agg", "AGG")
        return download_benchmark_data(ticker)
    else:
        filename = config.data.get("agg_file", "agg_monthly.csv")
        date_col = config.data.get("date_column", "date")
        return_col = config.data.get("return_columns", {}).get("agg", "return")
        return load_csv_returns(filename, date_column=date_col, return_column=return_col)


def download_benchmarks() -> None:
    """Download benchmark data and save to CSV files."""
    config = get_config()
    data_path = get_data_raw_path()
    data_path.mkdir(parents=True, exist_ok=True)

    # Download SP500
    sp500_ticker = config.data.get("benchmark_tickers", {}).get("sp500", "SPY")
    sp500_df = download_benchmark_data(sp500_ticker)
    sp500_file = data_path / config.data.get("sp500_file", "sp500_monthly.csv")
    sp500_df.to_csv(sp500_file, index=False)
    print(f"Downloaded SP500 data to {sp500_file}")

    # Download AGG
    agg_ticker = config.data.get("benchmark_tickers", {}).get("agg", "AGG")
    agg_df = download_benchmark_data(agg_ticker)
    agg_file = data_path / config.data.get("agg_file", "agg_monthly.csv")
    agg_df.to_csv(agg_file, index=False)
    print(f"Downloaded AGG data to {agg_file}")

