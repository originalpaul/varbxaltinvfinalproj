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

    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])

    columns_to_keep = [date_column, return_column]
    if return_column not in df.columns:
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

    df = df.sort_values(date_column).reset_index(drop=True)

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

    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)

    if hist.empty:
        raise ValueError(f"No data downloaded for ticker {ticker}")

    hist = hist[["Close"]].copy()
    hist["return"] = hist["Close"].pct_change()
    hist = hist.dropna()

    hist_monthly = hist.resample("M").last()
    hist_monthly["return"] = hist_monthly["Close"].pct_change()
    hist_monthly = hist_monthly.dropna()

    hist_monthly = hist_monthly.reset_index()
    hist_monthly = hist_monthly.rename(columns={"Date": "date"})
    
    if hist_monthly["date"].dt.tz is not None:
        hist_monthly["date"] = hist_monthly["date"].dt.tz_localize(None)
    
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

    if config.data.get("use_yfinance", False):
        ticker = config.data.get("benchmark_tickers", {}).get("sp500", "SPY")
        return download_benchmark_data(ticker)
    else:
        filename = config.data.get("sp500_file", "sp500_monthly.csv")
        date_col = config.data.get("date_column", "date")
        return_col = config.data.get("return_columns", {}).get("sp500", "return")
        return load_csv_returns(filename, date_column=date_col, return_column=return_col)


def load_hfri_ed_data() -> pd.DataFrame:
    """Load HFRI ED (Merger Arbitrage Index) returns data from CSV.

    Returns:
        DataFrame with date and return columns
    """
    config = get_config()
    data_path = get_data_raw_path()
    
    project_root = data_path.parent.parent
    hfri_file = project_root / "hfri_merger_arb.csv"
    
    if hfri_file.exists():
        df = pd.read_csv(hfri_file)
        
        if "Performance Date" in df.columns and "Return" in df.columns:
            df = df.rename(columns={"Performance Date": "date", "Return": "return"})
            df = df[["date", "return"]].copy()
            
            df["date"] = pd.to_datetime(df["date"])
            
            df = df.sort_values("date").reset_index(drop=True)
            
            validate_returns_dataframe(df, date_column="date")
            
            return df
        else:
            available_cols = ", ".join(df.columns.tolist())
            raise ValueError(
                f"HFRI ED CSV file exists at {hfri_file} but doesn't have expected columns. "
                f"Expected: 'Performance Date' and 'Return'. "
                f"Found columns: {available_cols}"
            )
    
    filename = config.data.get("hfri_ed_file", "hfri_merger_arb.csv")
    filepath = data_path / filename
    
    if filepath.exists():
        date_col = config.data.get("date_column", "date")
        return_col = config.data.get("return_columns", {}).get("hfri_ed", "return")
        return load_csv_returns(filename, date_column=date_col, return_column=return_col)
    
    print(
        f"Warning: Could not load HFRI ED data. "
        f"CSV file not found at {hfri_file} or {filepath}. "
        f"Please ensure hfri_merger_arb.csv exists in the project root."
    )
    return pd.DataFrame(columns=["date", "return"])


def load_hfri_data() -> pd.DataFrame:
    """Load HFRI ED (Merger Arbitrage Index) returns data.
    
    This is an alias for load_hfri_ed_data() for backward compatibility.

    Returns:
        DataFrame with date and return columns
    """
    return load_hfri_ed_data()


def download_benchmarks() -> None:
    """Download benchmark data and save to CSV files."""
    config = get_config()
    data_path = get_data_raw_path()
    data_path.mkdir(parents=True, exist_ok=True)

    sp500_ticker = config.data.get("benchmark_tickers", {}).get("sp500", "SPY")
    sp500_df = download_benchmark_data(sp500_ticker)
    sp500_file = data_path / config.data.get("sp500_file", "sp500_monthly.csv")
    sp500_df.to_csv(sp500_file, index=False)
    print(f"Downloaded SP500 data to {sp500_file}")

    print("Note: HFRI ED data should be loaded from hfri_merger_arb.csv in project root")

