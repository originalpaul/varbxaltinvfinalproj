"""Regression analysis for alpha and beta calculations."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from src.config import get_config


def calculate_alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: Optional[int] = None,
    risk_free_rate: Optional[float] = None,
) -> dict[str, float]:
    """Calculate alpha and beta using OLS regression.

    Regression: (Return - Rf) = Alpha + Beta * (Benchmark - Rf) + Error

    Args:
        returns: Series of asset returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year (for annualizing alpha)
        risk_free_rate: Annual risk-free rate as decimal

    Returns:
        Dictionary with 'alpha', 'beta', 'r_squared', and 'alpha_annualized'
    """
    if periods_per_year is None:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)

    if risk_free_rate is None:
        config = get_config()
        risk_free_rate = config.analysis.get("risk_free_rate", 0.02)

    # Align series
    aligned = pd.DataFrame({"returns": returns, "benchmark": benchmark_returns}).dropna()

    if len(aligned) < 2:
        return {
            "alpha": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "alpha_annualized": np.nan,
        }

    asset_returns = aligned["returns"].values
    bench_returns = aligned["benchmark"].values

    # Convert annual risk-free rate to periodic
    periodic_rf = risk_free_rate / periods_per_year

    # Calculate excess returns
    asset_excess = asset_returns - periodic_rf
    bench_excess = bench_returns - periodic_rf

    # Reshape for sklearn
    X = bench_excess.reshape(-1, 1)
    y = asset_excess

    # Fit OLS regression
    model = LinearRegression()
    model.fit(X, y)

    # Extract coefficients
    beta = model.coef_[0]
    alpha = model.intercept_

    # Calculate R-squared
    y_pred = model.predict(X)
    r_squared = r2_score(y, y_pred)

    # Annualize alpha
    alpha_annualized = alpha * periods_per_year

    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        "alpha_annualized": alpha_annualized,
    }


def calculate_rolling_alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int,
    periods_per_year: Optional[int] = None,
    risk_free_rate: Optional[float] = None,
) -> pd.DataFrame:
    """Calculate rolling alpha and beta.

    Args:
        returns: Series of asset returns
        benchmark_returns: Series of benchmark returns
        window: Rolling window size in periods
        periods_per_year: Number of periods per year
        risk_free_rate: Annual risk-free rate as decimal

    Returns:
        DataFrame with columns 'alpha', 'beta', 'r_squared', 'alpha_annualized'
    """
    # Align series
    aligned = pd.DataFrame({"returns": returns, "benchmark": benchmark_returns}).dropna()

    if len(aligned) < window:
        return pd.DataFrame(
            columns=["alpha", "beta", "r_squared", "alpha_annualized"],
            index=aligned.index,
        )

    results = []

    for i in range(window - 1, len(aligned)):
        window_returns = aligned["returns"].iloc[i - window + 1 : i + 1]
        window_benchmark = aligned["benchmark"].iloc[i - window + 1 : i + 1]

        metrics = calculate_alpha_beta(
            window_returns,
            window_benchmark,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        )

        results.append(metrics)

    # Create DataFrame with same index as aligned data (starting from window-1)
    result_df = pd.DataFrame(
        results,
        index=aligned.index[window - 1 :],
        columns=["alpha", "beta", "r_squared", "alpha_annualized"],
    )

    return result_df


def calculate_tracking_error(
    returns: pd.Series, benchmark_returns: pd.Series, annualized: bool = True
) -> float:
    """Calculate tracking error (std dev of active returns).

    Args:
        returns: Series of asset returns
        benchmark_returns: Series of benchmark returns
        annualized: If True, annualize the tracking error

    Returns:
        Tracking error as decimal
    """
    # Align series
    aligned = pd.DataFrame({"returns": returns, "benchmark": benchmark_returns}).dropna()

    if len(aligned) < 2:
        return np.nan

    # Calculate active returns (difference)
    active_returns = aligned["returns"] - aligned["benchmark"]

    # Calculate standard deviation
    tracking_error = active_returns.std()

    if annualized:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)
        tracking_error = tracking_error * np.sqrt(periods_per_year)

    return tracking_error


def calculate_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: Optional[int] = None,
) -> float:
    """Calculate information ratio (active return / tracking error).

    Args:
        returns: Series of asset returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Information ratio
    """
    if periods_per_year is None:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)

    # Align series
    aligned = pd.DataFrame({"returns": returns, "benchmark": benchmark_returns}).dropna()

    if len(aligned) < 2:
        return np.nan

    # Calculate active returns
    active_returns = aligned["returns"] - aligned["benchmark"]

    # Mean active return (annualized)
    mean_active = active_returns.mean() * periods_per_year

    # Tracking error (annualized)
    tracking_error = calculate_tracking_error(returns, benchmark_returns, annualized=True)

    if tracking_error == 0 or np.isnan(tracking_error):
        return np.nan

    information_ratio = mean_active / tracking_error

    return information_ratio

