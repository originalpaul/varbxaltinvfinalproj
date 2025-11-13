"""Performance metrics calculations."""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import get_config


def calculate_cagr(
    returns: pd.Series, periods_per_year: Optional[int] = None
) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).

    Formula: CAGR = (Ending Value / Beginning Value)^(1/n) - 1
    Where n is the number of years.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year (e.g., 12 for monthly)

    Returns:
        CAGR as decimal (e.g., 0.10 for 10%)
    """
    if periods_per_year is None:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)

    if len(returns) == 0:
        return np.nan

    # Calculate cumulative return
    cumulative_return = (1 + returns).prod() - 1

    # Calculate number of years
    n_periods = len(returns)
    n_years = n_periods / periods_per_year

    if n_years <= 0:
        return np.nan

    # Calculate CAGR
    cagr = (1 + cumulative_return) ** (1 / n_years) - 1

    return cagr


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    periods_per_year: Optional[int] = None,
) -> float:
    """Calculate annualized Sharpe ratio.

    Formula: Sharpe = (Mean Return - Risk-Free Rate) / Std Dev of Returns
    Annualized: Sharpe = sqrt(periods_per_year) * (Mean - Rf) / Std Dev

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate as decimal (e.g., 0.02 for 2%)
        periods_per_year: Number of periods per year (e.g., 12 for monthly)

    Returns:
        Annualized Sharpe ratio
    """
    if periods_per_year is None:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)

    if risk_free_rate is None:
        config = get_config()
        risk_free_rate = config.analysis.get("risk_free_rate", 0.02)

    if len(returns) == 0:
        return np.nan

    # Convert annual risk-free rate to periodic
    periodic_rf = risk_free_rate / periods_per_year

    # Calculate excess returns
    excess_returns = returns - periodic_rf

    # Calculate mean and std dev
    mean_excess = excess_returns.mean()
    std_dev = excess_returns.std()

    if std_dev == 0 or np.isnan(std_dev):
        return np.nan

    # Annualize
    sharpe = np.sqrt(periods_per_year) * (mean_excess / std_dev)

    return sharpe


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown.

    Maximum drawdown is the largest peak-to-trough decline in cumulative returns.

    Args:
        returns: Series of periodic returns

    Returns:
        Maximum drawdown as decimal (e.g., -0.20 for -20%)
    """
    if len(returns) == 0:
        return np.nan

    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()

    # Calculate running maximum
    running_max = cumulative.expanding().max()

    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max

    # Maximum drawdown is the minimum (most negative) value
    max_dd = drawdown.min()

    return max_dd


def calculate_calmar_ratio(
    returns: pd.Series, periods_per_year: Optional[int] = None
) -> float:
    """Calculate Calmar ratio.

    Formula: Calmar = CAGR / |Max Drawdown|

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year (e.g., 12 for monthly)

    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(returns, periods_per_year=periods_per_year)
    max_dd = calculate_max_drawdown(returns)

    if max_dd == 0 or np.isnan(max_dd) or np.isnan(cagr):
        return np.nan

    calmar = cagr / abs(max_dd)

    return calmar


def calculate_total_return(returns: pd.Series) -> float:
    """Calculate total return over the period.

    Args:
        returns: Series of periodic returns

    Returns:
        Total return as decimal
    """
    if len(returns) == 0:
        return np.nan

    total_return = (1 + returns).prod() - 1
    return total_return


def calculate_rolling_cagr(
    returns: pd.Series, window: int, periods_per_year: Optional[int] = None
) -> pd.Series:
    """Calculate rolling CAGR.

    Args:
        returns: Series of periodic returns
        window: Rolling window size in periods
        periods_per_year: Number of periods per year

    Returns:
        Series of rolling CAGR values
    """
    rolling_cagr = returns.rolling(window=window).apply(
        lambda x: calculate_cagr(x, periods_per_year=periods_per_year),
        raw=False,
    )
    return rolling_cagr


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: Optional[float] = None,
    periods_per_year: Optional[int] = None,
) -> pd.Series:
    """Calculate rolling Sharpe ratio.

    Args:
        returns: Series of periodic returns
        window: Rolling window size in periods
        risk_free_rate: Annual risk-free rate as decimal
        periods_per_year: Number of periods per year

    Returns:
        Series of rolling Sharpe ratios
    """
    rolling_sharpe = returns.rolling(window=window).apply(
        lambda x: calculate_sharpe_ratio(
            x, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        raw=False,
    )
    return rolling_sharpe


def calculate_rolling_max_drawdown(returns: pd.Series, window: int) -> pd.Series:
    """Calculate rolling maximum drawdown.

    Args:
        returns: Series of periodic returns
        window: Rolling window size in periods

    Returns:
        Series of rolling max drawdown values
    """
    rolling_max_dd = returns.rolling(window=window).apply(
        lambda x: calculate_max_drawdown(x), raw=False
    )
    return rolling_max_dd


def calculate_all_metrics(
    returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    periods_per_year: Optional[int] = None,
) -> dict[str, float]:
    """Calculate all performance metrics.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate as decimal
        periods_per_year: Number of periods per year

    Returns:
        Dictionary with all performance metrics
    """
    return {
        "cagr": calculate_cagr(returns, periods_per_year=periods_per_year),
        "total_return": calculate_total_return(returns),
        "sharpe_ratio": calculate_sharpe_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        "max_drawdown": calculate_max_drawdown(returns),
        "calmar_ratio": calculate_calmar_ratio(returns, periods_per_year=periods_per_year),
    }

