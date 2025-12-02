"""Risk metrics calculations."""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import get_config


def calculate_volatility(
    returns: pd.Series, periods_per_year: Optional[int] = None, annualized: bool = True
) -> float:
    """Calculate volatility (standard deviation of returns).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year (e.g., 12 for monthly)
        annualized: If True, annualize the volatility

    Returns:
        Volatility as decimal (annualized if annualized=True)
    """
    if len(returns) == 0:
        return np.nan

    std_dev = returns.std()

    if annualized:
        if periods_per_year is None:
            config = get_config()
            periods_per_year = config.analysis.get("periods_per_year", 12)
        std_dev = std_dev * np.sqrt(periods_per_year)

    return std_dev


def calculate_var(
    returns: pd.Series, confidence_level: float = 0.95, method: str = "historical"
) -> float:
    """Calculate Value at Risk (VaR).

    Args:
        returns: Series of periodic returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: Method to use ('historical' or 'parametric')

    Returns:
        VaR as decimal (negative value, e.g., -0.05 for -5%)
    """
    if len(returns) == 0:
        return np.nan

    if method == "historical":
        # Historical VaR: percentile of returns
        var = np.percentile(returns, (1 - confidence_level) * 100)
    elif method == "parametric":
        # Parametric VaR: assumes normal distribution
        mean = returns.mean()
        std = returns.std()
        z_score = -np.abs(np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100))
        var = mean + z_score * std
    else:
        raise ValueError(f"Unknown method: {method}. Use 'historical' or 'parametric'")

    return var


def calculate_cvar(
    returns: pd.Series, confidence_level: float = 0.95
) -> float:
    """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    CVaR is the expected loss given that the loss exceeds VaR.

    Args:
        returns: Series of periodic returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        CVaR as decimal (negative value)
    """
    if len(returns) == 0:
        return np.nan

    var = calculate_var(returns, confidence_level=confidence_level)

    # Calculate mean of returns below VaR
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        return var

    cvar = tail_returns.mean()

    return cvar


def calculate_downside_deviation(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    annualized: bool = True,
    threshold: float = 0.0,
) -> float:
    """Calculate downside deviation.

    Downside deviation only considers returns below a threshold (typically 0).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year
        annualized: If True, annualize the deviation
        threshold: Threshold for downside (default 0)

    Returns:
        Downside deviation as decimal
    """
    if len(returns) == 0:
        return np.nan

    # Only consider returns below threshold
    downside_returns = returns[returns < threshold] - threshold

    if len(downside_returns) == 0:
        return 0.0

    # Calculate standard deviation of downside returns
    downside_std = downside_returns.std()

    if annualized:
        if periods_per_year is None:
            config = get_config()
            periods_per_year = config.analysis.get("periods_per_year", 12)
        downside_std = downside_std * np.sqrt(periods_per_year)

    return downside_std


def calculate_drawdown_series(returns: pd.Series) -> pd.Series:
    """Calculate drawdown series.

    Args:
        returns: Series of periodic returns

    Returns:
        Series of drawdown values (negative values indicate drawdowns)
    """
    if len(returns) == 0:
        return pd.Series(dtype=float)

    # Calculate cumulative returns
    cumulative = (1 + returns).cumprod()

    # Calculate running maximum
    running_max = cumulative.expanding().max()

    # Calculate drawdown
    drawdown = (cumulative - running_max) / running_max

    return drawdown


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int,
    periods_per_year: Optional[int] = None,
    annualized: bool = True,
) -> pd.Series:
    """Calculate rolling volatility.

    Args:
        returns: Series of periodic returns
        window: Rolling window size in periods
        periods_per_year: Number of periods per year
        annualized: If True, annualize the volatility

    Returns:
        Series of rolling volatility values
    """
    rolling_vol = returns.rolling(window=window).std()

    if annualized:
        if periods_per_year is None:
            config = get_config()
            periods_per_year = config.analysis.get("periods_per_year", 12)
        rolling_vol = rolling_vol * np.sqrt(periods_per_year)

    return rolling_vol


def calculate_rolling_var(
    returns: pd.Series, window: int, confidence_level: float = 0.95
) -> pd.Series:
    """Calculate rolling VaR.

    Args:
        returns: Series of periodic returns
        window: Rolling window size in periods
        confidence_level: Confidence level

    Returns:
        Series of rolling VaR values
    """
    rolling_var = returns.rolling(window=window).apply(
        lambda x: calculate_var(x, confidence_level=confidence_level), raw=False
    )
    return rolling_var


def calculate_standard_deviation(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    annualized: bool = False,
) -> float:
    """Calculate standard deviation of returns.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year
        annualized: If True, annualize the standard deviation

    Returns:
        Standard deviation as decimal
    """
    return calculate_volatility(returns, periods_per_year=periods_per_year, annualized=annualized)


def calculate_semi_deviation(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    annualized: bool = True,
    threshold: Optional[float] = None,
) -> float:
    """Calculate semi-deviation (downside deviation below mean or threshold).

    Semi-deviation is the standard deviation of returns below the mean (or threshold).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year
        annualized: If True, annualize the deviation
        threshold: Threshold for semi-deviation. If None, uses mean return.

    Returns:
        Semi-deviation as decimal
    """
    if len(returns) == 0:
        return np.nan

    if threshold is None:
        threshold = returns.mean()

    downside_returns = returns[returns < threshold] - threshold

    if len(downside_returns) == 0:
        return 0.0

    semi_std = downside_returns.std()

    if annualized:
        if periods_per_year is None:
            config = get_config()
            periods_per_year = config.analysis.get("periods_per_year", 12)
        semi_std = semi_std * np.sqrt(periods_per_year)

    return semi_std

