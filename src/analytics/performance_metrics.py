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


def calculate_cumulative_return(returns: pd.Series) -> float:
    """Calculate cumulative return over the period.

    Args:
        returns: Series of periodic returns

    Returns:
        Cumulative return as decimal
    """
    if len(returns) == 0:
        return np.nan
    return (1 + returns).prod() - 1


def calculate_compound_monthly_return(returns: pd.Series) -> float:
    """Calculate compound monthly return (geometric mean of monthly returns).

    Args:
        returns: Series of periodic returns

    Returns:
        Compound monthly return as decimal
    """
    if len(returns) == 0:
        return np.nan
    return (1 + returns).prod() ** (1 / len(returns)) - 1


def calculate_annualized_return(
    returns: pd.Series, periods_per_year: Optional[int] = None
) -> float:
    """Calculate annualized return (same as CAGR).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year

    Returns:
        Annualized return as decimal
    """
    return calculate_cagr(returns, periods_per_year=periods_per_year)


def calculate_average_monthly_return(returns: pd.Series) -> float:
    """Calculate average monthly return (arithmetic mean).

    Args:
        returns: Series of periodic returns

    Returns:
        Average monthly return as decimal
    """
    if len(returns) == 0:
        return np.nan
    return returns.mean()


def calculate_average_monthly_gain(returns: pd.Series) -> float:
    """Calculate average monthly gain (mean of positive returns).

    Args:
        returns: Series of periodic returns

    Returns:
        Average monthly gain as decimal
    """
    if len(returns) == 0:
        return np.nan
    gains = returns[returns > 0]
    if len(gains) == 0:
        return 0.0
    return gains.mean()


def calculate_average_monthly_loss(returns: pd.Series) -> float:
    """Calculate average monthly loss (mean of negative returns).

    Args:
        returns: Series of periodic returns

    Returns:
        Average monthly loss as decimal
    """
    if len(returns) == 0:
        return np.nan
    losses = returns[returns < 0]
    if len(losses) == 0:
        return 0.0
    return losses.mean()


def calculate_pct_up_months(returns: pd.Series) -> float:
    """Calculate percentage of months with positive returns.

    Args:
        returns: Series of periodic returns

    Returns:
        Percentage as decimal (e.g., 0.60 for 60%)
    """
    if len(returns) == 0:
        return np.nan
    return (returns > 0).sum() / len(returns)


def calculate_pct_down_months(returns: pd.Series) -> float:
    """Calculate percentage of months with negative returns.

    Args:
        returns: Series of periodic returns

    Returns:
        Percentage as decimal (e.g., 0.40 for 40%)
    """
    if len(returns) == 0:
        return np.nan
    return (returns < 0).sum() / len(returns)


def calculate_highest_monthly_performance(returns: pd.Series) -> float:
    """Calculate highest monthly return.

    Args:
        returns: Series of periodic returns

    Returns:
        Highest monthly return as decimal
    """
    if len(returns) == 0:
        return np.nan
    return returns.max()


def calculate_lowest_monthly_performance(returns: pd.Series) -> float:
    """Calculate lowest monthly return.

    Args:
        returns: Series of periodic returns

    Returns:
        Lowest monthly return as decimal
    """
    if len(returns) == 0:
        return np.nan
    return returns.min()


def calculate_gain_std_dev(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    annualized: bool = False,
) -> float:
    """Calculate standard deviation of gains (positive returns only).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year
        annualized: If True, annualize the standard deviation

    Returns:
        Gain standard deviation as decimal
    """
    if len(returns) == 0:
        return np.nan
    gains = returns[returns > 0]
    if len(gains) == 0:
        return 0.0
    std_dev = gains.std()
    if annualized:
        if periods_per_year is None:
            config = get_config()
            periods_per_year = config.analysis.get("periods_per_year", 12)
        std_dev = std_dev * np.sqrt(periods_per_year)
    return std_dev


def calculate_loss_std_dev(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    annualized: bool = False,
) -> float:
    """Calculate standard deviation of losses (negative returns only).

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year
        annualized: If True, annualize the standard deviation

    Returns:
        Loss standard deviation as decimal
    """
    if len(returns) == 0:
        return np.nan
    losses = returns[returns < 0]
    if len(losses) == 0:
        return 0.0
    std_dev = losses.std()
    if annualized:
        if periods_per_year is None:
            config = get_config()
            periods_per_year = config.analysis.get("periods_per_year", 12)
        std_dev = std_dev * np.sqrt(periods_per_year)
    return std_dev


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    periods_per_year: Optional[int] = None,
    mar: float = 0.0,
    annualized: bool = True,
) -> float:
    """Calculate Sortino ratio.

    Formula: Sortino = (Mean Return - MAR) / Downside Deviation
    Uses downside deviation below MAR threshold.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate as decimal (for RFR variant)
        periods_per_year: Number of periods per year
        mar: Minimum Acceptable Return threshold (default 0.0)
        annualized: If True, return annualized ratio

    Returns:
        Sortino ratio
    """
    if periods_per_year is None:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)

    if risk_free_rate is not None:
        # For RFR variant, use risk-free rate as threshold
        periodic_rf = risk_free_rate / periods_per_year
        threshold = periodic_rf
    else:
        # For MAR variant, use MAR threshold
        threshold = mar

    if len(returns) == 0:
        return np.nan

    # Calculate excess returns over threshold
    excess_returns = returns - threshold
    mean_excess = excess_returns.mean()

    # Calculate downside deviation
    from src.analytics.risk import calculate_downside_deviation

    downside_dev = calculate_downside_deviation(
        returns, periods_per_year=periods_per_year, annualized=annualized, threshold=threshold
    )

    if downside_dev == 0 or np.isnan(downside_dev):
        return np.nan

    if annualized:
        # Annualize mean excess return
        mean_excess = mean_excess * periods_per_year

    sortino = mean_excess / downside_dev

    return sortino


def calculate_sharpe_ratio_mar(
    returns: pd.Series,
    periods_per_year: Optional[int] = None,
    mar: float = 0.0,
    annualized: bool = True,
) -> float:
    """Calculate Sharpe ratio using MAR (Minimum Acceptable Return) instead of RFR.

    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year
        mar: Minimum Acceptable Return threshold (default 0.0)
        annualized: If True, return annualized ratio

    Returns:
        Sharpe ratio (MAR-based)
    """
    return calculate_sortino_ratio(
        returns,
        risk_free_rate=None,
        periods_per_year=periods_per_year,
        mar=mar,
        annualized=annualized,
    )


def calculate_treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    periods_per_year: Optional[int] = None,
) -> float:
    """Calculate Treynor ratio.

    Formula: Treynor = (Mean Return - Rf) / Beta

    Args:
        returns: Series of asset returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Annual risk-free rate as decimal
        periods_per_year: Number of periods per year

    Returns:
        Treynor ratio
    """
    if periods_per_year is None:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)

    if risk_free_rate is None:
        config = get_config()
        risk_free_rate = config.analysis.get("risk_free_rate", 0.02)

    if len(returns) == 0:
        return np.nan

    # Calculate beta
    from src.analytics.regression import calculate_alpha_beta

    regression_result = calculate_alpha_beta(
        returns, benchmark_returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
    )
    beta = regression_result["beta"]

    if beta == 0 or np.isnan(beta):
        return np.nan

    # Convert annual risk-free rate to periodic
    periodic_rf = risk_free_rate / periods_per_year

    # Calculate excess return (annualized)
    excess_return = (returns.mean() - periodic_rf) * periods_per_year

    treynor = excess_return / beta

    return treynor


def calculate_active_premium(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: Optional[int] = None,
) -> float:
    """Calculate active premium (excess return over benchmark).

    Args:
        returns: Series of asset returns
        benchmark_returns: Series of benchmark returns
        periods_per_year: Number of periods per year

    Returns:
        Active premium as decimal (annualized)
    """
    if periods_per_year is None:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)

    # Align series
    aligned = pd.DataFrame({"returns": returns, "benchmark": benchmark_returns}).dropna()

    if len(aligned) < 2:
        return np.nan

    # Calculate annualized returns
    asset_cagr = calculate_cagr(aligned["returns"], periods_per_year=periods_per_year)
    bench_cagr = calculate_cagr(aligned["benchmark"], periods_per_year=periods_per_year)

    active_premium = asset_cagr - bench_cagr

    return active_premium


def calculate_skewness(returns: pd.Series) -> float:
    """Calculate skewness of returns.

    Args:
        returns: Series of periodic returns

    Returns:
        Skewness
    """
    if len(returns) == 0:
        return np.nan
    return returns.skew()


def calculate_kurtosis(returns: pd.Series) -> float:
    """Calculate kurtosis of returns.

    Args:
        returns: Series of periodic returns

    Returns:
        Kurtosis
    """
    if len(returns) == 0:
        return np.nan
    return returns.kurtosis()


def calculate_all_metrics(
    returns: pd.Series,
    risk_free_rate: Optional[float] = None,
    periods_per_year: Optional[int] = None,
    benchmark_returns: Optional[pd.Series] = None,
) -> dict[str, float]:
    """Calculate all performance metrics.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate as decimal
        periods_per_year: Number of periods per year
        benchmark_returns: Optional benchmark returns for relative metrics

    Returns:
        Dictionary with all performance metrics
    """
    if periods_per_year is None:
        config = get_config()
        periods_per_year = config.analysis.get("periods_per_year", 12)

    if risk_free_rate is None:
        config = get_config()
        risk_free_rate = config.analysis.get("risk_free_rate", 0.02)

    metrics = {
        "cagr": calculate_cagr(returns, periods_per_year=periods_per_year),
        "total_return": calculate_total_return(returns),
        "cumulative_return": calculate_cumulative_return(returns),
        "compound_monthly_return": calculate_compound_monthly_return(returns),
        "annualized_return": calculate_annualized_return(returns, periods_per_year=periods_per_year),
        "average_monthly_return": calculate_average_monthly_return(returns),
        "average_monthly_gain": calculate_average_monthly_gain(returns),
        "average_monthly_loss": calculate_average_monthly_loss(returns),
        "pct_up_months": calculate_pct_up_months(returns),
        "pct_down_months": calculate_pct_down_months(returns),
        "highest_monthly_performance": calculate_highest_monthly_performance(returns),
        "lowest_monthly_performance": calculate_lowest_monthly_performance(returns),
        "max_drawdown": calculate_max_drawdown(returns),
        "skewness": calculate_skewness(returns),
        "kurtosis": calculate_kurtosis(returns),
    }

    # Add standard deviation metrics
    from src.analytics.risk import calculate_volatility

    metrics["standard_deviation"] = calculate_volatility(
        returns, periods_per_year=periods_per_year, annualized=False
    )
    metrics["annualized_standard_deviation"] = calculate_volatility(
        returns, periods_per_year=periods_per_year, annualized=True
    )

    # Add gain/loss standard deviation
    metrics["gain_std_dev"] = calculate_gain_std_dev(
        returns, periods_per_year=periods_per_year, annualized=False
    )
    metrics["annualized_gain_std_dev"] = calculate_gain_std_dev(
        returns, periods_per_year=periods_per_year, annualized=True
    )
    metrics["loss_std_dev"] = calculate_loss_std_dev(
        returns, periods_per_year=periods_per_year, annualized=False
    )
    metrics["annualized_loss_std_dev"] = calculate_loss_std_dev(
        returns, periods_per_year=periods_per_year, annualized=True
    )

    # Add Sharpe ratios (both monthly and annualized)
    # Monthly Sharpe Ratio RFR
    periodic_rf = risk_free_rate / periods_per_year
    excess_returns_rfr = returns - periodic_rf
    mean_excess_rfr = excess_returns_rfr.mean()
    std_dev_rfr = excess_returns_rfr.std()
    if std_dev_rfr != 0 and not np.isnan(std_dev_rfr):
        metrics["sharpe_ratio_rfr"] = mean_excess_rfr / std_dev_rfr
    else:
        metrics["sharpe_ratio_rfr"] = np.nan
    
    # Annualized Sharpe Ratio RFR
    metrics["annualized_sharpe_ratio_rfr"] = calculate_sharpe_ratio(
        returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
    )
    
    # Monthly Sharpe Ratio MAR
    excess_returns_mar = returns - 0.0
    mean_excess_mar = excess_returns_mar.mean()
    std_dev_mar = excess_returns_mar.std()
    if std_dev_mar != 0 and not np.isnan(std_dev_mar):
        metrics["sharpe_ratio_mar"] = mean_excess_mar / std_dev_mar
    else:
        metrics["sharpe_ratio_mar"] = np.nan
    
    # Annualized Sharpe Ratio MAR
    metrics["annualized_sharpe_ratio_mar"] = calculate_sharpe_ratio_mar(
        returns, periods_per_year=periods_per_year, mar=0.0, annualized=True
    )

    # Add Sortino ratios (both monthly and annualized)
    # Monthly Sortino Ratio RFR
    metrics["sortino_ratio_rfr"] = calculate_sortino_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        mar=0.0,
        annualized=False,
    )
    # Annualized Sortino Ratio RFR
    metrics["annualized_sortino_ratio_rfr"] = calculate_sortino_ratio(
        returns,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
        mar=0.0,
        annualized=True,
    )
    # Monthly Sortino Ratio MAR
    metrics["sortino_ratio_mar"] = calculate_sortino_ratio(
        returns,
        risk_free_rate=None,
        periods_per_year=periods_per_year,
        mar=0.0,
        annualized=False,
    )
    # Annualized Sortino Ratio MAR
    metrics["annualized_sortino_ratio_mar"] = calculate_sortino_ratio(
        returns,
        risk_free_rate=None,
        periods_per_year=periods_per_year,
        mar=0.0,
        annualized=True,
    )

    # Add downside deviation metrics
    from src.analytics.risk import calculate_downside_deviation

    periodic_rf = risk_free_rate / periods_per_year
    metrics["downside_deviation_rfr"] = calculate_downside_deviation(
        returns, periods_per_year=periods_per_year, annualized=False, threshold=periodic_rf
    )
    metrics["annualized_downside_deviation_rfr"] = calculate_downside_deviation(
        returns, periods_per_year=periods_per_year, annualized=True, threshold=periodic_rf
    )
    metrics["downside_deviation_mar"] = calculate_downside_deviation(
        returns, periods_per_year=periods_per_year, annualized=False, threshold=0.0
    )
    metrics["annualized_downside_deviation_mar"] = calculate_downside_deviation(
        returns, periods_per_year=periods_per_year, annualized=True, threshold=0.0
    )

    # Add semi deviation
    from src.analytics.risk import calculate_semi_deviation

    metrics["semi_deviation"] = calculate_semi_deviation(
        returns, periods_per_year=periods_per_year, annualized=False
    )
    metrics["annualized_semi_deviation"] = calculate_semi_deviation(
        returns, periods_per_year=periods_per_year, annualized=True
    )

    # Add relative metrics if benchmark provided
    if benchmark_returns is not None:
        from src.analytics.regression import (
            calculate_alpha_beta,
            calculate_correlation,
            calculate_information_ratio,
            calculate_tracking_error,
            calculate_up_down_capture,
        )

        regression_result = calculate_alpha_beta(
            returns, benchmark_returns, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        )
        metrics["beta"] = regression_result["beta"]
        metrics["alpha"] = regression_result["alpha"]
        metrics["alpha_annualized"] = regression_result["alpha_annualized"]
        metrics["r_squared"] = regression_result["r_squared"]

        metrics["correlation"] = calculate_correlation(returns, benchmark_returns)

        metrics["treynor_ratio"] = calculate_treynor_ratio(
            returns, benchmark_returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )

        metrics["active_premium"] = calculate_active_premium(
            returns, benchmark_returns, periods_per_year=periods_per_year
        )

        metrics["tracking_error"] = calculate_tracking_error(
            returns, benchmark_returns, annualized=True
        )

        metrics["information_ratio"] = calculate_information_ratio(
            returns, benchmark_returns, periods_per_year=periods_per_year
        )

        capture_result = calculate_up_down_capture(returns, benchmark_returns)
        metrics["up_capture"] = capture_result["up_capture"]
        metrics["down_capture"] = capture_result["down_capture"]

        # Jensen Alpha (same as alpha from regression)
        metrics["jensen_alpha"] = regression_result["alpha"]
        metrics["jensen_alpha_annualized"] = regression_result["alpha_annualized"]

    return metrics

