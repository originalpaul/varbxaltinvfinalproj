"""Tests for risk metrics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.risk import (
    calculate_cvar,
    calculate_downside_deviation,
    calculate_drawdown_series,
    calculate_rolling_var,
    calculate_rolling_volatility,
    calculate_var,
    calculate_volatility,
)


def test_calculate_volatility(sample_returns):
    """Test volatility calculation."""
    vol = calculate_volatility(sample_returns, periods_per_year=12, annualized=True)
    assert isinstance(vol, float)
    assert vol > 0
    assert not np.isnan(vol)


def test_calculate_var(sample_returns):
    """Test VaR calculation."""
    var = calculate_var(sample_returns, confidence_level=0.95, method="historical")
    assert isinstance(var, float)
    assert not np.isnan(var)


def test_calculate_cvar(sample_returns):
    """Test CVaR calculation."""
    cvar = calculate_cvar(sample_returns, confidence_level=0.95)
    assert isinstance(cvar, float)
    assert not np.isnan(cvar)
    # CVaR should be more negative than VaR
    var = calculate_var(sample_returns, confidence_level=0.95)
    assert cvar <= var


def test_calculate_downside_deviation(sample_returns):
    """Test downside deviation calculation."""
    dd = calculate_downside_deviation(sample_returns, periods_per_year=12, annualized=True)
    assert isinstance(dd, float)
    assert dd >= 0
    assert not np.isnan(dd)


def test_calculate_drawdown_series(sample_returns):
    """Test drawdown series calculation."""
    drawdown = calculate_drawdown_series(sample_returns)
    assert isinstance(drawdown, pd.Series)
    assert len(drawdown) == len(sample_returns)
    assert drawdown.max() <= 0  # Drawdowns should be negative or zero


def test_calculate_rolling_volatility(sample_returns):
    """Test rolling volatility calculation."""
    window = 12
    rolling_vol = calculate_rolling_volatility(
        sample_returns, window=window, periods_per_year=12, annualized=True
    )
    assert isinstance(rolling_vol, pd.Series)
    assert len(rolling_vol) == len(sample_returns)


def test_calculate_rolling_var(sample_returns):
    """Test rolling VaR calculation."""
    window = 12
    rolling_var = calculate_rolling_var(sample_returns, window=window, confidence_level=0.95)
    assert isinstance(rolling_var, pd.Series)
    assert len(rolling_var) == len(sample_returns)


def test_empty_series():
    """Test that empty series return NaN."""
    empty = pd.Series(dtype=float)
    assert np.isnan(calculate_volatility(empty))
    assert np.isnan(calculate_var(empty))
    assert np.isnan(calculate_cvar(empty))

