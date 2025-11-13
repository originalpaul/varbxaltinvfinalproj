"""Tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.performance_metrics import (
    calculate_cagr,
    calculate_calmar_ratio,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_total_return,
    calculate_rolling_cagr,
    calculate_rolling_sharpe,
)

def test_calculate_total_return(sample_returns):
    """Test total return calculation."""
    total_return = calculate_total_return(sample_returns)
    assert isinstance(total_return, float)
    assert not np.isnan(total_return)


def test_calculate_cagr(sample_returns):
    """Test CAGR calculation."""
    cagr = calculate_cagr(sample_returns, periods_per_year=12)
    assert isinstance(cagr, float)
    assert not np.isnan(cagr)

    # Test with known values
    # 12 months of 1% monthly returns should give ~12.68% CAGR
    monthly_1pct = pd.Series([0.01] * 12)
    cagr_1pct = calculate_cagr(monthly_1pct, periods_per_year=12)
    expected = (1.01 ** 12) - 1
    assert abs(cagr_1pct - expected) < 1e-10


def test_calculate_sharpe_ratio(sample_returns):
    """Test Sharpe ratio calculation."""
    sharpe = calculate_sharpe_ratio(sample_returns, risk_free_rate=0.02, periods_per_year=12)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)


def test_calculate_max_drawdown(sample_returns):
    """Test max drawdown calculation."""
    max_dd = calculate_max_drawdown(sample_returns)
    assert isinstance(max_dd, float)
    assert max_dd <= 0  # Drawdown should be negative or zero


def test_calculate_calmar_ratio(sample_returns):
    """Test Calmar ratio calculation."""
    calmar = calculate_calmar_ratio(sample_returns, periods_per_year=12)
    assert isinstance(calmar, float)


def test_calculate_rolling_cagr(sample_returns):
    """Test rolling CAGR calculation."""
    window = 12
    rolling_cagr = calculate_rolling_cagr(sample_returns, window=window, periods_per_year=12)
    assert isinstance(rolling_cagr, pd.Series)
    assert len(rolling_cagr) == len(sample_returns)
    # First window-1 values should be NaN
    assert rolling_cagr.iloc[: window - 1].isna().all()


def test_calculate_rolling_sharpe(sample_returns):
    """Test rolling Sharpe calculation."""
    window = 12
    rolling_sharpe = calculate_rolling_sharpe(
        sample_returns, window=window, risk_free_rate=0.02, periods_per_year=12
    )
    assert isinstance(rolling_sharpe, pd.Series)
    assert len(rolling_sharpe) == len(sample_returns)


def test_empty_series():
    """Test that empty series return NaN."""
    empty = pd.Series(dtype=float)
    assert np.isnan(calculate_cagr(empty))
    assert np.isnan(calculate_sharpe_ratio(empty))
    assert np.isnan(calculate_max_drawdown(empty))

