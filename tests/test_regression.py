"""Tests for regression analysis."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.regression import (
    calculate_alpha_beta,
    calculate_information_ratio,
    calculate_rolling_alpha_beta,
    calculate_tracking_error,
)


def test_calculate_alpha_beta(sample_returns, sample_benchmark_returns):
    """Test alpha and beta calculation."""
    result = calculate_alpha_beta(
        sample_returns, sample_benchmark_returns, periods_per_year=12, risk_free_rate=0.02
    )

    assert "alpha" in result
    assert "beta" in result
    assert "r_squared" in result
    assert "alpha_annualized" in result

    assert isinstance(result["alpha"], float)
    assert isinstance(result["beta"], float)
    assert isinstance(result["r_squared"], float)
    assert isinstance(result["alpha_annualized"], float)

    assert 0 <= result["r_squared"] <= 1


def test_calculate_rolling_alpha_beta(sample_returns, sample_benchmark_returns):
    """Test rolling alpha and beta calculation."""
    window = 12
    result = calculate_rolling_alpha_beta(
        sample_returns,
        sample_benchmark_returns,
        window=window,
        periods_per_year=12,
        risk_free_rate=0.02,
    )

    assert isinstance(result, pd.DataFrame)
    assert "alpha" in result.columns
    assert "beta" in result.columns
    assert "r_squared" in result.columns
    assert "alpha_annualized" in result.columns


def test_calculate_tracking_error(sample_returns, sample_benchmark_returns):
    """Test tracking error calculation."""
    te = calculate_tracking_error(sample_returns, sample_benchmark_returns, annualized=True)
    assert isinstance(te, float)
    assert te >= 0
    assert not np.isnan(te)


def test_calculate_information_ratio(sample_returns, sample_benchmark_returns):
    """Test information ratio calculation."""
    ir = calculate_information_ratio(sample_returns, sample_benchmark_returns, periods_per_year=12)
    assert isinstance(ir, float)
    assert not np.isnan(ir)


def test_empty_series():
    """Test that empty series return NaN."""
    empty = pd.Series(dtype=float)
    result = calculate_alpha_beta(empty, empty)
    assert np.isnan(result["alpha"])
    assert np.isnan(result["beta"])

