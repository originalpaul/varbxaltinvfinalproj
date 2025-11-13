"""Pytest fixtures for testing."""

import pandas as pd
import pytest
import numpy as np


@pytest.fixture
def sample_returns() -> pd.Series:
    """Create sample returns series for testing."""
    dates = pd.date_range("2020-01-01", periods=60, freq="M")
    # Generate returns with some volatility
    np.random.seed(42)
    returns = pd.Series(
        np.random.normal(0.01, 0.02, 60),  # Mean 1%, std 2% monthly
        index=dates,
        name="return",
    )
    return returns


@pytest.fixture
def sample_returns_positive() -> pd.Series:
    """Create sample returns series with positive trend."""
    dates = pd.date_range("2020-01-01", periods=60, freq="M")
    # Generate returns with positive trend
    np.random.seed(42)
    trend = np.linspace(0.005, 0.015, 60)
    returns = pd.Series(
        trend + np.random.normal(0, 0.01, 60),
        index=dates,
        name="return",
    )
    return returns


@pytest.fixture
def sample_returns_dataframe() -> pd.DataFrame:
    """Create sample returns DataFrame for testing."""
    dates = pd.date_range("2020-01-01", periods=60, freq="M")
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "return": np.random.normal(0.01, 0.02, 60),
        }
    )
    return df


@pytest.fixture
def sample_benchmark_returns() -> pd.Series:
    """Create sample benchmark returns series."""
    dates = pd.date_range("2020-01-01", periods=60, freq="M")
    np.random.seed(43)
    returns = pd.Series(
        np.random.normal(0.008, 0.025, 60),  # Slightly lower mean, higher vol
        index=dates,
        name="benchmark",
    )
    return returns


@pytest.fixture
def sample_returns_multiple() -> pd.DataFrame:
    """Create sample DataFrame with multiple return series."""
    dates = pd.date_range("2020-01-01", periods=60, freq="M")
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "varbx": np.random.normal(0.01, 0.02, 60),
            "sp500": np.random.normal(0.008, 0.025, 60),
            "agg": np.random.normal(0.003, 0.01, 60),
        }
    )
    return df

