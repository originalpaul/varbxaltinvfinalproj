# VARBX Due Diligence Repository

Institutional-grade due diligence analysis for the First Trust Merger Arbitrage Fund (VARBX). This repository provides comprehensive performance metrics, risk analysis, rolling analytics, and visualizations benchmarked against the S&P 500 and Bloomberg US Aggregate Bond Index.

## Project Structure

```
varbx_dd/
├── data/
│   ├── raw/              # Raw data files (CSV)
│   └── interim/          # Processed data
├── notebooks/            # Jupyter notebooks for analysis
│   ├── 01_clean_data.ipynb
│   ├── 02_performance_metrics.ipynb
│   ├── 03_risk_analysis.ipynb
│   ├── 04_visualizations.ipynb
│   └── 05_report_export.ipynb
├── outputs/
│   ├── figures/          # Exported charts
│   └── tables/           # Exported tables
├── src/                  # Source code
│   ├── analytics/        # Performance and risk metrics
│   ├── data/             # Data loading and preprocessing
│   ├── utils/            # Utility functions
│   └── viz/              # Visualization functions
├── tests/                # Unit tests
├── config.yml            # Configuration file
├── pyproject.toml        # Project metadata
├── requirements.txt      # Python dependencies
└── Makefile             # Build automation
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd varbxaltinvfinalproj
```

2. Install dependencies:
```bash
make install
# or
pip install -r requirements.txt
```

3. Set up pre-commit hooks (optional but recommended):
```bash
make setup
```

## Configuration

Edit `config.yml` to configure:
- Data file paths and sources
- Benchmark tickers (if using yfinance)
- Analysis parameters (risk-free rate, rolling windows, etc.)
- Visualization settings
- Export formats

## Data Requirements

### CSV Format

Place your data files in `data/raw/` with the following structure:

**VARBX Returns** (`varbx_monthly_returns.csv`):
```csv
date,return
2020-01-31,0.0123
2020-02-29,0.0045
...
```

**S&P 500 Returns** (`sp500_monthly.csv`):
```csv
date,return
2020-01-31,0.0156
2020-02-29,-0.0023
...
```

**Bloomberg US Aggregate** (`agg_monthly.csv`):
```csv
date,return
2020-01-31,0.0034
2020-02-29,0.0021
...
```

### Using yfinance (Alternative)

To download benchmark data automatically, set `use_yfinance: true` in `config.yml` and run:
```bash
make data
```

This will download SPY (S&P 500 proxy) and AGG (Bloomberg US Aggregate proxy) data.

## Usage

### Running the Analysis

Execute notebooks in sequence:

1. **Data Cleaning** (`01_clean_data.ipynb`):
   - Loads and validates data
   - Aligns timeframes
   - Saves cleaned data to `data/interim/`

2. **Performance Metrics** (`02_performance_metrics.ipynb`):
   - Calculates CAGR, Sharpe ratio, Calmar ratio, max drawdown
   - Computes rolling metrics

3. **Risk Analysis** (`03_risk_analysis.ipynb`):
   - Calculates volatility, VaR, CVaR
   - Analyzes drawdowns

4. **Visualizations** (`04_visualizations.ipynb`):
   - Generates charts and plots

5. **Report Export** (`05_report_export.ipynb`):
   - Exports tables and figures to `outputs/`

### Makefile Commands

```bash
make install      # Install dependencies
make setup        # Set up pre-commit hooks
make test         # Run tests with coverage
make lint         # Run linting (ruff, mypy)
make format       # Format code (black, isort)
make clean        # Remove generated files
make data         # Download benchmark data
make all          # Run full pipeline (format, lint, test)
```

## Performance Metrics

### Compound Annual Growth Rate (CAGR)

The CAGR measures the mean annual growth rate of an investment:

\[
\text{CAGR} = \left(\frac{V_{\text{end}}}{V_{\text{start}}}\right)^{\frac{1}{n}} - 1
\]

Where:
- \(V_{\text{end}}\) = Ending value
- \(V_{\text{start}}\) = Starting value
- \(n\) = Number of years

For periodic returns:

\[
\text{CAGR} = \left(\prod_{i=1}^{n}(1 + r_i)\right)^{\frac{1}{n/p}} - 1
\]

Where:
- \(r_i\) = Periodic return
- \(n\) = Number of periods
- \(p\) = Periods per year

### Sharpe Ratio

The Sharpe ratio measures risk-adjusted return:

\[
\text{Sharpe} = \frac{\bar{R} - R_f}{\sigma_R} \times \sqrt{p}
\]

Where:
- \(\bar{R}\) = Mean periodic return
- \(R_f\) = Risk-free rate (periodic)
- \(\sigma_R\) = Standard deviation of returns
- \(p\) = Periods per year (for annualization)

### Calmar Ratio

The Calmar ratio compares CAGR to maximum drawdown:

\[
\text{Calmar} = \frac{\text{CAGR}}{|\text{Max Drawdown}|}
\]

### Maximum Drawdown

Maximum drawdown is the largest peak-to-trough decline:

\[
\text{Max DD} = \min_{t} \frac{V_t - \max_{s \leq t} V_s}{\max_{s \leq t} V_s}
\]

Where \(V_t\) is the cumulative value at time \(t\).

### Alpha and Beta

Alpha and beta are calculated using OLS regression:

\[
R_i - R_f = \alpha + \beta (R_m - R_f) + \epsilon
\]

Where:
- \(R_i\) = Asset return
- \(R_m\) = Market (benchmark) return
- \(R_f\) = Risk-free rate
- \(\alpha\) = Alpha (excess return)
- \(\beta\) = Beta (sensitivity to market)

## Risk Metrics

### Volatility

Annualized volatility:

\[
\sigma_{\text{annual}} = \sigma_{\text{periodic}} \times \sqrt{p}
\]

### Value at Risk (VaR)

VaR at confidence level \(\alpha\):

\[
\text{VaR}_\alpha = \text{Percentile}(R, 1 - \alpha)
\]

### Conditional VaR (CVaR)

Expected shortfall:

\[
\text{CVaR}_\alpha = E[R | R \leq \text{VaR}_\alpha]
\]

### Downside Deviation

Standard deviation of returns below a threshold:

\[
\sigma_{\text{downside}} = \sqrt{\frac{1}{n} \sum_{r_i < \tau} (r_i - \tau)^2}
\]

Where \(\tau\) is the threshold (typically 0).

## Testing

Run tests with coverage:

```bash
make test
```

The test suite targets ≥90% coverage for the `src/` directory.

## Code Quality

This project uses:
- **ruff** for linting
- **black** for code formatting
- **isort** for import sorting
- **mypy** for type checking (loose mode)
- **pre-commit** hooks for automated checks

Run all quality checks:

```bash
make lint      # Linting
make format    # Formatting
```

## Reproducibility

- All dependencies are pinned in `requirements.txt`
- Configuration is centralized in `config.yml`
- Random seeds are set where applicable
- Data processing is deterministic

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `make all`
5. Submit a pull request

## References

- First Trust Merger Arbitrage Fund (VARBX) documentation
- S&P 500 Index
- Bloomberg US Aggregate Bond Index

