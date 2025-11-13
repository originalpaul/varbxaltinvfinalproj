"""create the files and shi from pdfs"""

import re
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber
import yfinance as yf


def extract_varbx_returns_from_pdf(pdf_path: Path) -> pd.DataFrame:

    returns_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
                
            # Look for tables
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                    
                # Search for return data patterns
                for row in table:
                    if not row:
                        continue
                    row_text = ' '.join([str(cell) if cell else '' for cell in row])
                    
                    # Look for date patterns and return values
                    # Pattern: YYYY-MM or MM/YYYY or similar
                    date_match = re.search(r'(\d{4})[-\s/](\d{1,2})', row_text)
                    if date_match:
                        year = int(date_match.group(1))
                        month = int(date_match.group(2))
                        
                        # Look for return percentage in the row
                        # Pattern: -X.XX% or X.XX% or (X.XX)%
                        return_match = re.search(r'[\(]?(-?\d+\.?\d*)\s*%', row_text)
                        if return_match:
                            return_val = float(return_match.group(1)) / 100.0
                            # Create date (last day of month)
                            try:
                                date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                                returns_data.append({'date': date, 'return': return_val})
                            except:
                                pass
    
    if not returns_data:
        # Try alternative: extract from text patterns
        with pdfplumber.open(pdf_path) as pdf:
            full_text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
            
            # Look for patterns like "Month Year: X.XX%"
            patterns = [
                r'(\w+)\s+(\d{4}):\s*(-?\d+\.?\d*)\s*%',
                r'(\d{1,2})/(\d{4}):\s*(-?\d+\.?\d*)\s*%',
                r'(\d{4})-(\d{2}):\s*(-?\d+\.?\d*)\s*%',
            ]
            
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
                'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
                'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
                'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
                'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            
            for pattern in patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    try:
                        if len(match.groups()) == 3:
                            g1, g2, g3 = match.groups()
                            
                            # Try to parse as month/year or year/month
                            if g1.isdigit() and len(g1) <= 2:
                                month = int(g1)
                                year = int(g2)
                            elif g2.isdigit() and len(g2) <= 2:
                                month = int(g2)
                                year = int(g1)
                            elif g1.lower() in month_map:
                                month = month_map[g1.lower()]
                                year = int(g2)
                            else:
                                continue
                                
                            return_val = float(g3) / 100.0
                            date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                            returns_data.append({'date': date, 'return': return_val})
                    except:
                        continue
    
    if returns_data:
        df = pd.DataFrame(returns_data)
        df = df.drop_duplicates(subset=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    
    return pd.DataFrame(columns=['date', 'return'])


def download_benchmark_returns(ticker: str, start_date: Optional[str] = None) -> pd.DataFrame:
    """Download benchmark returns using yfinance.
    
    Args:
        ticker: Ticker symbol (e.g., 'SPY', 'AGG')
        start_date: Start date in YYYY-MM-DD format
        
    Returns:
        DataFrame with date and return columns
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date)
    
    if hist.empty:
        raise ValueError(f"No data downloaded for ticker {ticker}")
    
    # Calculate monthly returns
    hist = hist[["Close"]].copy()
    hist_monthly = hist.resample("M").last()
    hist_monthly["return"] = hist_monthly["Close"].pct_change()
    hist_monthly = hist_monthly.dropna()
    
    # Reset index to get date as column
    hist_monthly = hist_monthly.reset_index()
    hist_monthly = hist_monthly.rename(columns={"Date": "date"})
    hist_monthly = hist_monthly[["date", "return"]].copy()
    
    return hist_monthly


def main():
    """Main function to extract all data and create CSV files."""
    project_root = Path(__file__).parent.parent
    data_raw_path = project_root / "data" / "raw"
    data_raw_path.mkdir(parents=True, exist_ok=True)
    
    print("Extracting VARBX data from PDFs...")
    
    # Try to extract from all PDF files
    pdf_files = [
        project_root / "Annual-Class-A_First-Trust-Merger-Arbitrage-Fund_9.30.24.pdf",
        project_root / "Annual-Class-C_First-Trust-Merger-Arbitrage-Fund_9.30.24.pdf",
        project_root / "Semi-Annual-Class-I_First-Trust-Merger-Arbitrage-Fund_3.31.25.pdf",
        project_root / "Annual-Financials-and-Other-Information-First-Trust-Funds_9-30-24_Final (1).pdf",
    ]
    
    all_varbx_data = []
    for pdf_file in pdf_files:
        if pdf_file.exists():
            print(f"  Processing {pdf_file.name}...")
            try:
                df = extract_varbx_returns_from_pdf(pdf_file)
                if not df.empty:
                    all_varbx_data.append(df)
                    print(f"    Found {len(df)} returns")
            except Exception as e:
                print(f"    Error: {e}")
    
    # Combine all VARBX data
    if all_varbx_data:
        varbx_df = pd.concat(all_varbx_data, ignore_index=True)
        varbx_df = varbx_df.drop_duplicates(subset=['date'])
        varbx_df = varbx_df.sort_values('date').reset_index(drop=True)
    else:
        # If extraction fails, create sample data structure
        print("  Warning: Could not extract from PDFs, creating placeholder structure")
        # Try to get VARBX data from yfinance if available
        try:
            varbx_df = download_benchmark_returns("VARBX", start_date="2020-01-01")
        except:
            # Create empty structure
            varbx_df = pd.DataFrame(columns=['date', 'return'])
    
    # Save VARBX data
    varbx_file = data_raw_path / "varbx_monthly_returns.csv"
    varbx_df.to_csv(varbx_file, index=False)
    print(f"\nSaved VARBX data to {varbx_file} ({len(varbx_df)} rows)")
    
    # Download benchmark data
    print("\nDownloading benchmark data...")
    
    print("  Downloading S&P 500 (SPY)...")
    sp500_df = download_benchmark_returns("SPY", start_date="2020-01-01")
    sp500_file = data_raw_path / "sp500_monthly.csv"
    sp500_df.to_csv(sp500_file, index=False)
    print(f"  Saved S&P 500 data to {sp500_file} ({len(sp500_df)} rows)")
    
    print("  Downloading Bloomberg US Aggregate (AGG)...")
    agg_df = download_benchmark_returns("AGG", start_date="2020-01-01")
    agg_file = data_raw_path / "agg_monthly.csv"
    agg_df.to_csv(agg_file, index=False)
    print(f"  Saved AGG data to {agg_file} ({len(agg_df)} rows)")
    
    print("\nData extraction complete!")


if __name__ == "__main__":
    main()

