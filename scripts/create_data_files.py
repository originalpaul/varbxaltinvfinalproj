#!/usr/bin/env python3
"""Create data files from PDFs and yfinance downloads."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pandas as pd
    import yfinance as yf
    import pdfplumber
except ImportError as e:
    print(f"Error: Missing required library. Please install: pip install pandas yfinance pdfplumber")
    print(f"Missing: {e.name}")
    sys.exit(1)

import re
from datetime import datetime


def extract_returns_from_pdf_text(pdf_path: Path) -> list:
    """Extract return data from PDF text."""
    returns = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            
            # Look for various date/return patterns
            # Pattern 1: "Month Year: X.XX%"
            pattern1 = r'(\w+)\s+(\d{4}):\s*(-?\d+\.?\d*)\s*%'
            # Pattern 2: "MM/YYYY: X.XX%"
            pattern2 = r'(\d{1,2})/(\d{4}):\s*(-?\d+\.?\d*)\s*%'
            # Pattern 3: "YYYY-MM: X.XX%"
            pattern3 = r'(\d{4})-(\d{2}):\s*(-?\d+\.?\d*)\s*%'
            
            month_map = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2,
                'march': 3, 'mar': 3, 'april': 4, 'apr': 4,
                'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
                'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
                'october': 10, 'oct': 10, 'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            
            for pattern in [pattern1, pattern2, pattern3]:
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    try:
                        g1, g2, g3 = match.groups()
                        
                        # Parse month and year
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
                        
                        if 2000 <= year <= 2030 and 1 <= month <= 12:
                            return_val = float(g3) / 100.0
                            date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                            returns.append({'date': date, 'return': return_val})
                    except:
                        continue
            
            # Also try extracting tables
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    for row in table:
                        if not row or len(row) < 2:
                            continue
                        # Look for date and return in row
                        row_str = ' '.join([str(c) if c else '' for c in row])
                        date_match = re.search(r'(\d{4})[-\s/](\d{1,2})', row_str)
                        return_match = re.search(r'(-?\d+\.?\d*)\s*%', row_str)
                        if date_match and return_match:
                            try:
                                year = int(date_match.group(1))
                                month = int(date_match.group(2))
                                return_val = float(return_match.group(1)) / 100.0
                                if 2000 <= year <= 2030 and 1 <= month <= 12:
                                    date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                                    returns.append({'date': date, 'return': return_val})
                            except:
                                pass
    except Exception as e:
        print(f"  Error reading {pdf_path.name}: {e}")
    
    return returns


def download_varbx_data() -> pd.DataFrame:
    """Try to download VARBX data from yfinance."""
    try:
        ticker = yf.Ticker("VARBX")
        hist = ticker.history(start="2020-01-01")
        if not hist.empty:
            hist_monthly = hist.resample("ME").last()
            hist_monthly["return"] = hist_monthly["Close"].pct_change()
            hist_monthly = hist_monthly.dropna()
            hist_monthly = hist_monthly.reset_index()
            hist_monthly = hist_monthly.rename(columns={"Date": "date"})
            return hist_monthly[["date", "return"]].copy()
    except:
        pass
    return pd.DataFrame(columns=['date', 'return'])


def main():
    """Main function."""
    project_root = Path(__file__).parent.parent
    data_raw_path = project_root / "data" / "raw"
    data_raw_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating data files...")
    print("=" * 60)
    
    # Extract VARBX data from PDFs
    print("\n1. Extracting VARBX data from PDFs...")
    pdf_files = [
        project_root / "Annual-Class-A_First-Trust-Merger-Arbitrage-Fund_9.30.24.pdf",
        project_root / "Annual-Class-C_First-Trust-Merger-Arbitrage-Fund_9.30.24.pdf",
        project_root / "Semi-Annual-Class-I_First-Trust-Merger-Arbitrage-Fund_3.31.25.pdf",
        project_root / "Annual-Financials-and-Other-Information-First-Trust-Funds_9-30-24_Final (1).pdf",
    ]
    
    all_returns = []
    for pdf_file in pdf_files:
        if pdf_file.exists():
            print(f"  Processing {pdf_file.name}...")
            returns = extract_returns_from_pdf_text(pdf_file)
            if returns:
                all_returns.extend(returns)
                print(f"    Found {len(returns)} returns")
    
    # Try yfinance as fallback
    if not all_returns:
        print("  Trying yfinance for VARBX...")
        varbx_df = download_varbx_data()
        if not varbx_df.empty:
            all_returns = varbx_df.to_dict('records')
            print(f"    Downloaded {len(all_returns)} returns from yfinance")
    
    # Create VARBX DataFrame
    if all_returns:
        varbx_df = pd.DataFrame(all_returns)
        varbx_df = varbx_df.drop_duplicates(subset=['date'])
        varbx_df = varbx_df.sort_values('date').reset_index(drop=True)
    else:
        print("  Warning: Could not extract VARBX data. Creating placeholder.")
        # Create placeholder with recent dates
        dates = pd.date_range("2020-01-31", periods=60, freq="M")
        varbx_df = pd.DataFrame({
            'date': dates,
            'return': [0.005] * 60  # Placeholder 0.5% monthly return
        })
        print("  Created placeholder data - please replace with actual VARBX returns")
    
    # Save VARBX data
    # Ensure date is in proper format (date only, no timezone)
    if 'date' in varbx_df.columns:
        varbx_df['date'] = pd.to_datetime(varbx_df['date']).dt.date
    varbx_file = data_raw_path / "varbx_monthly_returns.csv"
    varbx_df.to_csv(varbx_file, index=False)
    print(f"\n  Saved: {varbx_file} ({len(varbx_df)} rows)")
    min_date = varbx_df['date'].min()
    max_date = varbx_df['date'].max()
    if isinstance(min_date, pd.Timestamp):
        min_date = min_date.date()
    if isinstance(max_date, pd.Timestamp):
        max_date = max_date.date()
    print(f"  Date range: {min_date} to {max_date}")
    
    # Download benchmark data
    print("\n2. Downloading S&P 500 (SPY)...")
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(start="2020-01-01")
        hist_monthly = hist.resample("ME").last()
        hist_monthly["return"] = hist_monthly["Close"].pct_change()
        hist_monthly = hist_monthly.dropna()
        hist_monthly = hist_monthly.reset_index()
        hist_monthly = hist_monthly.rename(columns={"Date": "date"})
        sp500_df = hist_monthly[["date", "return"]].copy()
        # Convert date to date only (no timezone)
        sp500_df['date'] = pd.to_datetime(sp500_df['date']).dt.date
        
        sp500_file = data_raw_path / "sp500_monthly.csv"
        sp500_df.to_csv(sp500_file, index=False)
        print(f"  Saved: {sp500_file} ({len(sp500_df)} rows)")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n3. Downloading Bloomberg US Aggregate (AGG)...")
    try:
        agg = yf.Ticker("AGG")
        hist = agg.history(start="2020-01-01")
        hist_monthly = hist.resample("ME").last()
        hist_monthly["return"] = hist_monthly["Close"].pct_change()
        hist_monthly = hist_monthly.dropna()
        hist_monthly = hist_monthly.reset_index()
        hist_monthly = hist_monthly.rename(columns={"Date": "date"})
        agg_df = hist_monthly[["date", "return"]].copy()
        # Convert date to date only (no timezone)
        agg_df['date'] = pd.to_datetime(agg_df['date']).dt.date
        
        agg_file = data_raw_path / "agg_monthly.csv"
        agg_df.to_csv(agg_file, index=False)
        print(f"  Saved: {agg_file} ({len(agg_df)} rows)")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Data file creation complete!")
    print(f"\nFiles created in: {data_raw_path}")


if __name__ == "__main__":
    main()

