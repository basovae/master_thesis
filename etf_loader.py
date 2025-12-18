"""
Simple Portfolio Data Retrieval - 50+ ETFs, 20+ Years
Outputs: multiasset_daily_returns.csv
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

ASSETS = {
    # US Equity - Sectors
    'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare',
    'XLE': 'Energy', 'XLI': 'Industrials', 'XLP': 'Staples',
    'XLY': 'Discretionary', 'XLU': 'Utilities', 'XLRE': 'Real Estate',
    'XLB': 'Materials', 'XLC': 'Communication',
    # US Equity - Size/Style
    'SPY': 'Large Cap', 'IWM': 'Small Cap', 'MDY': 'Mid Cap',
    'VTV': 'Value', 'VUG': 'Growth', 'MTUM': 'Momentum',
    'QUAL': 'Quality', 'SIZE': 'Size Factor', 'USMV': 'Min Vol',
    # International
    'EFA': 'Developed ex-US', 'EEM': 'Emerging Markets',
    'VGK': 'Europe', 'EWJ': 'Japan', 'FXI': 'China',
    'EWY': 'South Korea', 'INDA': 'India', 'EWZ': 'Brazil',
    'EWG': 'Germany', 'EWU': 'UK',
    # Fixed Income
    'AGG': 'Total Bond', 'TLT': 'Long Treasury', 'IEF': 'Intermediate Treasury',
    'SHY': 'Short Treasury', 'LQD': 'IG Corporate', 'HYG': 'High Yield',
    'MUB': 'Municipal', 'TIP': 'TIPS', 'EMB': 'EM Bonds',
    'BND': 'Total Bond Market',
    # Alternatives
    'GLD': 'Gold', 'SLV': 'Silver', 'VNQ': 'REITs',
    'DBA': 'Agriculture', 'GSG': 'Commodities',
    'IAU': 'Gold Alt', 'GDX': 'Gold Miners',
    # Additional
    'VOO': 'Vanguard S&P 500', 'QQQ': 'Nasdaq 100',
    'SCHD': 'US Dividend', 'VXUS': 'Total International', 'VWO': 'Emerging Markets',
}

def main():
    tickers = list(ASSETS.keys())
    end_date = datetime.now()
    start_date = end_date - timedelta(days=20*365)

    print("Downloading data...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=True, group_by='ticker')

    # Extract close prices
    prices = {}
    min_days = 20 * 252 * 0.8

    for ticker in tickers:
        try:
            close_prices = data[ticker]['Close'] if len(tickers) > 1 else data['Close']
            valid_data = close_prices.dropna()
            if len(valid_data) >= min_days:
                prices[ticker] = valid_data
        except Exception:
            pass

    if not prices:
        print("No valid assets downloaded")
        return

    # Create returns
    prices_df = pd.DataFrame(prices).dropna()
    returns = prices_df.pct_change().dropna().clip(lower=-0.5, upper=0.5)

    # Save
    returns.to_csv("data.csv")
    print(f"Saved: data.csv ({returns.shape[0]} days, {returns.shape[1]} assets)")

if __name__ == "__main__":
    main()