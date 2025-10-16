from pathlib import Path
import pandas as pd
from .historical_data import download_historical

HISTORICAL_FOLDER = Path("historical_data")

def get_market_cap(ticker):
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        if info is None:
            print(f"⚠️ [market_data.py] No info returned for {ticker}")
            return 0
        return info.get("marketCap", 0) or 0
    except Exception as e:
        print(f"⚠️ [market_data.py] Error getting market cap for {ticker}: {e}")
        return 0

def get_historical_data(ticker):
    """
    Wrapper to get cached historical data.
    """
    file_path = HISTORICAL_FOLDER / f"{ticker}.csv"
    if file_path.exists():
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Ensure Close is numeric
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        return df.dropna(subset=['Close'])
    else:
        return download_historical(ticker)
