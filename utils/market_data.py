from pathlib import Path
import pandas as pd
import yfinance as yf
from utils.historical_data import download_historical

HISTORICAL_FOLDER = Path("historical_data")

def get_market_cap(ticker):
    """
    Retrieves market capitalization from yfinance safely.
    """
    try:
        info = yf.Ticker(ticker).info
        if not info:
            print(f"⚠️ [market_data.py] No info returned for {ticker}")
            return 0
        market_cap = info.get("marketCap", 0)
        if isinstance(market_cap, pd.Series):
            market_cap = market_cap.iloc[-1]
        return float(market_cap or 0)
    except Exception as e:
        print(f"⚠️ [market_data.py] Error getting market cap for {ticker}: {e}")
        return 0


def get_historical_data(ticker):
    """
    Loads cached historical data for a ticker if available; downloads if missing.
    """
    try:
        file_path = HISTORICAL_FOLDER / f"{ticker}.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close'])
            return df
        else:
            return download_historical(ticker)
    except Exception as e:
        print(f"⚠️ [market_data.py] Error loading historical for {ticker}: {e}")
        return pd.DataFrame()
