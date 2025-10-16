import yfinance as yf
import pandas as pd

def download_historical(ticker, period="2y", interval="1d"):
    """
    Download historical data, clean it, and ensure numeric Close column.
    Returns a DataFrame with datetime index and numeric OHLCV.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

        if data.empty:
            print(f"⚠️ [historical_data.py] Empty data for {ticker}")
            return pd.DataFrame()

        # Flatten columns if multi-level
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        # Keep only numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        numeric_cols = [c for c in numeric_cols if c in data.columns]
        data = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Drop rows with missing Close
        data = data.dropna(subset=['Close'])
        return data

    except Exception as e:
        print(f"⚠️ [historical_data.py] Failed to download {ticker}: {e}")
        return pd.DataFrame()
