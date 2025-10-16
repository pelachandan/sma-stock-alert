import yfinance as yf
import pandas as pd
import time
import random
from pathlib import Path

HISTORICAL_FOLDER = Path("historical_data")
HISTORICAL_FOLDER.mkdir(exist_ok=True)

def download_historical(ticker, period="2y", interval="1d", max_retries=5):
    """
    Downloads historical data sequentially with exponential backoff.
    Returns a cleaned DataFrame.
    """
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
            if data.empty or 'Close' not in data.columns:
                raise ValueError(f"Missing 'Close' column")

            # Flatten columns if multi-level
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]

            # Keep only numeric OHLCV
            numeric_cols = [c for c in ['Open','High','Low','Close','Adj Close','Volume'] if c in data.columns]
            data = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Drop rows with NaN Close
            data = data.dropna(subset=['Close'])

            # Save / update historical cache
            file_path = HISTORICAL_FOLDER / f"{ticker}.csv"
            if file_path.exists():
                cached = pd.read_csv(file_path, index_col=0, parse_dates=True)
                # Only append new dates
                new_data = data[~data.index.isin(cached.index)]
                if not new_data.empty:
                    updated = pd.concat([cached, new_data])
                    updated.to_csv(file_path)
                    return updated
                return cached
            else:
                data.to_csv(file_path)
                return data

        except Exception as e:
            wait = 2 ** attempt + random.random()
            print(f"⚠️ [historical_data.py] Attempt {attempt} failed for {ticker}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    print(f"❌ [historical_data.py] Failed to download {ticker} after {max_retries} attempts")
    return pd.DataFrame()
