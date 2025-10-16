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
    Automatically handles yfinance's '_<TICKER>' column suffix pattern.
    """
    for attempt in range(1, max_retries + 1):
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                group_by='ticker'
            )

            # --- Handle Series case ---
            if isinstance(data, pd.Series):
                data = data.to_frame().T

            # --- Flatten MultiIndex ---
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]

            # --- Fix suffix issue like "Close_MMM" ---
            suffix = f"_{ticker.upper()}"
            renamed_cols = {col: col.replace(suffix, '') for col in data.columns if col.endswith(suffix)}
            if renamed_cols:
                data = data.rename(columns=renamed_cols)

            # --- Validate 'Close' column ---
            if 'Close' not in data.columns:
                print(f"⚠️ [historical_data.py] {ticker}: Missing 'Close' column. Columns found: {list(data.columns)}")
                raise ValueError("Missing 'Close' column")

            # --- Clean numeric data ---
            numeric_cols = [c for c in ['Open','High','Low','Close','Adj Close','Volume'] if c in data.columns]
            data = data[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna(subset=['Close'])

            if data.empty:
                raise ValueError("No valid price data after cleaning")

            # --- Cache logic ---
            file_path = HISTORICAL_FOLDER / f"{ticker}.csv"
            if file_path.exists():
                try:
                    cached = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    new_data = data[~data.index.isin(cached.index)]
                    if not new_data.empty:
                        updated = pd.concat([cached, new_data]).sort_index()
                        updated.to_csv(file_path)
                        print(f"✅ [historical_data.py] Updated cache for {ticker}, added {len(new_data)} rows.")
                        return updated
                    else:
                        print(f"ℹ️ [historical_data.py] No new data for {ticker}. Using cached.")
                        return cached
                except Exception as e:
                    print(f"⚠️ [historical_data.py] Error reading cache for {ticker}: {e}. Overwriting.")
                    data.to_csv(file_path)
                    return data
            else:
                data.to_csv(file_path)
                print(f"✅ [historical_data.py] Cached new data for {ticker}.")
                return data

        except Exception as e:
            wait = 2 ** attempt + random.random()
            print(f"⚠️ [historical_data.py] Attempt {attempt} failed for {ticker}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    print(f"❌ [historical_data.py] Failed to download {ticker} after {max_retries} attempts")
    return pd.DataFrame()
