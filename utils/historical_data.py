import yfinance as yf
import pandas as pd
import os
import time
import random

HISTORICAL_FOLDER = "historical_data"
os.makedirs(HISTORICAL_FOLDER, exist_ok=True)

def scalar(val):
    if val is None:
        return None
    if isinstance(val, pd.Series):
        val = val.dropna()
        if len(val) == 0:
            return None
        return float(val.iloc[-1])
    if pd.isna(val):
        return None
    return float(val)

def download_data(ticker, period="2y", interval="1d", max_retries=5, base_delay=2):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False
            )
            if not data.empty:
                time.sleep(random.uniform(1, 2))
                return data
        except Exception as e:
            print(f"⚠️ [historical_data.py] Attempt {attempt + 1} failed for {ticker}: {e}")
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
    print(f"❌ [historical_data.py] Failed to download data for {ticker} after {max_retries} attempts.")
    return pd.DataFrame()

def get_historical_data(ticker):
    file_path = os.path.join(HISTORICAL_FOLDER, f"{ticker}.csv")
    df = pd.DataFrame()
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(
                file_path,
                index_col=0,
                parse_dates=[0],
                date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d", errors="coerce")
            )
        except Exception as e:
            print(f"⚠️ [historical_data.py] Failed to read CSV for {ticker}: {e}")
            df = pd.DataFrame()

        if not df.empty:
            last_date = df.index.max()
            today = pd.Timestamp.today().normalize()
            if today > last_date:
                try:
                    new_data = yf.download(
                        ticker,
                        start=(last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        end=today.strftime("%Y-%m-%d"),
                        progress=False,
                        auto_adjust=True,
                        threads=False
                    )
                    if not new_data.empty:
                        df = pd.concat([df, new_data])
                        df = df[~df.index.duplicated(keep="last")]
                        df.to_csv(file_path)
                except Exception as e:
                    print(f"⚠️ [historical_data.py] Failed to append data for {ticker}: {e}")
    else:
        df = download_data(ticker, period="2y")
        if not df.empty:
            df.to_csv(file_path)
    return df
