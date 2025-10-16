import yfinance as yf
import pandas as pd
import time
import random
import traceback
from utils.ledger_utils import update_sma_ledger, update_highs_ledger

# ----------------- Market Cap -----------------
def get_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("marketCap", 0)
    except Exception as e:
        print(f"⚠️ [market_data.py] Error fetching market cap for {ticker}: {e}")
        return 0

# ----------------- Helper: download with exponential backoff -----------------
def download_data(ticker, period="1y", interval="1d", max_retries=5, base_delay=2):
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
                time.sleep(random.uniform(1, 3))
                return data
        except Exception as e:
            print(f"⚠️ [market_data.py] Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
    print(f"❌ [market_data.py] Failed to download data for {ticker} after {max_retries} attempts.")
    return pd.DataFrame()

# ----------------- SMA Signals -----------------
def get_sma_signals(ticker):
    try:
        data = download_data(ticker)
        if data.empty or len(data) < 200:
            return None

        # Compute SMAs
        data["SMA20"] = data["Close"].rolling(20).mean()
        data["SMA50"] = data["Close"].rolling(50).mean()
        data["SMA200"] = data["Close"].rolling(200).mean()

        for i in range(-20, 0):
            today = data.iloc[i]
            yesterday = data.iloc[i - 1]

            try:
                sma20_today = today["SMA20"].iloc[0] if isinstance(today["SMA20"], pd.Series) else today["SMA20"]
                sma50_today = today["SMA50"].iloc[0] if isinstance(today["SMA50"], pd.Series) else today["SMA50"]
                sma200_today = today["SMA200"].iloc[0] if isinstance(today["SMA200"], pd.Series) else today["SMA200"]
                sma20_yesterday = yesterday["SMA20"].iloc[0] if isinstance(yesterday["SMA20"], pd.Series) else yesterday["SMA20"]
                sma50_yesterday = yesterday["SMA50"].iloc[0] if isinstance(yesterday["SMA50"], pd.Series) else yesterday["SMA50"]
            except Exception as e:
                print(f"⚠️ [market_data.py] SMA conversion error for {ticker}: {e}")
                continue

            if pd.isna(sma20_today) or pd.isna(sma50_today) or pd.isna(sma200_today):
                continue

            crossed = (sma20_yesterday <= sma50_yesterday) and (sma20_today > sma50_today)
            cond2 = sma50_today > sma200_today

            if crossed and cond2:
                crossover_date = today.name
                crossover_price = today["Close"].iloc[0] if isinstance(today["Close"], pd.Series) else today["Close"]
                current_price = data["Close"].iloc[-1]
                pct_from_crossover = (current_price - crossover_price) / crossover_price * 100

                if 5 <= pct_from_crossover <= 10:
                    crossover_info = {
                        "SMA20": sma20_today,
                        "SMA50": sma50_today,
                        "SMA200": sma200_today,
                        "CrossoverDate": crossover_date,
                    }
                    update_sma_ledger(ticker, crossover_info)
                    print(f"✅ {ticker}: SMA crossover {pct_from_crossover:.2f}% above "
                          f"crossover price on {crossover_date.date()}")
                    return ticker

        return None

    except Exception as e:
        print(f"⚠️ [market_data.py] Unexpected error in get_sma_signals for {ticker}: {e}")
        print(traceback.format_exc())
        return None

# ----------------- 52-week High -----------------
def check_new_high(ticker):
    try:
        data = download_data(ticker)
        if data.empty:
            return None

        today = data.iloc[-1]
        max_close = data["Close"].max()

        # Ensure scalar for comparison
        close_today = today["Close"].iloc[0] if isinstance(today["Close"], pd.Series) else today["Close"]

        if close_today >= max_close:
            update_highs_ledger(ticker, close_today, today.name)
            print(f"🔥 {ticker}: New 52-week high at {close_today}")
            return ticker

        return None

    except Exception as e:
        print(f"⚠️ [market_data.py] Unexpected error in check_new_high for {ticker}: {e}")
        print(traceback.format_exc())
        return None
