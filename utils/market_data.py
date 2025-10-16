import yfinance as yf
import pandas as pd
import time
import random
import traceback
from utils.ledger_utils import update_sma_ledger, update_highs_ledger

# ----------------- Helper -----------------
def scalar(val):
    """
    Convert a pandas Series, numpy scalar, or pandas NA to a Python float/int.
    - Single-element Series -> returns the element
    - Multi-element Series -> returns the max as a safe fallback
    - NaN or NA -> returns None
    """
    if val is None:
        return None
    if isinstance(val, pd.Series):
        val = val.dropna()
        if len(val) == 0:
            return None
        elif len(val) == 1:
            return val.iloc[0]
        else:
            return val.max()
    if pd.isna(val):
        return None
    return float(val)

# ----------------- Market Cap -----------------
def get_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).info
        cap = info.get("marketCap", 0)
        return scalar(cap)
    except Exception as e:
        print(f"⚠️ [market_data.py] Error fetching market cap for {ticker}: {e}")
        return 0

# ----------------- Download Data with Exponential Backoff -----------------
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
                # small random delay to avoid throttling
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

        # Compute moving averages
        data["SMA20"] = data["Close"].rolling(20).mean()
        data["SMA50"] = data["Close"].rolling(50).mean()
        data["SMA200"] = data["Close"].rolling(200).mean()

        for i in range(-20, 0):
            today = data.iloc[i]
            yesterday = data.iloc[i - 1]

            try:
                sma20_today = scalar(today["SMA20"])
                sma50_today = scalar(today["SMA50"])
                sma200_today = scalar(today["SMA200"])
                sma20_yesterday = scalar(yesterday["SMA20"])
                sma50_yesterday = scalar(yesterday["SMA50"])
                close_today = scalar(today["Close"])
            except Exception as e:
                print(f"⚠️ [market_data.py] SMA conversion error for {ticker}: {e}")
                continue

            if None in [sma20_today, sma50_today, sma200_today, sma20_yesterday, sma50_yesterday, close_today]:
                continue

            crossed = (sma20_yesterday <= sma50_yesterday) and (sma20_today > sma50_today)
            cond2 = sma50_today > sma200_today

            if crossed and cond2:
                crossover_date = today.name
                crossover_price = close_today
                current_price = scalar(data["Close"].iloc[-1])
                if current_price is None:
                    continue

                pct_from_crossover = (current_price - crossover_price) / crossover_price * 100

                # Only keep stocks that have risen 5–10% since crossover
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
        close_today = scalar(today["Close"])
        max_close = scalar(data["Close"].max())

        if None in [close_today, max_close]:
            return None

        if close_today >= max_close:
            update_highs_ledger(ticker, close_today, today.name)
            print(f"🔥 {ticker}: New 52-week high at {close_today}")
            return ticker

        return None

    except Exception as e:
        print(f"⚠️ [market_data.py] Unexpected error in check_new_high for {ticker}: {e}")
        print(traceback.format_exc())
        return None
