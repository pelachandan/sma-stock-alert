import yfinance as yf
import pandas as pd
import time
import random
from utils.ledger_utils import update_sma_ledger, update_highs_ledger

# ----------------- Market Cap -----------------
def get_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("marketCap", 0)
    except Exception:
        return 0

# ----------------- Helper to download data with exponential backoff -----------------
def download_data(ticker, period="1y", interval="1d", max_retries=5, base_delay=2):
    """
    Download historical data with exponential backoff and randomized delays.
    Returns a DataFrame (empty if all retries fail).
    """
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
                # Random delay to reduce detection as bot
                time.sleep(random.uniform(1, 3))
                return data
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(delay + random.uniform(0, 1))  # add small jitter
            delay *= 2  # exponential backoff

    print(f"❌ Failed to download data for {ticker} after {max_retries} attempts.")
    return pd.DataFrame()  # return empty DataFrame on failure

# ----------------- SMA Signals -----------------
def get_sma_signals(ticker):
    """
    Detects SMA crossover signals for a given ticker.
    Logic:
      1. Identify SMA20 crossing above SMA50 within the last 20 trading days.
      2. Ensure SMA50 > SMA200 (trend confirmation).
      3. Verify current Close is 5–10% higher than the crossover day's Close.
      4. Logs and returns the ticker if all conditions are met.
    """
    data = download_data(ticker)
    if data.empty or len(data) < 200:
        return None

    # Compute moving averages
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()

    # Look back 20 trading days
    for i in range(-20, 0):
        today = data.iloc[i]
        yesterday = data.iloc[i - 1]

        try:
            sma20_today = float(today["SMA20"].iloc[0])
            sma50_today = float(today["SMA50"].iloc[0])
            sma200_today = float(today["SMA200"].iloc[0])
            sma20_yesterday = float(yesterday["SMA20"].iloc[0])
            sma50_yesterday = float(yesterday["SMA50"].iloc[0])
        except Exception:
            continue

        if pd.isna(sma20_today) or pd.isna(sma50_today) or pd.isna(sma200_today):
            continue

        crossed = (sma20_yesterday <= sma50_yesterday) and (sma20_today > sma50_today)
        cond2 = sma50_today > sma200_today

        if crossed and cond2:
            crossover_date = today.name
            crossover_price = today["Close"]
            current_price = data.iloc[-1]["Close"]
            pct_from_crossover = (current_price - crossover_price) / crossover_price * 100

            if 5 <= pct_from_crossover <= 10:
                crossover_info = {
                    "SMA20": sma20_today,
                    "SMA50": sma50_today,
                    "SMA200": sma200_today,
                    "CrossoverDate": crossover_date,
                }
                update_sma_ledger(ticker, crossover_info)
                print(
                    f"✅ {ticker}: SMA crossover {pct_from_crossover:.2f}% above "
                    f"crossover price on {crossover_date.date()}"
                )
                return ticker

    return None

# ----------------- New Highs -----------------
def check_new_high(ticker):
    data = download_data(ticker)
    if data.empty:
        return None

    today = data.iloc[-1]
    max_close = data["Close"].max()
    if today["Close"] >= max_close:
        update_highs_ledger(ticker, today["Close"], today.name)
        return ticker

    return None
