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

# ----------------- Helper: download with exponential backoff -----------------
def download_data(ticker, period="1y", interval="1d", max_retries=5, base_delay=2):
    """
    Download historical data with exponential backoff and random delays.
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
                # Random small delay after successful download
                time.sleep(random.uniform(1, 3))
                return data
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed for {ticker}: {e}")
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2  # exponential backoff

    print(f"❌ Failed to download data for {ticker} after {max_retries} attempts.")
    return pd.DataFrame()  # empty DataFrame on failure

# ----------------- SMA Signals -----------------
def get_sma_signals(ticker):
    """
    Detect SMA20 crossing above SMA50 in last 20 days with SMA50>SMA200.
    Current Close must be 5–10% above crossover price.
    """
    data = download_data(ticker)
    if data.empty or len(data) < 200:
        return None

    # Compute SMAs
    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()
    data["SMA200"] = data["Close"].rolling(200).mean()

    # Look back 20 trading days
    for i in range(-20, 0):
        today = data.iloc[i]
        yesterday = data.iloc[i - 1]

        # Skip if any SMA is NaN
        if pd.isna(today["SMA20"]) or pd.isna(today["SMA50"]) or pd.isna(today["SMA200"]):
            continue
        if pd.isna(yesterday["SMA20"]) or pd.isna(yesterday["SMA50"]):
            continue

        # Convert to scalar float to avoid Series ambiguity
        sma20_today = today["SMA20"].item()
        sma50_today = today["SMA50"].item()
        sma200_today = today["SMA200"].item()

        sma20_yesterday = yesterday["SMA20"].item()
        sma50_yesterday = yesterday["SMA50"].item()

        # SMA crossover condition
        crossed = (sma20_yesterday <= sma50_yesterday) and (sma20_today > sma50_today)
        cond2 = sma50_today > sma200_today

        if crossed and cond2:
            crossover_date = today.name
            crossover_price = today["Close"]
            current_price = data.iloc[-1]["Close"]
            pct_from_crossover = (current_price - crossover_price) / crossover_price * 100

            # Only 5–10% above crossover price
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

# ----------------- 52-week High -----------------
def check_new_high(ticker):
    data = download_data(ticker)
    if data.empty:
        return None

    today = data.iloc[-1]
    max_close = data["Close"].max()
    if today["Close"] >= max_close:
        update_highs_ledger(ticker, today["Close"], today.name)
        print(f"🔥 {ticker}: New 52-week high at {today['Close']}")
        return ticker

    return None
