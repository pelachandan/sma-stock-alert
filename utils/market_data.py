import yfinance as yf
import pandas as pd
from utils.ledger_utils import update_sma_ledger, update_highs_ledger

def get_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("marketCap", 0)
    except Exception:
        return 0

def get_sma_signals(ticker):
    """
    Detects SMA crossover signals for a given ticker.

    Logic:
      1. Identify SMA20 crossing above SMA50 within the last 20 trading days.
      2. Ensure SMA50 > SMA200 (trend confirmation).
      3. Verify current Close is 5–10% higher than the crossover day's Close.
      4. Logs and returns the ticker if all conditions are met.
    """
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if len(data) < 200:
            return None

        # Compute moving averages
        data["SMA20"] = data["Close"].rolling(20).mean()
        data["SMA50"] = data["Close"].rolling(50).mean()
        data["SMA200"] = data["Close"].rolling(200).mean()

        # Look back up to 20 trading days for a recent crossover
        for i in range(-20, 0):
            today = data.iloc[i]
            yesterday = data.iloc[i - 1]

            if pd.isna(today["SMA20"]) or pd.isna(today["SMA50"]) or pd.isna(today["SMA200"]):
                continue

            crossed = yesterday["SMA20"] <= yesterday["SMA50"] and today["SMA20"] > today["SMA50"]
            cond2 = today["SMA50"] > today["SMA200"]

            if crossed and cond2:
                crossover_date = today.name
                crossover_price = today["Close"]

                # Get current price
                current_price = data.iloc[-1]["Close"]
                pct_from_crossover = (current_price - crossover_price) / crossover_price * 100

                # Only keep stocks that have risen 5–10% since crossover
                if 5 <= pct_from_crossover <= 10:
                    crossover_info = {
                        "SMA20": today["SMA20"],
                        "SMA50": today["SMA50"],
                        "SMA200": today["SMA200"],
                        "CrossoverDate": crossover_date,
                    }
                    update_sma_ledger(ticker, crossover_info)

                    print(
                        f"✅ {ticker}: SMA crossover {pct_from_crossover:.2f}% above crossover price "
                        f"on {crossover_date.date()}"
                    )

                    return ticker

        return None
    except Exception as e:
        print(f"Error in {ticker}: {e}")
        return None

def check_new_high(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty:
            return None
        today = data.iloc[-1]
        max_close = data["Close"].max()
        if today["Close"] >= max_close:
            update_highs_ledger(ticker, today["Close"], today.name)
            return ticker
        return None
    except Exception:
        return None
