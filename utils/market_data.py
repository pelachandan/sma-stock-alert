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
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if len(data) < 200:
            return None

        data["SMA20"] = data["Close"].rolling(20).mean()
        data["SMA50"] = data["Close"].rolling(50).mean()
        data["SMA200"] = data["Close"].rolling(200).mean()

        for i in range(-10, 0):
            today = data.iloc[i]
            yesterday = data.iloc[i - 1]
            if pd.isna(today["SMA20"]) or pd.isna(today["SMA50"]) or pd.isna(today["SMA200"]):
                continue

            crossed = yesterday["SMA20"] <= yesterday["SMA50"] and today["SMA20"] > today["SMA50"]
            cond2 = today["SMA50"] > today["SMA200"]
            diff_pct = (today["SMA20"] - today["SMA50"]) / today["SMA50"] * 100

            # Less strict diff threshold so more signals appear
            if crossed and cond2 and diff_pct >= 1:
                crossover_info = {
                    "SMA20": today["SMA20"],
                    "SMA50": today["SMA50"],
                    "SMA200": today["SMA200"],
                    "CrossoverDate": today.name,
                }
                update_sma_ledger(ticker, crossover_info)
                return ticker
        return None
    except Exception:
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
