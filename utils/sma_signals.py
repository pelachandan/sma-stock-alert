import pandas as pd
from utils.market_data import get_historical_data

def get_sma_signals(ticker):
    """
    Detects bullish SMA crossover (20>50 while 50>200) within last 20 days.
    Returns a dictionary with ticker info if 5–10% above crossover, else None.
    """
    df = get_historical_data(ticker)
    if df.empty or len(df) < 200:
        return None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    for i in range(-20, 0):
        today = df.iloc[i]
        yesterday = df.iloc[i - 1]
        if any(pd.isna([today["SMA20"], today["SMA50"], today["SMA200"]])):
            continue

        crossed = yesterday["SMA20"] <= yesterday["SMA50"] and today["SMA20"] > today["SMA50"]
        if crossed and today["SMA50"] > today["SMA200"]:
            crossover_price = today["Close"]
            current_price = df.iloc[-1]["Close"]
            pct = round((current_price - crossover_price) / crossover_price * 100, 2)
            if 5 <= pct <= 10:
                return {
                    "ticker": ticker,
                    "CrossoverDate": str(today.name.date()),  # index is date
                    "CrossoverPrice": round(crossover_price, 2),
                    "CurrentPrice": round(current_price, 2),
                    "PctAbove": pct,
                }
    return None
