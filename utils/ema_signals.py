import pandas as pd
from utils.ema_utils import compute_ema_incremental

def get_ema_signals(ticker):
    """
    Detects bullish EMA crossover (20>50 while 50>200) within last 20 days.
    Returns a dictionary with ticker info if 5–10% above crossover, else None.
    """
    df = compute_ema_incremental(ticker)
    if df.empty or len(df) < 200:
        return None

    for i in range(-20, 0):
        today = df.iloc[i]
        yesterday = df.iloc[i - 1]
        if any(pd.isna([today["EMA20"], today["EMA50"], today["EMA200"]])):
            continue

        crossed = yesterday["EMA20"] <= yesterday["EMA50"] and today["EMA20"] > today["EMA50"]
        if crossed and today["EMA50"] > today["EMA200"]:
            crossover_price = today["Close"]
            current_price = df.iloc[-1]["Close"]
            pct = round((current_price - crossover_price) / crossover_price * 100, 2)
            if 5 <= pct <= 10:
                return {
                    "ticker": ticker,
                    "CrossoverDate": str(today.name.date()),
                    "CrossoverPrice": round(crossover_price, 2),
                    "CurrentPrice": round(current_price, 2),
                    "PctAbove": pct,
                    "EMA20": today["EMA20"],
                    "EMA50": today["EMA50"],
                    "EMA200": today["EMA200"],
                }
    return None
