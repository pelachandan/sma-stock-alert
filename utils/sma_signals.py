def get_sma_signals(ticker):
    df = get_historical_data(ticker)
    if df.empty or len(df) < 200:
        return None

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    # Look for crossover in last 20 days
    for i in range(-20, 0):
        today = df.iloc[i]
        yesterday = df.iloc[i-1]

        if any(pd.isna([today["SMA20"], today["SMA50"], today["SMA200"]])):
            continue

        crossed = yesterday["SMA20"] <= yesterday["SMA50"] and today["SMA20"] > today["SMA50"]
        cond2 = today["SMA50"] > today["SMA200"]

        if crossed and cond2:
            crossover_price = today["Close"]
            current_price = df.iloc[-1]["Close"]
            pct_from_crossover = (current_price - crossover_price) / crossover_price * 100
            if 5 <= pct_from_crossover <= 10:
                return ticker
    return None

def check_new_high(ticker):
    df = get_historical_data(ticker)
    if df.empty:
        return None
    close_today = df.iloc[-1]["Close"]
    max_close = df["Close"].max()
    if close_today >= max_close:
        return ticker
    return None
