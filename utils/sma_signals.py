from utils.historical_data import get_historical_data, scalar
from utils.ledger_utils import update_sma_ledger
import traceback

def get_sma_signals(ticker):
    try:
        data = get_historical_data(ticker)
        if data.empty or len(data) < 200:
            return None

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
                current_price = scalar(data["Close"].iloc[-1])
            except Exception as e:
                print(f"⚠️ [sma_signals.py] SMA conversion error for {ticker}: {e}")
                continue

            if None in [sma20_today, sma50_today, sma200_today, sma20_yesterday, sma50_yesterday, close_today, current_price]:
                continue

            crossed = (sma20_yesterday <= sma50_yesterday) and (sma20_today > sma50_today)
            cond2 = sma50_today > sma200_today

            if crossed and cond2:
                pct_from_crossover = (current_price - close_today) / close_today * 100

                if 5 <= pct_from_crossover <= 10:
                    crossover_info = {
                        "SMA20": sma20_today,
                        "SMA50": sma50_today,
                        "SMA200": sma200_today,
                        "CrossoverDate": today.name,
                    }
                    update_sma_ledger(ticker, crossover_info)
                    print(f"✅ {ticker}: SMA crossover {pct_from_crossover:.2f}% above "
                          f"crossover price on {today.name.date()}")
                    return ticker

        return None

    except Exception as e:
        print(f"⚠️ [sma_signals.py] Unexpected error for {ticker}: {e}")
        print(traceback.format_exc())
        return None
