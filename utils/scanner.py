import pandas as pd
from utils.market_data import get_market_cap, get_sma_signals, check_new_high
from config import MIN_MARKET_CAP, SP500_SOURCE

def run_scan():
    sp500 = pd.read_csv(SP500_SOURCE)
    tickers = sp500["Symbol"].tolist()

    sma_signals = []
    new_highs = []
    failed_tickers = []

    for ticker in tickers:
        try:
            cap = get_market_cap(ticker)
            if not cap or cap < MIN_MARKET_CAP:
                continue

            sma = get_sma_signals(ticker)
            high = check_new_high(ticker)

            if sma:
                print(f"✅ SMA crossover detected: {ticker}")
                sma_signals.append(ticker)
            if high:
                print(f"🔥 New 52-week high: {ticker}")
                new_highs.append(ticker)

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")
            failed_tickers.append(ticker)
            continue

    if failed_tickers:
        print(f"⚠️ Failed tickers: {failed_tickers}")

    return sma_signals, new_highs
