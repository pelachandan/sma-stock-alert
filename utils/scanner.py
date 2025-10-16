import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from utils.market_data import get_market_cap, get_sma_signals, check_new_high
from config import MIN_MARKET_CAP, SP500_SOURCE

def run_scan():
    sp500 = pd.read_csv(SP500_SOURCE)
    tickers = sp500["Symbol"].tolist()

    sma_signals = []
    new_highs = []

    def process_ticker(ticker):
        cap = get_market_cap(ticker)
        if cap and cap > MIN_MARKET_CAP:
            sma = get_sma_signals(ticker)
            high = check_new_high(ticker)
            return ticker, sma, high
        return ticker, None, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_ticker, tickers))

    for ticker, sma, high in results:
        if sma:
            print(f"✅ SMA crossover detected: {ticker}")
            sma_signals.append(ticker)
        if high:
            print(f"🔥 New 52-week high: {ticker}")
            new_highs.append(ticker)

    return sma_signals, new_highs
