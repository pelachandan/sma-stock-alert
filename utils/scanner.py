import time
import pandas as pd
from config import MIN_MARKET_CAP, SP500_SOURCE
from utils.market_data import get_market_cap
from utils.sma_signals import get_sma_signals
from utils.highs import check_new_high

BACKOFF_BASE = 2
MAX_RETRIES = 5

def run_scan():
    """Scan S&P 500 tickers for SMA crossovers and 52-week highs."""
    sp500 = pd.read_csv(SP500_SOURCE)
    tickers = sp500["Symbol"].tolist()

    sma_signals = []
    new_highs = []

    for ticker in tickers:
        # -------------------
        # Step 1: Get Market Cap
        # -------------------
        attempt = 0
        market_cap = None
        while attempt < MAX_RETRIES:
            market_cap = get_market_cap(ticker)
            if market_cap and market_cap > MIN_MARKET_CAP:
                break
            wait = BACKOFF_BASE ** attempt
            print(f"⚠️ [scanner.py] Retry {attempt+1} for {ticker} market cap in {wait}s")
            time.sleep(wait)
            attempt += 1
        else:
            print(f"⚠️ [scanner.py] Skipping {ticker} due to low or missing market cap")
            continue

        # -------------------
        # Step 2: SMA Signals
        # -------------------
        try:
            sma_result = get_sma_signals(ticker)
            if sma_result:
                sma_signals.append(ticker)
        except Exception as e:
            print(f"⚠️ [scanner.py] Error processing SMA for {ticker}: {e}")

        # -------------------
        # Step 3: 52-week Highs
        # -------------------
        try:
            high_result = check_new_high(ticker)
            if high_result:
                new_highs.append(ticker)
        except Exception as e:
            print(f"⚠️ [scanner.py] Error processing new high for {ticker}: {e}")

    print("🚀 Scan completed!")
    print(f"✅ SMA Crossovers: {sma_signals}")
    print(f"🔥 New 52-week Highs: {new_highs}")

    return sma_signals, new_highs
