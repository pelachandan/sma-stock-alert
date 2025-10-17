import pandas as pd
import yfinance as yf
from utils.market_data import get_historical_data
from utils.ledger_utils import update_highs_ledger


def check_new_high(ticker):
    """
    Checks if current closing price is a new 52-week (2-year) high.
    Returns a dict with ticker info if new high is detected.
    """
    try:
        df = get_historical_data(ticker)
        if df.empty or "Close" not in df.columns:
            return None

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce").dropna()
        max_close = df["Close"].max()
        close_today = df["Close"].iloc[-1]

        # new high condition
        if close_today >= max_close:
            date = df.index[-1]
            info = yf.Ticker(ticker).info
            name = info.get("shortName", ticker)
            update_highs_ledger(ticker, name, close_today, date)
            return {
                "Ticker": ticker,
                "Company": name,
                "Close": round(close_today, 2),
                "HighDate": str(date.date()),
            }

        return None

    except Exception as e:
        print(f"⚠️ [highs.py] Unexpected error for {ticker}: {e}")
        return None
