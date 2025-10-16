import pandas as pd
from utils.market_data import get_historical_data
from utils.ledger_utils import update_highs_ledger

def check_new_high(ticker):
    """
    Checks if current closing price is a new 52-week (2-year) high.
    """
    try:
        df = get_historical_data(ticker)
        if df.empty or 'Close' not in df.columns:
            return None

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce').dropna()
        max_close = df['Close'].max()
        close_today = df['Close'].iloc[-1]

        if close_today >= max_close:
            update_highs_ledger(ticker, close_today, df.index[-1])
            return ticker
        return None
    except Exception as e:
        print(f"⚠️ [highs.py] Unexpected error for {ticker}: {e}")
        return None
