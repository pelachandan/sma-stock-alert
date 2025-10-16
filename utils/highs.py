from utils.historical_data import get_historical_data, scalar
from utils.ledger_utils import update_highs_ledger
import traceback

def check_new_high(ticker):
    try:
        data = get_historical_data(ticker)
        if data.empty:
            return None

        close_today = scalar(data["Close"].iloc[-1])
        max_close = scalar(data["Close"].max())

        if None in [close_today, max_close]:
            return None

        if close_today >= max_close:
            update_highs_ledger(ticker, close_today, data.index[-1])
            print(f"🔥 {ticker}: New 52-week high at {close_today}")
            return ticker

        return None

    except Exception as e:
        print(f"⚠️ [highs.py] Unexpected error for {ticker}: {e}")
        print(traceback.format_exc())
        return None
