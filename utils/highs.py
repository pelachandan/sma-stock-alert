from .historical_data import download_historical
from .ledger_utils import update_highs_ledger

def check_new_high(ticker):
    try:
        data = download_historical(ticker)
        if data.empty:
            return None

        # Ensure Close is numeric
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])

        max_close = data['Close'].max()
        close_today = data['Close'].iloc[-1]

        if close_today >= max_close:
            update_highs_ledger(ticker, close_today, data.index[-1])
            return ticker
        return None
    except Exception as e:
        print(f"⚠️ [highs.py] Unexpected error for {ticker}: {e}")
        return None
