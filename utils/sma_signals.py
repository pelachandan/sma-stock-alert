from .historical_data import download_historical
from .ledger_utils import update_sma_ledger

def get_sma_signals(ticker):
    try:
        data = download_historical(ticker)
        if data.empty or len(data) < 200:
            return None

        # Ensure Close is numeric
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data = data.dropna(subset=['Close'])

        data['SMA20'] = data['Close'].rolling(20).mean()
        data['SMA50'] = data['Close'].rolling(50).mean()
        data['SMA200'] = data['Close'].rolling(200).mean()

        # Rest of SMA crossover logic here...
    except Exception as e:
        print(f"⚠️ [sma_signals.py] Unexpected error for {ticker}: {e}")
        return None
