# utils/market_data.py
import yfinance as yf

def get_market_cap(ticker):
    """
    Returns the market capitalization of a ticker.
    Safely handles exceptions and returns 0 if unavailable.
    """
    try:
        info = yf.Ticker(ticker).info
        cap = info.get("marketCap", 0)
        if isinstance(cap, (int, float)):
            return cap
        # Handle case where cap is returned as a pandas Series
        if hasattr(cap, "iloc"):
            return float(cap.iloc[0])
        return 0
    except Exception as e:
        print(f"⚠️ [market_data.py] Error getting market cap for {ticker}: {e}")
        return 0
