import yfinance as yf
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import os

# Ledger file to maintain state between runs
LEDGER_FILE = "ledger.csv"

# 1. Get S&P 500 tickers
sp500 = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
tickers = sp500['Symbol'].tolist()


# ----------------- Ledger Management -----------------
def load_ledger():
    if os.path.exists(LEDGER_FILE):
        return pd.read_csv(LEDGER_FILE, parse_dates=['CrossoverDate'])
    else:
        return pd.DataFrame(columns=['Ticker', 'SMA20', 'SMA50', 'SMA200', 'CrossoverDate'])


def save_ledger(df):
    df.to_csv(LEDGER_FILE, index=False)


def update_ledger(ticker, crossover_info):
    ledger = load_ledger()

    # Check if ticker already in ledger
    if ticker in ledger['Ticker'].values:
        existing = ledger[ledger['Ticker'] == ticker].iloc[0]
        # Remove if SMA20 went below SMA50
        if crossover_info['SMA20'] < crossover_info['SMA50']:
            ledger = ledger[ledger['Ticker'] != ticker]
            save_ledger(ledger)
        return ledger

    # Add new ticker
    new_row = {
        "Ticker": ticker,
        "SMA20": crossover_info['SMA20'],
        "SMA50": crossover_info['SMA50'],
        "SMA200": crossover_info['SMA200'],
        "CrossoverDate": crossover_info['CrossoverDate']
    }
    ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
    save_ledger(ledger)
    return ledger


# ----------------- SMA Crossover Detection -----------------
def get_sma_signals(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if len(data) < 200:
            return None

        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()

        # Look for crossover in last 10 days
        for i in range(-10, 0):
            today = data.iloc[i]
            yesterday = data.iloc[i - 1]

            # Skip if NaN
            if pd.isna(today['SMA20']) or pd.isna(today['SMA50']) or pd.isna(today['SMA200']):
                continue

            crossed = yesterday['SMA20'] <= yesterday['SMA50'] and today['SMA20'] > today['SMA50']
            # SMA50 above SMA200 (bullish medium-term trend)
            cond2 = today['SMA50'] > today['SMA200']

            # SMA20 crossed above SMA50 by ≥3%
            diff_pct = (today['SMA20'] - today['SMA50']) / today['SMA50'] * 100

            if crossed and cond2 and diff_pct >= 3:
                crossover_info = {
                    "SMA20": today['SMA20'],
                    "SMA50": today['SMA50'],
                    "SMA200": today['SMA200'],
                    "CrossoverDate": today.name
                }
                # Update ledger
                update_ledger(ticker, crossover_info)
                return ticker
        return None

    except Exception:
        return None


def get_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('marketCap', 0)
    except Exception:
        return 0


def run_scan():
    signals = []
    for t in tickers:
        cap = get_market_cap(t)
        if cap and cap > 5_000_000_000:
            result = get_sma_signals(t)
            if result:
                signals.append(result)
    return signals


# ----------------- Email Alert -----------------
def send_email_alert(symbols):
    sender = os.getenv("EMAIL_SENDER")
    receiver = os.getenv("EMAIL_RECEIVER")
    password = os.getenv("EMAIL_PASSWORD")
    subject = f"Daily SMA Alert – {datetime.now().strftime('%Y-%m-%d')}"

    body = "Buy signals:\n" + "\n".join(symbols) if symbols else "No signals today."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())


# ----------------- Main -----------------
if __name__ == "__main__":
    print("Running SMA crossover scan...")
    buy_list = run_scan()
    print("Signals found:", buy_list)
    send_email_alert(buy_list)
