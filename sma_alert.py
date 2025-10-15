import yfinance as yf
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import os

# ----------------- Ledger Files -----------------
SMA_LEDGER_FILE = "ledger.csv"
HIGHS_LEDGER_FILE = "highs_ledger.csv"

def load_ledger(file):
    if os.path.exists(file):
        return pd.read_csv(file, parse_dates=['CrossoverDate'] if 'ledger' in file else ['HighDate'])
    else:
        if 'ledger' in file:
            return pd.DataFrame(columns=['Ticker', 'SMA20', 'SMA50', 'SMA200', 'CrossoverDate'])
        else:
            return pd.DataFrame(columns=['Ticker', 'Close', 'HighDate'])

def save_ledger(df, file):
    df.to_csv(file, index=False)

# ----------------- Get S&P 500 tickers -----------------
sp500 = pd.read_csv("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv")
tickers = sp500['Symbol'].tolist()

# ----------------- SMA Crossover -----------------
def update_sma_ledger(ticker, crossover_info):
    ledger = load_ledger(SMA_LEDGER_FILE)

    if ticker in ledger['Ticker'].values:
        existing = ledger[ledger['Ticker'] == ticker].iloc[0]
        if crossover_info['SMA20'] < crossover_info['SMA50']:
            ledger = ledger[ledger['Ticker'] != ticker]
            save_ledger(ledger, SMA_LEDGER_FILE)
        return ledger

    new_row = {
        "Ticker": ticker,
        "SMA20": crossover_info['SMA20'],
        "SMA50": crossover_info['SMA50'],
        "SMA200": crossover_info['SMA200'],
        "CrossoverDate": crossover_info['CrossoverDate']
    }
    ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
    save_ledger(ledger, SMA_LEDGER_FILE)
    return ledger

def get_sma_signals(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if len(data) < 200:
            return None

        data['SMA20'] = data['Close'].rolling(20).mean()
        data['SMA50'] = data['Close'].rolling(50).mean()
        data['SMA200'] = data['Close'].rolling(200).mean()

        for i in range(-10, 0):
            today = data.iloc[i]
            yesterday = data.iloc[i - 1]

            if pd.isna(today['SMA20']) or pd.isna(today['SMA50']) or pd.isna(today['SMA200']):
                continue

            crossed = yesterday['SMA20'] <= yesterday['SMA50'] and today['SMA20'] > today['SMA50']
            cond2 = today['SMA50'] > today['SMA200']
            diff_pct = (today['SMA20'] - today['SMA50']) / today['SMA50'] * 100

            if crossed and cond2 and diff_pct >= 3:
                crossover_info = {
                    "SMA20": today['SMA20'],
                    "SMA50": today['SMA50'],
                    "SMA200": today['SMA200'],
                    "CrossoverDate": today.name
                }
                update_sma_ledger(ticker, crossover_info)
                return ticker
        return None
    except Exception:
        return None

# ----------------- New Highs -----------------
def update_highs_ledger(ticker, close, date):
    highs_ledger = load_ledger(HIGHS_LEDGER_FILE)
    if ticker in highs_ledger['Ticker'].values:
        return highs_ledger

    new_row = {
        "Ticker": ticker,
        "Close": close,
        "HighDate": date
    }
    highs_ledger = pd.concat([highs_ledger, pd.DataFrame([new_row])], ignore_index=True)
    save_ledger(highs_ledger, HIGHS_LEDGER_FILE)
    return highs_ledger

def check_new_high(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty:
            return None

        today = data.iloc[-1]
        max_close = data['Close'].max()
        if today['Close'] >= max_close:
            update_highs_ledger(ticker, today['Close'], today.name)
            return ticker
        return None
    except Exception:
        return None

# ----------------- Market Cap -----------------
def get_market_cap(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get('marketCap', 0)
    except Exception:
        return 0

# ----------------- Run Scan -----------------
def run_scan():
    sma_signals = []
    new_highs = []
    for t in tickers:
        cap = get_market_cap(t)
        if cap and cap > 5_000_000_000:
            signal = get_sma_signals(t)
            if signal:
                sma_signals.append(signal)
            high_signal = check_new_high(t)
            if high_signal:
                new_highs.append(high_signal)
    return sma_signals, new_highs

# ----------------- Email Alerts -----------------
def send_email_alert(symbols, subject_prefix="Daily SMA Alert"):
    if not symbols:
        body = "No signals today."
    else:
        body = "Tickers:\n" + "\n".join(symbols)

    sender = os.getenv("EMAIL_SENDER")
    receiver = os.getenv("EMAIL_RECEIVER")
    password = os.getenv("EMAIL_PASSWORD")
    subject = f"{subject_prefix} – {datetime.now().strftime('%Y-%m-%d')}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())

# ----------------- Main -----------------
if __name__ == "__main__":
    print("Running SMA crossover and 52-week high scan...")
    sma_list, high_list = run_scan()
    print("SMA Signals:", sma_list)
    print("New Highs:", high_list)

    send_email_alert(sma_list, "SMA Crossover Alert")
    send_email_alert(high_list, "New Highs Alert")
