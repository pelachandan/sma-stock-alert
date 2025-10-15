import yfinance as yf
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# 1. Get S&P 500 tickers
sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
tickers = sp500['Symbol'].tolist()

def get_sma_signals(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if len(data) < 200:
            return None

        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()

        today = data.iloc[-1]
        yesterday = data.iloc[-2]

        crossed = yesterday['SMA20'] <= yesterday['SMA50'] and today['SMA20'] > today['SMA50']
        cond2 = today['SMA50'] > today['SMA200']

        return ticker if crossed and cond2 else None
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


def send_email_alert(symbols):
    sender = "your_email@gmail.com"
    receiver = "your_email@gmail.com"
    password = "APP_PASSWORD"  # set in GitHub Secrets
    subject = f"Daily SMA Alert – {datetime.now().strftime('%Y-%m-%d')}"

    body = "Buy signals:\n" + "\n".join(symbols) if symbols else "No signals today."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())


if __name__ == "__main__":
    print("Running SMA crossover scan...")
    buy_list = run_scan()
    print("Signals found:", buy_list)

    if buy_list:
        send_email_alert(buy_list)
    else:
        send_email_alert([])
