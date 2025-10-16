import os
import pandas as pd
from config import SMA_LEDGER_FILE, HIGHS_LEDGER_FILE

def load_ledger(file):
    if os.path.exists(file):
        if "ledger" in file and "highs" not in file:
            return pd.read_csv(file, parse_dates=["CrossoverDate"])
        else:
            return pd.read_csv(file, parse_dates=["HighDate"])
    else:
        if "ledger" in file and "highs" not in file:
            return pd.DataFrame(columns=["Ticker", "SMA20", "SMA50", "SMA200", "CrossoverDate"])
        else:
            return pd.DataFrame(columns=["Ticker", "Close", "HighDate"])

def save_ledger(df, file):
    df.to_csv(file, index=False)

def update_sma_ledger(ticker, crossover_info):
    ledger = load_ledger(SMA_LEDGER_FILE)
    if ticker in ledger["Ticker"].values:
        existing = ledger[ledger["Ticker"] == ticker].iloc[0]
        if crossover_info["SMA20"] < crossover_info["SMA50"]:
            ledger = ledger[ledger["Ticker"] != ticker]
            save_ledger(ledger, SMA_LEDGER_FILE)
        return ledger

    new_row = {
        "Ticker": ticker,
        "SMA20": crossover_info["SMA20"],
        "SMA50": crossover_info["SMA50"],
        "SMA200": crossover_info["SMA200"],
        "CrossoverDate": crossover_info["CrossoverDate"],
    }
    ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
    save_ledger(ledger, SMA_LEDGER_FILE)
    return ledger

def update_highs_ledger(ticker, close, date):
    highs_ledger = load_ledger(HIGHS_LEDGER_FILE)
    if ticker in highs_ledger["Ticker"].values:
        return highs_ledger
    new_row = {"Ticker": ticker, "Close": close, "HighDate": date}
    highs_ledger = pd.concat([highs_ledger, pd.DataFrame([new_row])], ignore_index=True)
    save_ledger(highs_ledger, HIGHS_LEDGER_FILE)
    return highs_ledger
