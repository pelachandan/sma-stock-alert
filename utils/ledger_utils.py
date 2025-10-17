import os
import pandas as pd

# ----------------- Ledger Files -----------------
SMA_LEDGER_FILE = "ledger.csv"
HIGHS_LEDGER_FILE = "highs_ledger.csv"

# ----------------- Load & Save Ledger -----------------
def load_ledger(file):
    if os.path.exists(file):
        return pd.read_csv(
            file,
            parse_dates=['CrossoverDate'] if 'ledger' in file else ['HighDate']
        )
    else:
        if 'ledger' in file:
            return pd.DataFrame(columns=['Ticker', 'SMA20', 'SMA50', 'SMA200', 'CrossoverDate'])
        else:
            return pd.DataFrame(columns=["Ticker", "Company", "Close", "HighDate"])

def save_ledger(df, file):
    df.to_csv(file, index=False)

# ----------------- SMA Ledger -----------------
def update_sma_ledger(ticker, crossover_info):
    ledger = load_ledger(SMA_LEDGER_FILE)

    # Remove entry if SMA20 dropped below SMA50
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

    new_df = pd.DataFrame([new_row]).dropna(axis=1, how='all')
    ledger = pd.concat([ledger, new_df], ignore_index=True)
    save_ledger(ledger, SMA_LEDGER_FILE)
    return ledger

# ----------------- Highs Ledger -----------------
def update_highs_ledger(ticker, company, close, date):
    highs_ledger = load_ledger(HIGHS_LEDGER_FILE)

    if ticker in highs_ledger["Ticker"].values:
        return highs_ledger  # already recorded

    new_row = {
        "Ticker": ticker,
        "Company": company,
        "Close": close,
        "HighDate": date
    }

    new_df = pd.DataFrame([new_row]).dropna(axis=1, how="all")
    highs_ledger = pd.concat([highs_ledger, new_df], ignore_index=True)
    save_ledger(highs_ledger, HIGHS_LEDGER_FILE)
    return highs_ledger
