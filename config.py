import os

# ----------------- Configuration -----------------
SMA_LEDGER_FILE = "ledger.csv"
HIGHS_LEDGER_FILE = "highs_ledger.csv"

# Market cap threshold (in USD)
MIN_MARKET_CAP = 1_000_000_000  # 1B

# Email credentials from environment
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Yahoo Finance & S&P500
SP500_SOURCE = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
