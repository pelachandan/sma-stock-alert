import pandas as pd

from main import filter_trades_by_same_day_exits


def test_filter_trades_by_same_day_exits_blocks_reentry_tickers():
    trades = pd.DataFrame(
        [
            {"Ticker": "XOM", "Strategy": "RallyPattern_Position"},
            {"Ticker": "COST", "Strategy": "RallyPattern_Position"},
            {"Ticker": "AAPL", "Strategy": "RelativeStrength_Ranker_Position"},
        ]
    )

    filtered = filter_trades_by_same_day_exits(trades, {"XOM", "COST"})

    assert filtered["Ticker"].tolist() == ["AAPL"]


def test_filter_trades_by_same_day_exits_noops_without_exits():
    trades = pd.DataFrame([{"Ticker": "XOM"}, {"Ticker": "AAPL"}])

    filtered = filter_trades_by_same_day_exits(trades, set())

    assert filtered["Ticker"].tolist() == ["XOM", "AAPL"]
