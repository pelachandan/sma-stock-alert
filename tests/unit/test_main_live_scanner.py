import pandas as pd

from main import filter_trades_by_same_day_exits


def test_filter_trades_by_same_day_exits_blocks_exited_tickers():
    trades = pd.DataFrame(
        [
            {"Ticker": "NTAP", "Strategy": "RallyPattern_Position"},
            {"Ticker": "TXN", "Strategy": "RelativeStrength_Ranker_Position"},
            {"Ticker": "PSX", "Strategy": "RallyPattern_Position"},
        ]
    )

    filtered = filter_trades_by_same_day_exits(trades, {"NTAP", "PSX"})

    assert filtered["Ticker"].tolist() == ["TXN"]


def test_filter_trades_by_same_day_exits_noops_without_exits():
    trades = pd.DataFrame([{"Ticker": "NTAP"}, {"Ticker": "TXN"}])

    filtered = filter_trades_by_same_day_exits(trades, set())

    assert filtered["Ticker"].tolist() == ["NTAP", "TXN"]
