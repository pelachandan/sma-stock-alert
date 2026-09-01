import pandas as pd

from main import filter_trades_by_same_day_exits, is_streak_option_alert, split_streak_option_alerts


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


def test_streak_option_alert_is_excluded_from_equity_tracker_candidates():
    trades = pd.DataFrame(
        [
            {"Ticker": "NVDA", "Strategy": "Streak_Position"},
            {"Ticker": "MSFT", "Strategy": "RallyPattern_Position"},
        ]
    )

    equity_trades, option_alerts = split_streak_option_alerts(trades)

    assert equity_trades["Ticker"].tolist() == ["MSFT"]
    assert option_alerts["Ticker"].tolist() == ["NVDA"]
    assert is_streak_option_alert(option_alerts.iloc[0])
