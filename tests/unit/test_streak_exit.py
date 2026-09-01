import pandas as pd

from src.backtesting.engine import WalkForwardBacktester


def test_streak_backtester_exits_at_next_session_close(monkeypatch):
    import src.backtesting.engine as engine

    monkeypatch.setattr(engine, "BACKTEST_BROKERAGE_ENABLED", False)
    monkeypatch.setattr(engine, "BACKTEST_TAX_ENABLED", False)
    monkeypatch.setattr(WalkForwardBacktester, "_delete_backtest_tracker_files", lambda self, path: None)
    backtester = WalkForwardBacktester(tickers=[])
    entry_date = pd.Timestamp("2024-01-02")
    position = {
        "ticker": "TEST",
        "strategy": "Streak_Position",
        "direction": "LONG",
        "entry_date": entry_date,
        "entry_price": 100.0,
        "stop_price": 1.0,
        "initial_shares": 1,
        "current_shares": 1,
        "risk_amount": 1.0,
        "max_days": 1,
        "days_held": 1,
        "pyramid_adds": [],
    }
    bar = pd.Series({"Close": 102.0, "Low": 95.0, "High": 103.0})

    result = backtester._evaluate_exit_conditions(
        position, pd.Timestamp("2024-01-03"), bar, 102.0, 2.0, pd.DataFrame()
    )

    assert result["ExitReason"] == "NextSessionClose"
    assert result["Exit"] == 102.0


def test_streak_backtester_enters_at_next_session_open(monkeypatch):
    import src.backtesting.engine as engine

    monkeypatch.setattr(WalkForwardBacktester, "_delete_backtest_tracker_files", lambda self, path: None)
    monkeypatch.setitem(engine.POSITION_MAX_PER_STRATEGY, "Streak_Position", 1)
    entry_day = pd.Timestamp("2024-01-03")
    data = pd.DataFrame(
        {"Open": [100.0, 101.5], "High": [101.0, 103.0], "Low": [99.0, 100.0], "Close": [100.5, 102.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-02"), entry_day]),
    )
    monkeypatch.setattr(engine, "get_historical_data", lambda ticker: data)
    backtester = WalkForwardBacktester(tickers=[])
    backtester.pending_entries = [{
        "Ticker": "TEST",
        "Strategy": "Streak_Position",
        "Direction": "LONG",
        "Entry": 100.5,
        "StopLoss": 1.0,
        "MaxDays": 1,
    }]

    backtester._execute_pending_entries(entry_day)

    assert len(backtester.open_positions) == 1
    assert backtester.open_positions[0]["entry_date"] == entry_day
    assert backtester.open_positions[0]["entry_price"] == 101.5
