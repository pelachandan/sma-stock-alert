import pandas as pd


def test_monitor_exits_streak_at_next_trading_session_close(monkeypatch):
    from src.position_management import monitor

    index = pd.date_range("2024-01-02", periods=100, freq="B")

    class Tracker:
        def get_all_positions(self):
            return {
                "TEST": {
                    "entry_price": 100.0,
                    "entry_date": index[-1].strftime("%Y-%m-%d"),
                    "strategy": "Streak_Position",
                    "stop_loss": 1.0,
                    "direction": "LONG",
                }
            }

        def _save_positions(self):
            return None

    close = pd.Series([100.0] * 99 + [102.0], index=index)
    data = pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1_000_000},
        index=index,
    )
    monkeypatch.setattr(monitor, "get_historical_data", lambda ticker: data)

    actions = monitor.monitor_positions(Tracker())

    assert actions["exits"][0]["type"] == "NEXT_SESSION_CLOSE"
    assert actions["exits"][0]["current_price"] == 102.0
