import tempfile
from pathlib import Path

import pandas as pd

from src.scanning.rs_bought_tracker import StrategyStateTracker
from src.scanning.validator import pre_buy_check


def _history_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-03-01", periods=80, freq="B")
    return pd.DataFrame(
        {
            "Open": [100.0 + (i * 0.1) for i in range(len(dates))],
            "High": [101.0 + (i * 0.1) for i in range(len(dates))],
            "Low": [99.0 + (i * 0.1) for i in range(len(dates))],
            "Close": [100.0 + (i * 0.1) for i in range(len(dates))],
            "Volume": [2_000_000 for _ in range(len(dates))],
        },
        index=dates,
    )


def test_pre_buy_check_blocks_invalidated_rally_same_trigger_retry(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="rally-validator-") as tmp_dir:
        base = Path(tmp_dir) / "backtest"
        tracker = StrategyStateTracker(
            strategy_name="RallyPattern_Position",
            file_path=str(base / "rally_pattern_bought.json"),
            history_file_path=str(base / "rally_pattern_trade_history.json"),
            load_from_file=False,
        )
        tracker.add_bought(
            ticker="MCHP",
            entry_date="2026-06-01",
            entry_price=100.0,
            strategy="RallyPattern_Position",
            setup_type="power_breakout",
            trigger_level=98.5,
        )
        tracker.close_position(
            ticker="MCHP",
            exit_date="2026-06-02",
            exit_price=95.0,
            exit_reason="ZONE_SUPPORT_FAIL",
            profit_loss=-5.0,
            r_multiple=-0.8,
            days_held=1,
        )

        monkeypatch.setattr("src.scanning.validator.get_historical_data", lambda ticker: _history_frame())

        signals = [
            {
                "Ticker": "MCHP",
                "Strategy": "RallyPattern_Position",
                "Entry": 100.2,
                "StopLoss": 96.0,
                "Target": 108.6,
                "Score": 88.0,
                "Direction": "LONG",
                "SetupType": "power_breakout",
                "TriggerLevel": 98.6,
            }
        ]

        trades = pre_buy_check(
            signals,
            as_of_date=pd.Timestamp("2026-06-03"),
            strategy_trackers={"RallyPattern_Position": tracker},
        )

        assert trades.empty


def test_pre_buy_check_allows_rally_retry_after_fresh_breakout(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="rally-validator-") as tmp_dir:
        base = Path(tmp_dir) / "backtest"
        tracker = StrategyStateTracker(
            strategy_name="RallyPattern_Position",
            file_path=str(base / "rally_pattern_bought.json"),
            history_file_path=str(base / "rally_pattern_trade_history.json"),
            load_from_file=False,
        )
        tracker.add_bought(
            ticker="MCHP",
            entry_date="2026-06-01",
            entry_price=100.0,
            strategy="RallyPattern_Position",
            setup_type="power_breakout",
            trigger_level=98.5,
        )
        tracker.close_position(
            ticker="MCHP",
            exit_date="2026-06-02",
            exit_price=95.0,
            exit_reason="ZONE_SUPPORT_FAIL",
            profit_loss=-5.0,
            r_multiple=-0.8,
            days_held=1,
        )

        monkeypatch.setattr("src.scanning.validator.get_historical_data", lambda ticker: _history_frame())

        signals = [
            {
                "Ticker": "MCHP",
                "Strategy": "RallyPattern_Position",
                "Entry": 103.0,
                "StopLoss": 98.0,
                "Target": 113.0,
                "Score": 90.0,
                "Direction": "LONG",
                "SetupType": "power_breakout",
                "TriggerLevel": 101.0,
            }
        ]

        trades = pre_buy_check(
            signals,
            as_of_date=pd.Timestamp("2026-06-03"),
            strategy_trackers={"RallyPattern_Position": tracker},
        )

        assert trades["Ticker"].tolist() == ["MCHP"]
