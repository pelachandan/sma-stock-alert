import logging

import pandas as pd

from src.backtesting.engine import WalkForwardBacktester


def _make_backtester() -> WalkForwardBacktester:
    backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
    backtester.strategy_positions = {}
    backtester.strategy_bucket_limits = {
        "RallyPattern_Position": {"emerging": 1, "confirmed": 2}
    }
    backtester.strategy_bucket_positions = {}
    return backtester


def test_resolve_trade_bucket_uses_rally_setup_type():
    assert WalkForwardBacktester._resolve_trade_bucket(
        {"Strategy": "RallyPattern_Position", "SetupType": "emerging_leader_breakout"}
    ) == "emerging"
    assert WalkForwardBacktester._resolve_trade_bucket(
        {"Strategy": "RallyPattern_Position", "SetupType": "emerging_leader_ignition"}
    ) == "emerging"
    assert WalkForwardBacktester._resolve_trade_bucket(
        {"Strategy": "RallyPattern_Position", "SetupType": "power_breakout"}
    ) == "confirmed"
    assert WalkForwardBacktester._resolve_trade_bucket(
        {"Strategy": "GapReversal_Position", "SetupType": "gap_long"}
    ) is None


def test_increment_and_decrement_position_counters_track_rally_buckets():
    backtester = _make_backtester()
    emerging_trade = {
        "Strategy": "RallyPattern_Position",
        "SetupType": "emerging_leader_shelf",
    }

    backtester._increment_position_counters(emerging_trade)

    assert backtester.strategy_positions["RallyPattern_Position"] == 1
    assert backtester.strategy_bucket_positions["RallyPattern_Position"]["emerging"] == 1

    backtester._decrement_position_counters(
        {"strategy": "RallyPattern_Position", "setup_type": "emerging_leader_shelf"}
    )

    assert backtester.strategy_positions["RallyPattern_Position"] == 0
    assert backtester.strategy_bucket_positions["RallyPattern_Position"]["emerging"] == 0


def test_bucket_skip_reason_blocks_only_full_rally_bucket():
    backtester = _make_backtester()
    backtester.strategy_positions["RallyPattern_Position"] = 1
    backtester.strategy_bucket_positions["RallyPattern_Position"] = {"emerging": 1}

    assert backtester._bucket_skip_reason(
        {"Strategy": "RallyPattern_Position", "SetupType": "emerging_leader_breakout"}
    ) == "RallyPattern_Position:emerging_cap"
    assert backtester._bucket_skip_reason(
        {"Strategy": "RallyPattern_Position", "SetupType": "power_breakout"}
    ) is None


def test_enter_position_applies_position_size_multiplier():
    backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
    backtester.log = logging.getLogger("test_backtester")
    backtester.current_capital = 100_000.0
    backtester.initial_capital = 100_000.0
    backtester.regime_params = {}
    backtester.open_positions = []

    success = backtester._enter_position(
        entry_day="2024-03-05",
        trade={
            "Ticker": "AAA",
            "Strategy": "RallyPattern_Position",
            "Entry": 100.0,
            "StopLoss": 95.0,
            "SignalType": "emerging_leader_ignition",
            "SetupType": "emerging_leader_ignition",
            "PositionSizeMultiplier": 0.5,
        },
    )

    assert success
    assert len(backtester.open_positions) == 1
    assert backtester.open_positions[0]["initial_shares"] == 200
    assert backtester.open_positions[0]["position_size_multiplier"] == 0.5


def test_scan_signals_for_day_uses_injected_provider():
    backtester = WalkForwardBacktester.__new__(WalkForwardBacktester)
    backtester.scan_provider = lambda day: [{"Ticker": "AAA", "Date": day}]
    backtester.tickers = ["AAA"]
    backtester.rs_bought_tracker = None

    signals = backtester._scan_signals_for_day("2024-03-05")

    assert signals[0]["Ticker"] == "AAA"
    assert signals[0]["Date"] == pd.Timestamp("2024-03-05")
