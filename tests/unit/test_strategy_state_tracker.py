import json
import tempfile
from pathlib import Path

from src.scanning.rs_bought_tracker import (
    StrategyStateTracker,
    history_file_path_for_strategy,
    strategy_file_key,
    tracker_file_path_for_strategy,
)


def test_strategy_file_paths_use_expected_keys():
    assert strategy_file_key("RelativeStrength_Ranker_Position") == "rs_ranker"
    assert strategy_file_key("RallyPattern_Position") == "rally_pattern"
    assert strategy_file_key("GapReversal_Position") == "gap_reversal"
    assert strategy_file_key("%B_MeanReversion_Position") == "percent_b_mean_reversion"
    assert tracker_file_path_for_strategy("RallyPattern_Position").endswith("rally_pattern_bought.json")
    assert history_file_path_for_strategy("GapReversal_Position").endswith("gap_reversal_trade_history.json")


def test_strategy_tracker_writes_closed_trade_history_without_prior_active_row():
    with tempfile.TemporaryDirectory(prefix="strategy-tracker-backtest-") as tmp_dir:
        base = Path(tmp_dir) / "backtest"
        bought_path = base / "rally_pattern_bought.json"
        history_path = base / "rally_pattern_trade_history.json"
        tracker = StrategyStateTracker(
            strategy_name="RallyPattern_Position",
            file_path=str(bought_path),
            history_file_path=str(history_path),
            load_from_file=False,
        )

        tracker.close_position(
            ticker="NVDA",
            exit_date="2026-04-22",
            exit_price=110.0,
            exit_reason="TIME_STOP",
            profit_loss=10.0,
            r_multiple=1.0,
            days_held=5,
            strategy="RallyPattern_Position",
            entry_date="2026-04-17",
            entry_price=100.0,
        )

        with open(history_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)

        assert len(history) == 1
        trade = next(iter(history.values()))
        assert trade["ticker"] == "NVDA"
        assert trade["strategy"] == "RallyPattern_Position"
        assert trade["entry_price"] == 100.0
        assert trade["exit_price"] == 110.0

        with open(bought_path, "r", encoding="utf-8") as handle:
            bought = json.load(handle)

        assert bought["NVDA"]["status"] == "closed"


def test_strategy_tracker_add_and_close_round_trip():
    with tempfile.TemporaryDirectory(prefix="strategy-tracker-backtest-") as tmp_dir:
        base = Path(tmp_dir) / "backtest"
        bought_path = base / "gap_reversal_bought.json"
        history_path = base / "gap_reversal_trade_history.json"
        tracker = StrategyStateTracker(
            strategy_name="GapReversal_Position",
            file_path=str(bought_path),
            history_file_path=str(history_path),
            load_from_file=False,
        )

        tracker.add_bought(
            ticker="PLTR",
            entry_date="2026-04-20",
            entry_price=40.5,
        )
        assert tracker.is_bought("PLTR")

        tracker.close_position(
            ticker="PLTR",
            exit_date="2026-04-22",
            exit_price=42.0,
            exit_reason="EMA21_TRAIL_GAP",
            profit_loss=1.5,
            r_multiple=0.8,
            days_held=2,
        )

        assert not tracker.is_bought("PLTR")
        assert tracker.is_closed("PLTR")

        with open(history_path, "r", encoding="utf-8") as handle:
            history = json.load(handle)

        trade = next(iter(history.values()))
        assert trade["strategy"] == "GapReversal_Position"
        assert trade["exit_reason"] == "EMA21_TRAIL_GAP"


def test_strategy_tracker_preserves_rally_metadata_on_close():
    with tempfile.TemporaryDirectory(prefix="strategy-tracker-backtest-") as tmp_dir:
        base = Path(tmp_dir) / "backtest"
        bought_path = base / "rally_pattern_bought.json"
        history_path = base / "rally_pattern_trade_history.json"
        tracker = StrategyStateTracker(
            strategy_name="RallyPattern_Position",
            file_path=str(bought_path),
            history_file_path=str(history_path),
            load_from_file=False,
        )

        tracker.add_bought(
            ticker="MCHP",
            entry_date="2026-05-01",
            entry_price=100.0,
            setup_type="power_breakout",
            trigger_level=98.5,
            leadership_stage="confirmed",
        )
        tracker.close_position(
            ticker="MCHP",
            exit_date="2026-05-02",
            exit_price=95.0,
            exit_reason="ZONE_SUPPORT_FAIL",
            profit_loss=-5.0,
            r_multiple=-0.8,
            days_held=1,
        )

        with open(bought_path, "r", encoding="utf-8") as handle:
            bought = json.load(handle)

        assert bought["MCHP"]["status"] == "closed"
        assert bought["MCHP"]["setup_type"] == "power_breakout"
        assert bought["MCHP"]["trigger_level"] == 98.5
        assert bought["MCHP"]["leadership_stage"] == "confirmed"
