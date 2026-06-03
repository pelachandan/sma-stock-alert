import pandas as pd
import pytest

from src.analysis.rally_pattern_strategy import RallyPatternStrategy
from src.strategies.rally_pattern import RallyPatternPosition


def _test_rally_config() -> dict:
    return {
        "live_config": {
            "max_days": 120,
            "target_r_multiple": 2.0,
            "min_history_bars": 252,
            "max_signal_age_days": 5,
        },
        "feature_config": RallyPatternStrategy.DEFAULT_FEATURE_CONFIG,
        "score_config": RallyPatternStrategy.DEFAULT_SCORE_CONFIG,
        "entry_logic_config": RallyPatternStrategy.DEFAULT_ENTRY_LOGIC_CONFIG,
        "exit_logic_config": RallyPatternStrategy.DEFAULT_EXIT_LOGIC_CONFIG,
        "strategy_config": RallyPatternStrategy.default_strategy_config(),
        "ranking_config": RallyPatternStrategy.DEFAULT_RANKING_CONFIG,
    }


def test_rally_pattern_requires_external_config(monkeypatch):
    monkeypatch.setattr(
        RallyPatternPosition,
        "_load_required_config",
        classmethod(lambda cls: {"live_config": {}, "strategy_config": {"placeholder": "..."}}),
    )

    with pytest.raises(ValueError, match="missing required sections|placeholder"):
        RallyPatternPosition()


def test_rally_pattern_requires_complete_strategy_config(monkeypatch):
    config = _test_rally_config()
    config["strategy_config"].pop("emerging_leader_min_score")
    monkeypatch.setattr(RallyPatternPosition, "_load_required_config", classmethod(lambda cls: config))

    with pytest.raises(ValueError, match="strategy_config missing required keys"):
        RallyPatternPosition()


def test_rally_pattern_run_packages_ranked_candidate(monkeypatch):
    config = _test_rally_config()
    monkeypatch.setattr(RallyPatternPosition, "_load_required_config", classmethod(lambda cls: config))
    strategy = RallyPatternPosition()
    raw_df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-03-05")],
            "ticker": ["AAA"],
            "open": [99.0],
            "high": [101.0],
            "low": [98.5],
            "close": [100.0],
            "volume": [1_500_000],
        }
    )
    candidates = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-03-05"),
                "ticker": "AAA",
                "close": 100.0,
                "score": 92.0,
                "setup_type": "power_breakout",
                "entry_structural_support": 95.0,
                "entry_risk_per_share": 5.0,
                "volume": 1_500_000,
                "prior_20bar_high": 94.0,
                "prior_5bar_low": 93.0,
                "setup_priority": 0,
                "volume_ratio_20": 1.8,
            }
        ]
    )

    monkeypatch.setattr(strategy, "_load_history_frame", lambda tickers, as_of_date: raw_df)
    monkeypatch.setattr(strategy, "_latest_candidate_rows", lambda loaded, scan_date: candidates)

    signals = strategy.run(["AAA"], pd.Timestamp("2024-03-05"))

    assert len(signals) == 1
    signal = signals[0]
    assert signal["Ticker"] == "AAA"
    assert signal["Strategy"] == "RallyPattern_Position"
    assert signal["SetupType"] == "power_breakout"
    assert signal["LeadershipStage"] == "confirmed"
    assert signal["PositionSizeMultiplier"] == 1.0
    assert signal["Entry"] == 100.0
    assert signal["StopLoss"] == 95.0
    assert signal["Target"] == 110.0
    assert signal["ZoneSupport"] == 94.0


def test_rally_pattern_packages_ignition_with_starter_size(monkeypatch):
    config = _test_rally_config()
    monkeypatch.setattr(RallyPatternPosition, "_load_required_config", classmethod(lambda cls: config))
    strategy = RallyPatternPosition()
    raw_df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-03-05")],
            "ticker": ["AAA"],
            "open": [104.0],
            "high": [108.0],
            "low": [103.0],
            "close": [107.0],
            "volume": [2_400_000],
        }
    )
    candidates = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-03-05"),
                "ticker": "AAA",
                "close": 107.0,
                "score": 91.0,
                "setup_type": "emerging_leader_ignition",
                "leadership_stage": "emerging",
                "entry_structural_support": 101.0,
                "entry_risk_per_share": 6.0,
                "volume": 2_400_000,
                "prior_20bar_high": 101.0,
                "prior_5bar_low": 100.0,
                "setup_priority": 2,
                "volume_ratio_20": 2.0,
            }
        ]
    )

    monkeypatch.setattr(strategy, "_load_history_frame", lambda tickers, as_of_date: raw_df)
    monkeypatch.setattr(strategy, "_latest_candidate_rows", lambda loaded, scan_date: candidates)

    signals = strategy.run(["AAA"], pd.Timestamp("2024-03-05"))

    assert len(signals) == 1
    signal = signals[0]
    assert signal["SetupType"] == "emerging_leader_ignition"
    assert signal["LeadershipStage"] == "emerging"
    assert signal["PositionSizeMultiplier"] == 0.5


def test_rally_pattern_latest_candidates_skip_stale_scan(monkeypatch):
    config = _test_rally_config()
    monkeypatch.setattr(RallyPatternPosition, "_load_required_config", classmethod(lambda cls: config))
    strategy = RallyPatternPosition()
    stale_candidates = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-03-01"),
                "ticker": "AAA",
                "score": 90.0,
                "setup_priority": 0,
                "volume_ratio_20": 1.2,
            }
        ]
    )

    monkeypatch.setattr(strategy.strategy, "rank_candidates", lambda raw_df: stale_candidates)

    filtered = strategy._latest_candidate_rows(pd.DataFrame(), pd.Timestamp("2024-03-10"))

    assert filtered.empty


def test_rally_pattern_build_signal_cache_reuses_latest_valid_signal_date(monkeypatch):
    config = _test_rally_config()
    config["live_config"]["max_signal_age_days"] = 3
    config["live_config"]["min_history_bars"] = 1
    monkeypatch.setattr(RallyPatternPosition, "_load_required_config", classmethod(lambda cls: config))
    strategy = RallyPatternPosition()
    raw_df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06")],
            "ticker": ["AAA", "AAA"],
            "open": [99.0, 100.0],
            "high": [101.0, 102.0],
            "low": [98.5, 99.5],
            "close": [100.0, 101.0],
            "volume": [1_500_000, 1_600_000],
        }
    )
    ranked = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-03-05"),
                "ticker": "AAA",
                "close": 100.0,
                "score": 92.0,
                "setup_type": "continuation_shelf",
                "entry_structural_support": 95.0,
                "entry_risk_per_share": 5.0,
                "volume": 1_500_000,
                "prior_20bar_high": 94.0,
                "prior_5bar_low": 93.0,
                "setup_priority": 0,
                "volume_ratio_20": 1.8,
            }
        ]
    )

    monkeypatch.setattr(strategy, "_load_history_frame", lambda tickers, as_of_date: raw_df)
    monkeypatch.setattr(strategy.strategy, "rank_candidates", lambda loaded: ranked)

    cache = strategy.build_signal_cache(
        ["AAA"],
        [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06"), pd.Timestamp("2024-03-10")],
    )

    assert len(cache[pd.Timestamp("2024-03-05")]) == 1
    assert len(cache[pd.Timestamp("2024-03-06")]) == 1
    assert cache[pd.Timestamp("2024-03-06")][0]["Date"] == pd.Timestamp("2024-03-05")
    assert cache[pd.Timestamp("2024-03-10")] == []


def test_rally_pattern_build_signal_cache_does_not_reuse_stale_aggressive_breakout(monkeypatch):
    config = _test_rally_config()
    config["live_config"]["max_signal_age_days"] = 3
    config["live_config"]["min_history_bars"] = 1
    monkeypatch.setattr(RallyPatternPosition, "_load_required_config", classmethod(lambda cls: config))
    strategy = RallyPatternPosition()
    raw_df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06")],
            "ticker": ["AAA", "AAA"],
            "open": [99.0, 100.0],
            "high": [101.0, 102.0],
            "low": [98.5, 99.5],
            "close": [100.0, 101.0],
            "volume": [1_500_000, 1_600_000],
        }
    )
    ranked = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-03-05"),
                "ticker": "AAA",
                "close": 100.0,
                "score": 92.0,
                "setup_type": "power_breakout",
                "entry_structural_support": 95.0,
                "entry_risk_per_share": 5.0,
                "volume": 1_500_000,
                "prior_20bar_high": 94.0,
                "prior_5bar_low": 93.0,
                "setup_priority": 0,
                "volume_ratio_20": 1.8,
            }
        ]
    )

    monkeypatch.setattr(strategy, "_load_history_frame", lambda tickers, as_of_date: raw_df)
    monkeypatch.setattr(strategy.strategy, "rank_candidates", lambda loaded: ranked)

    cache = strategy.build_signal_cache(
        ["AAA"],
        [pd.Timestamp("2024-03-05"), pd.Timestamp("2024-03-06")],
    )

    assert len(cache[pd.Timestamp("2024-03-05")]) == 1
    assert cache[pd.Timestamp("2024-03-06")] == []


def test_rally_pattern_build_signal_cache_respects_min_history_per_signal_date(monkeypatch):
    config = _test_rally_config()
    config["live_config"]["min_history_bars"] = 3
    monkeypatch.setattr(RallyPatternPosition, "_load_required_config", classmethod(lambda cls: config))
    strategy = RallyPatternPosition()
    raw_df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")],
            "ticker": ["AAA", "AAA", "AAA"],
            "open": [99.0, 100.0, 101.0],
            "high": [101.0, 102.0, 103.0],
            "low": [98.5, 99.5, 100.5],
            "close": [100.0, 101.0, 102.0],
            "volume": [1_500_000, 1_600_000, 1_700_000],
        }
    )
    ranked = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-03-01"),
                "ticker": "AAA",
                "close": 100.0,
                "score": 90.0,
                "setup_type": "power_breakout",
                "entry_structural_support": 95.0,
                "entry_risk_per_share": 5.0,
                "volume": 1_500_000,
                "prior_20bar_high": 94.0,
                "prior_5bar_low": 93.0,
                "setup_priority": 0,
                "volume_ratio_20": 1.8,
            },
            {
                "Date": pd.Timestamp("2024-03-05"),
                "ticker": "AAA",
                "close": 102.0,
                "score": 92.0,
                "setup_type": "power_breakout",
                "entry_structural_support": 97.0,
                "entry_risk_per_share": 5.0,
                "volume": 1_700_000,
                "prior_20bar_high": 96.0,
                "prior_5bar_low": 95.0,
                "setup_priority": 0,
                "volume_ratio_20": 1.9,
            },
        ]
    )

    monkeypatch.setattr(strategy, "_load_history_frame", lambda tickers, as_of_date: raw_df)
    monkeypatch.setattr(strategy.strategy, "rank_candidates", lambda loaded: ranked)

    cache = strategy.build_signal_cache(
        ["AAA"],
        [pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-05")],
    )

    assert cache[pd.Timestamp("2024-03-01")] == []
    assert cache[pd.Timestamp("2024-03-04")] == []
    assert len(cache[pd.Timestamp("2024-03-05")]) == 1
    assert cache[pd.Timestamp("2024-03-05")][0]["Date"] == pd.Timestamp("2024-03-05")


def test_rally_pattern_exit_conditions_return_engine_reason(monkeypatch):
    config = _test_rally_config()
    monkeypatch.setattr(RallyPatternPosition, "_load_required_config", classmethod(lambda cls: config))
    strategy = RallyPatternPosition()
    scored = pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-03-01"),
                "ticker": "AAA",
                "close": 100.0,
                "score": 90.0,
                "pct_from_20d_high": -0.01,
            },
            {
                "Date": pd.Timestamp("2024-03-04"),
                "ticker": "AAA",
                "close": 94.0,
                "score": 68.0,
                "pct_from_20d_high": -0.06,
            },
        ]
    )
    history = pd.DataFrame(
        {
            "Open": [99.0, 95.0],
            "High": [101.0, 96.0],
            "Low": [98.0, 93.0],
            "Close": [100.0, 94.0],
            "Volume": [1_000_000, 1_200_000],
        },
        index=[pd.Timestamp("2024-03-01"), pd.Timestamp("2024-03-04")],
    )

    monkeypatch.setattr(strategy.strategy, "score_dataframe", lambda raw_df: scored.copy())
    monkeypatch.setattr(strategy.strategy, "_augment_exit_support_columns", lambda df: df.copy())
    monkeypatch.setattr(strategy.strategy, "_exit_reason", lambda row, position: "zone_support_fail")

    exit_cond = strategy.get_exit_conditions(
        {
            "ticker": "AAA",
            "entry_date": pd.Timestamp("2024-03-01"),
            "entry_price": 100.0,
            "entry_score": 90.0,
            "setup_type": "power_breakout",
            "zone_support": 95.0,
        },
        history,
        pd.Timestamp("2024-03-04"),
    )

    assert exit_cond is not None
    assert exit_cond["reason"] == "zone_support_fail"
    assert exit_cond["exit_price"] == 94.0
