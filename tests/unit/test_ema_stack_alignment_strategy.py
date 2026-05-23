import pandas as pd

from src.strategies.ema_stack_alignment import EMAStackAlignment


def _ema_stack_settings():
    return {
        "ema_stack_alignment": {
            "ema_periods": {"fast": 20, "medium": 50, "long": 150},
            "cross_window_bars": 20,
            "slope_lookback_bars": 5,
            "min_rising_count": 2,
            "min_volume_ratio": 1.2,
            "rsi_min": 40.0,
            "rsi_max": 75.0,
            "max_close_above_fast_pct": 0.10,
            "stop_atr_mult": 3.5,
            "target_r_multiple": 2.0,
            "max_days": 120,
            "min_history_bars": 180,
        }
    }


def _sample_ohlcv():
    index = pd.date_range("2023-01-02", periods=220, freq="B")
    close = pd.Series([100 + (i * 0.15) + ((i % 7) * 0.05) for i in range(len(index))], index=index)
    return pd.DataFrame(
        {
            "Open": close - 0.3,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": [1_500_000] * (len(index) - 1) + [2_400_000],
        },
        index=index,
    )


def test_ema_stack_alignment_scan_packages_signal(monkeypatch):
    monkeypatch.setattr(
        EMAStackAlignment,
        "_load_external_settings",
        classmethod(lambda cls: _ema_stack_settings()),
    )
    monkeypatch.setattr(
        "src.strategies.ema_stack_alignment.compute_rsi",
        lambda close, period: pd.Series([55.0] * len(close), index=close.index),
    )
    monkeypatch.setattr(
        "src.strategies.ema_stack_alignment.detect_recent_bullish_alignment",
        lambda df, cross_window_bars: {
            "alignment_age_bars": 3,
            "cross_ages": {"fast_medium": 2, "fast_long": 3, "medium_long": 1},
            "cross_dates": {},
        },
    )

    strategy = EMAStackAlignment()
    signal = strategy.scan("NVDA", _sample_ohlcv(), pd.Timestamp("2024-01-05"))

    assert signal is not None
    assert signal["Ticker"] == "NVDA"
    assert signal["Strategy"] == "EMA_StackAlignment_Position"
    assert signal["SetupType"] == "ema_stack_alignment"
    assert signal["SignalType"] == "ema_stack_alignment"
    assert signal["AlignmentAgeBars"] == 3
    assert signal["CrossAgeFastMedium"] == 2
    assert signal["MaxDays"] == 120
    assert signal["StopLoss"] < signal["Entry"] < signal["Target"]


def test_monitor_positions_uses_ema_stack_alignment_trailing_exit(monkeypatch):
    from src.position_management import monitor as monitor_module

    class _Tracker:
        def __init__(self):
            self.positions = {
                "NVDA": {
                    "entry_price": 100.0,
                    "entry_date": "2024-01-02",
                    "strategy": "EMA_StackAlignment_Position",
                    "stop_loss": 80.0,
                    "max_days": 9999,
                    "closes_below_trail": 4,
                    "partial_exited": False,
                    "pyramids_added": 0,
                }
            }

        def get_all_positions(self):
            return self.positions

        def _save_positions(self):
            return None

    index = pd.date_range("2024-01-02", periods=120, freq="B")
    close = pd.Series([100.0] * 119 + [90.0], index=index)
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": [1_000_000] * len(index),
        },
        index=index,
    )

    monkeypatch.setattr(monitor_module, "get_historical_data", lambda ticker: df)

    actions = monitor_module.monitor_positions(_Tracker())

    assert any(exit_signal["type"] == "MA100_TRAIL" for exit_signal in actions["exits"])
