import pandas as pd

from src.analysis.ema_stack_alignment import detect_recent_bullish_alignment


def test_detect_recent_bullish_alignment_allows_non_ordered_recent_crosses():
    df = pd.DataFrame(
        {
            "ema_fast": [9.0, 11.0, 8.0, 12.0, 13.0],
            "ema_medium": [10.0, 10.0, 9.0, 11.0, 12.0],
            "ema_long": [11.0, 9.5, 10.0, 10.0, 11.0],
        },
        index=pd.date_range("2024-03-01", periods=5, freq="B"),
    )

    result = detect_recent_bullish_alignment(df, cross_window_bars=4)

    assert result is not None
    assert result["cross_ages"] == {
        "fast_medium": 1,
        "fast_long": 1,
        "medium_long": 1,
    }
    assert result["alignment_age_bars"] == 1


def test_detect_recent_bullish_alignment_rejects_stale_crosses():
    df = pd.DataFrame(
        {
            "ema_fast": [9.0, 12.0, 13.0, 14.0, 15.0],
            "ema_medium": [10.0, 11.0, 11.5, 12.0, 13.0],
            "ema_long": [11.0, 10.5, 10.8, 11.0, 11.2],
        },
        index=pd.date_range("2024-03-01", periods=5, freq="B"),
    )

    result = detect_recent_bullish_alignment(df, cross_window_bars=2)

    assert result is None
