import pandas as pd

from src.analysis.trend_quality import (
    rolling_log_regression_r2,
    rolling_log_regression_slope,
    rolling_max_drawdown,
    rolling_positive_fraction,
)


def test_rolling_log_regression_slope_is_positive_for_rising_series():
    series = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])

    result = rolling_log_regression_slope(series, 5)

    assert result.iloc[-1] > 0


def test_rolling_log_regression_r2_is_high_for_clean_trend():
    series = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0, 110.0])

    result = rolling_log_regression_r2(series, 5)

    assert result.iloc[-1] > 0.95


def test_rolling_persistence_and_drawdown_capture_trend_quality():
    series = pd.Series([100.0, 104.0, 103.0, 108.0, 112.0, 109.0, 115.0])
    baseline = pd.Series([95.0, 96.0, 97.0, 99.0, 100.0, 101.0, 103.0])

    persistence = rolling_positive_fraction(series - baseline, 5)
    drawdown = rolling_max_drawdown(series, 5)

    assert persistence.iloc[-1] == 1.0
    assert drawdown.iloc[-1] < 0
