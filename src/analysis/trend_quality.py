from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_log_regression_slope(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    safe = values.where(values > 0)
    logged = np.log(safe)
    return _rolling_regression_stat(logged, window=window, stat="slope").fillna(0.0)


def rolling_log_regression_r2(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    safe = values.where(values > 0)
    logged = np.log(safe)
    return _rolling_regression_stat(logged, window=window, stat="r2").fillna(0.0)


def rolling_positive_fraction(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return values.gt(0).astype(float).rolling(window, min_periods=window).mean().fillna(0.0)


def rolling_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    def compute(window_values: np.ndarray) -> float:
        mask = np.isfinite(window_values)
        if mask.sum() < 2:
            return np.nan
        clean = window_values[mask]
        rolling_peak = np.maximum.accumulate(clean)
        drawdowns = (clean / rolling_peak) - 1.0
        return float(drawdowns.min())

    return values.rolling(window, min_periods=window).apply(compute, raw=True).fillna(0.0)


def _rolling_regression_stat(series: pd.Series, *, window: int, stat: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    index = np.arange(window, dtype=float)

    def compute(window_values: np.ndarray) -> float:
        mask = np.isfinite(window_values)
        if mask.sum() < 3:
            return np.nan

        x = index[mask]
        y = window_values[mask]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = float(np.dot(x_centered, x_centered))
        if denominator <= 0:
            return np.nan

        slope = float(np.dot(x_centered, y_centered) / denominator)
        if stat == "slope":
            return slope

        if stat == "r2":
            ss_tot = float(np.dot(y_centered, y_centered))
            if ss_tot <= 0:
                return 0.0
            intercept = float(y.mean() - (slope * x.mean()))
            fitted = intercept + (slope * x)
            ss_res = float(np.dot(y - fitted, y - fitted))
            return max(0.0, 1.0 - (ss_res / ss_tot))

        raise ValueError(f"Unsupported rolling regression stat: {stat}")

    return values.rolling(window, min_periods=window).apply(compute, raw=True)
