from __future__ import annotations

from typing import Any

import pandas as pd


def add_ema_stack_columns(
    df: pd.DataFrame,
    *,
    close_col: str = "Close",
    fast_period: int,
    medium_period: int,
    long_period: int,
) -> pd.DataFrame:
    working = df.copy()
    close = pd.to_numeric(working[close_col], errors="coerce")
    working["ema_fast"] = close.ewm(span=fast_period, adjust=False).mean()
    working["ema_medium"] = close.ewm(span=medium_period, adjust=False).mean()
    working["ema_long"] = close.ewm(span=long_period, adjust=False).mean()
    return working


def detect_recent_bullish_alignment(
    df: pd.DataFrame,
    *,
    fast_col: str = "ema_fast",
    medium_col: str = "ema_medium",
    long_col: str = "ema_long",
    cross_window_bars: int,
) -> dict[str, Any] | None:
    if df.empty or len(df) < 2:
        return None

    latest = df.iloc[-1]
    fast = pd.to_numeric(df[fast_col], errors="coerce")
    medium = pd.to_numeric(df[medium_col], errors="coerce")
    long = pd.to_numeric(df[long_col], errors="coerce")
    latest_fast = float(latest[fast_col])
    latest_medium = float(latest[medium_col])
    latest_long = float(latest[long_col])

    if not (latest_fast > latest_medium > latest_long):
        return None

    cross_info = {
        "fast_medium": _last_bullish_cross(df.index, fast, medium),
        "fast_long": _last_bullish_cross(df.index, fast, long),
        "medium_long": _last_bullish_cross(df.index, medium, long),
    }
    if any(info is None for info in cross_info.values()):
        return None

    cross_ages = {name: int(info["bars_ago"]) for name, info in cross_info.items()}
    if any(age > cross_window_bars for age in cross_ages.values()):
        return None

    return {
        "alignment_age_bars": max(cross_ages.values()),
        "cross_ages": cross_ages,
        "cross_dates": {name: info["date"] for name, info in cross_info.items()},
    }


def _last_bullish_cross(index: pd.Index, left: pd.Series, right: pd.Series) -> dict[str, Any] | None:
    prev_left = left.shift(1)
    prev_right = right.shift(1)
    crosses = (prev_left <= prev_right) & (left > right)
    cross_positions = crosses[crosses.fillna(False)].index
    if len(cross_positions) == 0:
        return None

    cross_index = cross_positions[-1]
    bars_ago = len(index) - 1 - index.get_loc(cross_index)
    return {
        "date": pd.Timestamp(cross_index),
        "bars_ago": int(bars_ago),
    }
