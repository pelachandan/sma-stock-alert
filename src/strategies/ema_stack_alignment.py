from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import src.config.settings as cfg
from src.analysis.ema_stack_alignment import add_ema_stack_columns, detect_recent_bullish_alignment
from src.data.indicators import compute_rsi
from src.storage.gcs import download_file
from src.strategies.base import BaseStrategy


class EMAStackAlignment(BaseStrategy):
    name = "EMA_StackAlignment_Position"
    description = "Recent bullish EMA stack alignment after fast/medium/long cross completion"
    EXTERNAL_SETTINGS_PATH = Path("config") / "settings.json"
    REQUIRED_EXTERNAL_KEYS = {"ema_stack_alignment"}
    REQUIRED_CONFIG_KEYS = {
        "ema_periods",
        "cross_window_bars",
        "slope_lookback_bars",
        "min_rising_count",
        "min_volume_ratio",
        "rsi_min",
        "rsi_max",
        "max_close_above_fast_pct",
        "stop_atr_mult",
        "target_r_multiple",
        "max_days",
        "min_history_bars",
    }

    def __init__(self) -> None:
        super().__init__()
        settings = self._load_external_settings()
        config = dict(settings["ema_stack_alignment"])
        missing = sorted(self.REQUIRED_CONFIG_KEYS - set(config))
        if missing:
            raise ValueError(f"EMA stack alignment config missing required keys: {missing}")

        periods = config["ema_periods"]
        if not {"fast", "medium", "long"}.issubset(periods):
            raise ValueError("EMA stack alignment config requires ema_periods.fast/medium/long.")

        self.fast_period = int(periods["fast"])
        self.medium_period = int(periods["medium"])
        self.long_period = int(periods["long"])
        self.cross_window_bars = int(config["cross_window_bars"])
        self.slope_lookback_bars = int(config["slope_lookback_bars"])
        self.min_rising_count = int(config["min_rising_count"])
        self.min_volume_ratio = float(config["min_volume_ratio"])
        self.rsi_min = float(config["rsi_min"])
        self.rsi_max = float(config["rsi_max"])
        self.max_close_above_fast_pct = float(config["max_close_above_fast_pct"])
        self.stop_atr_mult = float(config["stop_atr_mult"])
        self.target_r_multiple = float(config["target_r_multiple"])
        self.max_days = int(config["max_days"])
        self.min_history_bars = int(config["min_history_bars"])

    def scan(
        self,
        ticker: str,
        df: pd.DataFrame,
        as_of_date: pd.Timestamp = None,
    ) -> Optional[dict[str, Any]]:
        working = df.copy()
        if not isinstance(working.index, pd.DatetimeIndex):
            working.index = pd.to_datetime(working.index, errors="coerce")
            working = working[working.index.notna()]
        if as_of_date is not None:
            working = working[working.index <= pd.Timestamp(as_of_date)]
        if working.empty or len(working) < self.min_history_bars:
            return None

        working = add_ema_stack_columns(
            working,
            fast_period=self.fast_period,
            medium_period=self.medium_period,
            long_period=self.long_period,
        )
        working["AvgVolume20"] = working["Volume"].rolling(20).mean()
        working["VolumeRatio20"] = working["Volume"] / working["AvgVolume20"].replace(0, pd.NA)
        working["ATR20"] = _calculate_atr(working, period=20)
        working["RSI14"] = compute_rsi(working["Close"], 14)

        alignment = detect_recent_bullish_alignment(
            working,
            cross_window_bars=self.cross_window_bars,
        )
        if alignment is None:
            return None

        latest = working.iloc[-1]
        if pd.isna(latest["VolumeRatio20"]) or pd.isna(latest["ATR20"]) or pd.isna(latest["RSI14"]):
            return None
        if float(latest["VolumeRatio20"]) < self.min_volume_ratio:
            return None
        if not (self.rsi_min <= float(latest["RSI14"]) <= self.rsi_max):
            return None

        rising_count = self._rising_ema_count(working)
        if rising_count < self.min_rising_count:
            return None

        close = float(latest["Close"])
        ema_fast = float(latest["ema_fast"])
        ema_medium = float(latest["ema_medium"])
        if close < min(ema_fast, ema_medium):
            return None

        close_above_fast_pct = (close - ema_fast) / ema_fast if ema_fast > 0 else 0.0
        if close_above_fast_pct > self.max_close_above_fast_pct:
            return None

        atr20 = float(latest["ATR20"])
        if atr20 <= 0:
            return None
        stop = close - (self.stop_atr_mult * atr20)
        if stop <= 0 or stop >= close:
            return None
        target = close + (self.target_r_multiple * (close - stop))
        score = self._score_signal(
            volume_ratio=float(latest["VolumeRatio20"]),
            alignment_age_bars=int(alignment["alignment_age_bars"]),
            rising_count=rising_count,
            close_above_fast_pct=close_above_fast_pct,
        )

        signal_date = pd.Timestamp(working.index[-1])
        return {
            "Ticker": ticker,
            "Strategy": self.name,
            "Direction": "LONG",
            "Priority": cfg.STRATEGY_PRIORITY.get(self.name, 5),
            "Price": round(close, 2),
            "Close": round(close, 2),
            "Entry": round(close, 2),
            "StopLoss": round(stop, 2),
            "StopPrice": round(stop, 2),
            "Target": round(target, 2),
            "Score": round(score, 2),
            "Volume": int(float(latest["Volume"])),
            "Date": signal_date,
            "AsOfDate": signal_date,
            "MaxDays": self.max_days,
            "SetupType": "ema_stack_alignment",
            "SignalType": "ema_stack_alignment",
            "AlignmentAgeBars": int(alignment["alignment_age_bars"]),
            "CrossAgeFastMedium": int(alignment["cross_ages"]["fast_medium"]),
            "CrossAgeFastLong": int(alignment["cross_ages"]["fast_long"]),
            "CrossAgeMediumLong": int(alignment["cross_ages"]["medium_long"]),
            "EMAFast": round(float(latest["ema_fast"]), 2),
            "EMAMedium": round(float(latest["ema_medium"]), 2),
            "EMALong": round(float(latest["ema_long"]), 2),
            "VolumeRatio20": round(float(latest["VolumeRatio20"]), 2),
            "RSI14": round(float(latest["RSI14"]), 2),
        }

    @classmethod
    def _load_external_settings(cls) -> dict[str, Any]:
        if cls.EXTERNAL_SETTINGS_PATH.exists():
            with cls.EXTERNAL_SETTINGS_PATH.open("r", encoding="utf-8") as handle:
                settings = json.load(handle)
        else:
            with tempfile.TemporaryDirectory(prefix="ema-stack-settings-") as tmp_dir:
                local_path = Path(tmp_dir) / "settings.json"
                if not download_file("config/settings.json", local_path):
                    raise FileNotFoundError(
                        "Missing required settings file: config\\settings.json "
                        "(expected locally or in GCS)."
                    )
                with local_path.open("r", encoding="utf-8") as handle:
                    settings = json.load(handle)

        missing = sorted(key for key in cls.REQUIRED_EXTERNAL_KEYS if key not in settings)
        if missing:
            raise ValueError(f"EMA stack alignment config missing required settings keys: {missing}")
        return settings

    def _rising_ema_count(self, working: pd.DataFrame) -> int:
        if len(working) <= self.slope_lookback_bars:
            return 0
        lookback = self.slope_lookback_bars + 1
        latest = working.iloc[-1]
        past = working.iloc[-lookback]
        return sum(
            float(latest[column]) > float(past[column])
            for column in ("ema_fast", "ema_medium", "ema_long")
        )

    def _score_signal(
        self,
        *,
        volume_ratio: float,
        alignment_age_bars: int,
        rising_count: int,
        close_above_fast_pct: float,
    ) -> float:
        freshness = max(0.0, 1.0 - (alignment_age_bars / max(self.cross_window_bars, 1)))
        volume_score = min(volume_ratio / max(self.min_volume_ratio, 0.01), 2.0) / 2.0
        rising_score = rising_count / 3.0
        extension_penalty = min(max(close_above_fast_pct, 0.0) / max(self.max_close_above_fast_pct, 0.01), 1.0)
        return 55.0 + (freshness * 20.0) + (volume_score * 15.0) + (rising_score * 15.0) - (extension_penalty * 10.0)


def _calculate_atr(df: pd.DataFrame, *, period: int) -> pd.Series:
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()
