"""Walk-forward-safe cross-sectional next-day green ranker."""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd

import src.config.settings as cfg
from src.data.market import get_historical_data
from src.storage.gcs import download_file
from src.strategies.base import BaseStrategy


class StreakPosition(BaseStrategy):
    """Select one next-session bullish option idea from a fixed liquid universe."""

    name = "Streak_Position"
    description = "Daily one-year empirical ranker for one next-day green ITM-call idea"
    ALLOWED_TICKERS = ("NVDA", "TSLA", "AAPL", "AMZN", "META", "GOOGL", "MSFT", "AMD")
    EXTERNAL_SETTINGS_PATH = "config/settings.json"
    REQUIRED_EXTERNAL_KEYS = {
        "STREAK_RANKER_ROLLING_SESSIONS",
        "STREAK_RANKER_MIN_TRAINING_SESSIONS",
        "STREAK_RANKER_SMOOTHING",
    }

    def scan(
        self,
        ticker: str,
        df: pd.DataFrame,
        as_of_date: pd.Timestamp = None,
    ) -> Optional[dict[str, Any]]:
        """Cross-sectional ranking is performed by ``run``, never ticker-by-ticker."""
        return None

    def run(
        self,
        tickers: list[str],
        as_of_date: pd.Timestamp = None,
    ) -> list[dict[str, Any]]:
        """Return the single highest-ranked eligible universe member for ``as_of_date``."""
        if as_of_date is None:
            return []

        as_of = pd.Timestamp(as_of_date).normalize()
        settings = self._load_external_settings()
        histories = {
            ticker: self._prepare_history(get_historical_data(ticker), as_of)
            for ticker in self.ALLOWED_TICKERS
        }
        if any(history.empty for history in histories.values()):
            return []

        qqq = self._prepare_history(get_historical_data("QQQ"), as_of)
        frames = {
            ticker: self._feature_frame(history, qqq)
            for ticker, history in histories.items()
        }
        candidates = []
        for ticker, frame in frames.items():
            if as_of not in frame.index:
                return []
            candidate = frame.loc[as_of]
            if candidate[list(self._REQUIRED_NUMERIC_FEATURES)].isna().any():
                return []
            candidates.append((ticker, candidate))

        training = self._training_rows(
            frames,
            as_of,
            rolling_sessions=int(settings["STREAK_RANKER_ROLLING_SESSIONS"]),
        )
        if training.empty or training.index.nunique() < int(settings["STREAK_RANKER_MIN_TRAINING_SESSIONS"]):
            return []

        ranked = []
        for order, (ticker, candidate) in enumerate(candidates):
            probability, details = self._estimate_probability(
                training,
                candidate,
                smoothing=float(settings["STREAK_RANKER_SMOOTHING"]),
            )
            ranked.append((probability, order, ticker, candidate, details))

        probability, _, ticker, candidate, details = sorted(
            ranked, key=lambda item: (-item[0], item[1])
        )[0]
        latest_close = float(candidate["Close"])
        return [self._signal(ticker, as_of, latest_close, candidate, probability, details)]

    _MODEL_FEATURES = (
        "CandleDirection",
        "StreakLength",
        "RSI14",
        "PercentB20",
        "VolumeRatio20",
        "Return5",
        "Return20",
        "EMA20AboveEMA50",
        "QQQRegime",
    )
    _REQUIRED_NUMERIC_FEATURES = (
        "RSI14",
        "PercentB20",
        "VolumeRatio20",
        "Return5",
        "Return20",
        "EMA20",
        "EMA50",
    )

    @staticmethod
    def _prepare_history(data: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        if data is None or data.empty or "Close" not in data:
            return pd.DataFrame()
        working = data.copy()
        if not isinstance(working.index, pd.DatetimeIndex):
            working.index = pd.to_datetime(working.index, errors="coerce")
        working = working[working.index.notna()].sort_index()
        if getattr(working.index, "tz", None) is not None:
            working.index = working.index.tz_localize(None)
        working.index = working.index.normalize()
        working = working[~working.index.duplicated(keep="last")]
        working = working[working.index <= as_of]
        working["Close"] = pd.to_numeric(working["Close"], errors="coerce")
        if "Volume" in working:
            working["Volume"] = pd.to_numeric(working["Volume"], errors="coerce")
        return working.dropna(subset=["Close"])

    @classmethod
    def _feature_frame(cls, data: pd.DataFrame, qqq: pd.DataFrame) -> pd.DataFrame:
        close = data["Close"]
        returns = close.pct_change()
        direction = pd.Series("FLAT", index=data.index, dtype="object")
        direction.loc[returns > 0] = "GREEN"
        direction.loc[returns < 0] = "RED"
        group = direction.ne(direction.shift()).cumsum()
        streak = direction.groupby(group).cumcount() + 1
        streak.loc[direction == "FLAT"] = 0

        delta = close.diff()
        gains = delta.clip(lower=0).rolling(14).mean()
        losses = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + gains / losses.mask(losses == 0)))
        middle = close.rolling(20).mean()
        std = close.rolling(20).std()
        percent_b = (close - (middle - 2 * std)) / (4 * std)
        volume = data.get("Volume", pd.Series(index=data.index, dtype=float))
        volume_ratio = volume / volume.rolling(20).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        qqq_regime = pd.Series("UNAVAILABLE", index=data.index, dtype="object")
        if not qqq.empty and "Close" in qqq:
            qqq_close = qqq["Close"].reindex(data.index).ffill()
            qqq_ema20 = qqq_close.ewm(span=20, adjust=False).mean()
            available = qqq_close.notna() & qqq_ema20.notna()
            qqq_regime.loc[available & (qqq_close > qqq_ema20)] = "RISK_ON"
            qqq_regime.loc[available & (qqq_close <= qqq_ema20)] = "RISK_OFF"

        return pd.DataFrame(
            {
                "Close": close,
                "CandleDirection": direction,
                "StreakLength": streak.clip(upper=5),
                "RSI14": rsi,
                "PercentB20": percent_b,
                "VolumeRatio20": volume_ratio,
                "Return5": close.pct_change(5),
                "Return20": close.pct_change(20),
                "EMA20": ema20,
                "EMA50": ema50,
                "EMA20AboveEMA50": ema20 > ema50,
                "QQQRegime": qqq_regime,
                # The final row has no label because data was already capped at as_of.
                "Label": (close.shift(-1) > close).where(close.shift(-1).notna()),
            },
            index=data.index,
        )

    @classmethod
    def _training_rows(
        cls,
        frames: dict[str, pd.DataFrame],
        as_of: pd.Timestamp,
        *,
        rolling_sessions: int,
    ) -> pd.DataFrame:
        prior_dates = sorted(
            {
                date
                for frame in frames.values()
                for date in frame.index
                if date < as_of
            }
        )
        dates = prior_dates[-rolling_sessions:]
        rows = [
            frame.loc[frame.index.isin(dates)].assign(Ticker=ticker)
            for ticker, frame in frames.items()
        ]
        training = pd.concat(rows)
        return training.dropna(subset=["Label"])

    @classmethod
    def _estimate_probability(
        cls,
        training: pd.DataFrame,
        candidate: pd.Series,
        *,
        smoothing: float,
    ) -> tuple[float, list[tuple[str, str, int, float]]]:
        labels = training["Label"].astype(bool)
        base_rate = (labels.sum() + smoothing) / (len(labels) + 2 * smoothing)
        conditional_rates = []
        details = []
        for feature in cls._MODEL_FEATURES:
            bucket = cls._bucket(feature, candidate[feature])
            buckets = training[feature].map(lambda value: cls._bucket(feature, value))
            matching = labels[buckets == bucket]
            sample_size = len(matching)
            rate = (matching.sum() + smoothing * base_rate) / (sample_size + smoothing)
            conditional_rates.append(rate)
            details.append((feature, bucket, sample_size, rate))

        probability = (base_rate + sum(conditional_rates)) / (len(conditional_rates) + 1)
        return float(probability), details

    @staticmethod
    def _bucket(feature: str, value: Any) -> str:
        if pd.isna(value):
            return "missing"
        if feature in {"CandleDirection", "QQQRegime"}:
            return str(value)
        if feature == "EMA20AboveEMA50":
            return "above" if bool(value) else "below"
        if feature == "StreakLength":
            return str(int(value))

        boundaries = {
            "RSI14": (30, 45, 55, 70),
            "PercentB20": (0, 0.25, 0.5, 0.75, 1),
            "VolumeRatio20": (0.75, 1, 1.25, 1.75),
            "Return5": (-0.05, -0.02, 0.02, 0.05),
            "Return20": (-0.10, -0.04, 0.04, 0.10),
        }[feature]
        for boundary in boundaries:
            if float(value) < boundary:
                return f"<{boundary:g}"
        return f">={boundaries[-1]:g}"

    def _signal(
        self,
        ticker: str,
        as_of: pd.Timestamp,
        close: float,
        candidate: pd.Series,
        probability: float,
        details: list[tuple[str, str, int, float]],
    ) -> dict[str, Any]:
        evidence = ", ".join(
            f"{feature}={bucket} ({count} rows, {rate:.1%})"
            for feature, bucket, count, rate in details
        )
        reason = (
            f"Rolling one-year ({len(details)}-feature) smoothed empirical ranker: "
            f"RSI14={candidate['RSI14']:.1f}, %B20={candidate['PercentB20']:.2f}, "
            f"volume ratio={candidate['VolumeRatio20']:.2f}, 5d={candidate['Return5']:.1%}, "
            f"20d={candidate['Return20']:.1%}, EMA20 {'>' if candidate['EMA20AboveEMA50'] else '<='} EMA50, "
            f"QQQ={candidate['QQQRegime']}; bucket evidence: {evidence}."
        )
        return {
            "Ticker": ticker,
            "Strategy": self.name,
            "Direction": "LONG",
            "Priority": cfg.STRATEGY_PRIORITY.get(self.name, 4),
            "Close": round(close, 2),
            "Entry": round(close, 2),
            "EntryTiming": "NEXT_SESSION_OPEN",
            "StopLoss": round(close * 0.99, 2),
            "Target": round(close, 2),
            "Score": round(probability * 100, 2),
            "Volume": 0,
            "Date": as_of,
            "AsOfDate": as_of,
            "MaxDays": 1,
            "SignalType": "daily_next_green_rank",
            "CandleDirection": candidate["CandleDirection"],
            "StreakLength": int(candidate["StreakLength"]),
            "RSI14": round(float(candidate["RSI14"]), 2),
            "PercentB": round(float(candidate["PercentB20"]), 3),
            "VolumeRatio20": round(float(candidate["VolumeRatio20"]), 2),
            "Return5": round(float(candidate["Return5"]), 5),
            "Return20": round(float(candidate["Return20"]), 5),
            "EMA20": round(float(candidate["EMA20"]), 2),
            "EMA50": round(float(candidate["EMA50"]), 2),
            "QQQRegime": candidate["QQQRegime"],
            "ProbabilityNextGreen": round(probability, 4),
            "Prediction": "GREEN",
            "PredictionReason": reason,
        }

    @classmethod
    def _load_external_settings(cls) -> dict[str, Any]:
        from pathlib import Path

        path = Path(cls.EXTERNAL_SETTINGS_PATH)
        if not path.exists() and not download_file("config/settings.json", path):
            raise FileNotFoundError("Missing required settings file: config\\settings.json.")
        with path.open("r", encoding="utf-8") as handle:
            settings = json.load(handle)
        missing = sorted(cls.REQUIRED_EXTERNAL_KEYS - settings.keys())
        if missing:
            raise ValueError(f"Streak ranker config missing required settings keys: {missing}")
        return settings

    def get_exit_conditions(
        self,
        position: dict[str, Any],
        df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> Optional[dict[str, Any]]:
        if pd.Timestamp(current_date) < pd.Timestamp(position["entry_date"]) or df.empty:
            return None
        return {"reason": "NextSessionClose", "exit_price": float(df["Close"].iloc[-1])}
