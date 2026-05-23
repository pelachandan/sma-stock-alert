from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import src.config.settings as cfg
from src.analysis.rally_pattern_strategy import RallyPatternStrategy, _BacktestPosition
from src.data.market import get_historical_data
from src.storage.gcs import download_file
from src.strategies.base import BaseStrategy


class RallyPatternPosition(BaseStrategy):
    """Live daily scanner wrapper for the standalone rally-pattern engine."""

    name = "RallyPattern_Position"
    description = "Cross-sectional rally-pattern leader scan"
    CONFIG_PATH = Path("config") / "rally_pattern_config.json"
    REQUIRED_CONFIG_KEYS = {
        "live_config",
        "feature_config",
        "score_config",
        "entry_logic_config",
        "exit_logic_config",
        "strategy_config",
        "ranking_config",
    }

    @staticmethod
    def _debug_enabled() -> bool:
        return os.getenv("STOCK_ALERT_BACKTEST_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}

    def _debug(self, message: str) -> None:
        if self._debug_enabled():
            print(f"🔎 [rally] {message}")

    def __init__(self) -> None:
        super().__init__()
        config = self._load_required_config()
        self._validate_config(config)
        self.config = config
        self.live_config = dict(config["live_config"])
        self.strategy = RallyPatternStrategy(
            **dict(config["strategy_config"]),
            feature_config=dict(config["feature_config"]),
            score_config=dict(config["score_config"]),
            entry_logic_config=dict(config["entry_logic_config"]),
            exit_logic_config=dict(config["exit_logic_config"]),
            ranking_config=dict(config["ranking_config"]),
        )
        self.max_days = int(self.live_config["max_days"])
        self.target_r_multiple = float(self.live_config["target_r_multiple"])
        self.min_history_bars = int(self.live_config["min_history_bars"])
        self.max_signal_age_days = int(self.live_config["max_signal_age_days"])

    @classmethod
    def _load_required_config(cls) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="rally-pattern-config-") as tmp_dir:
            local_path = Path(tmp_dir) / "rally_pattern_config.json"
            if download_file("config/rally_pattern_config.json", local_path):
                with local_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)

        if cls.CONFIG_PATH.exists():
            with cls.CONFIG_PATH.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        raise FileNotFoundError(
            "Missing required rally config file: config/rally_pattern_config.json "
            "(expected locally or in GCS)."
        )

    @classmethod
    def _validate_config(cls, config: dict[str, Any]) -> None:
        missing = sorted(cls.REQUIRED_CONFIG_KEYS - set(config))
        if missing:
            raise ValueError(f"Rally pattern config missing required sections: {missing}")

        for key in cls.REQUIRED_CONFIG_KEYS:
            if not isinstance(config[key], dict):
                raise ValueError(f"Rally pattern config section '{key}' must be an object.")

        strategy_config = config["strategy_config"]
        if not strategy_config:
            raise ValueError("Rally pattern config strategy_config must not be empty.")
        if any(isinstance(value, str) and "..." in value for value in strategy_config.values()):
            raise ValueError("Rally pattern config strategy_config still contains placeholder values.")
        required_strategy_keys = set(RallyPatternStrategy.default_strategy_config())
        missing_strategy_keys = sorted(required_strategy_keys - set(strategy_config))
        if missing_strategy_keys:
            raise ValueError(
                "Rally pattern config strategy_config missing required keys: "
                f"{missing_strategy_keys}"
            )

    def scan(
        self,
        ticker: str,
        df: pd.DataFrame,
        as_of_date: pd.Timestamp = None,
    ) -> Optional[dict[str, Any]]:
        signals = self.run([ticker], as_of_date=as_of_date)
        return signals[0] if signals else None

    def run(
        self,
        tickers: list[str],
        as_of_date: pd.Timestamp = None,
    ) -> list[dict[str, Any]]:
        scan_date = pd.Timestamp(as_of_date) if as_of_date is not None else pd.Timestamp.today().normalize()
        universe = [str(ticker).upper() for ticker in tickers if str(ticker).strip()]
        for benchmark in self.strategy.BENCHMARK_TICKERS:
            if benchmark not in universe:
                universe.append(benchmark)

        self._debug(f"scan_date={scan_date.date()} requested={len(universe)}")

        raw_df = self._load_history_frame(universe, scan_date)
        if raw_df.empty:
            self._debug("raw history frame empty after local data load")
            return []
        self._debug(
            f"raw_rows={len(raw_df)} raw_tickers={raw_df['ticker'].nunique()} "
            f"date_range={pd.Timestamp(raw_df['Date'].min()).date()}..{pd.Timestamp(raw_df['Date'].max()).date()}"
        )

        candidates = self._latest_candidate_rows(raw_df, scan_date)
        self._debug(f"latest_candidates={len(candidates)}")
        signals: list[dict[str, Any]] = []
        for _, row in candidates.iterrows():
            signal = self._signal_from_row(row)
            if signal is not None:
                signals.append(signal)
        self._debug(f"signals_emitted={len(signals)}")
        return signals

    def build_signal_cache(
        self,
        tickers: list[str],
        scan_dates: list[pd.Timestamp] | pd.DatetimeIndex,
    ) -> dict[pd.Timestamp, list[dict[str, Any]]]:
        normalized_dates = sorted(pd.Timestamp(date) for date in scan_dates)
        if not normalized_dates:
            return {}

        universe = [str(ticker).upper() for ticker in tickers if str(ticker).strip()]
        for benchmark in self.strategy.BENCHMARK_TICKERS:
            if benchmark not in universe:
                universe.append(benchmark)

        last_scan_date = normalized_dates[-1]
        self._debug(
            f"precompute_start scan_dates={len(normalized_dates)} requested={len(universe)} "
            f"last_scan_date={last_scan_date.date()}"
        )
        raw_df = self._load_history_frame(universe, last_scan_date)
        empty_cache = {scan_date: [] for scan_date in normalized_dates}
        if raw_df.empty:
            self._debug("precompute raw history frame empty after local data load")
            return empty_cache

        self._debug(
            f"precompute_raw_rows={len(raw_df)} raw_tickers={raw_df['ticker'].nunique()} "
            f"date_range={pd.Timestamp(raw_df['Date'].min()).date()}..{pd.Timestamp(raw_df['Date'].max()).date()}"
        )
        raw_df = raw_df.sort_values(["ticker", "Date"]).reset_index(drop=True)
        raw_df["history_bar_count"] = raw_df.groupby("ticker", sort=False).cumcount() + 1
        ranked = self.strategy.rank_candidates(raw_df)
        if ranked.empty:
            self._debug("precompute rank_candidates returned no entry candidates")
            return empty_cache

        if "history_bar_count" in ranked.columns:
            ranked = ranked[ranked["history_bar_count"] >= self.min_history_bars].copy()
        else:
            history_counts = raw_df[["ticker", "Date", "history_bar_count"]]
            ranked = ranked.merge(history_counts, on=["ticker", "Date"], how="left")
            ranked = ranked[ranked["history_bar_count"].fillna(0) >= self.min_history_bars].copy()
        if ranked.empty:
            self._debug("precompute candidates suppressed by min_history_bars gating")
            return empty_cache

        candidates_by_date = {
            pd.Timestamp(signal_date): signal_rows.sort_values(
                by=["setup_priority", "score", "volume_ratio_20", "ticker"],
                ascending=[True, False, False, True],
            ).reset_index(drop=True)
            for signal_date, signal_rows in ranked.groupby("Date", sort=True)
        }
        signal_dates = sorted(candidates_by_date)
        self._debug(
            f"precompute_ranked_candidates={len(ranked)} "
            f"signal_dates={len(signal_dates)} first_signal_date={signal_dates[0].date()} "
            f"last_signal_date={signal_dates[-1].date()}"
        )

        cache: dict[pd.Timestamp, list[dict[str, Any]]] = {}
        latest_signal_date: pd.Timestamp | None = None
        next_signal_index = 0
        for scan_date in normalized_dates:
            while next_signal_index < len(signal_dates) and signal_dates[next_signal_index] <= scan_date:
                latest_signal_date = signal_dates[next_signal_index]
                next_signal_index += 1

            if latest_signal_date is None or (scan_date - latest_signal_date).days > self.max_signal_age_days:
                cache[scan_date] = []
                continue

            day_signals: list[dict[str, Any]] = []
            for _, row in candidates_by_date[latest_signal_date].iterrows():
                signal = self._signal_from_row(row)
                if signal is not None:
                    day_signals.append(signal)
            cache[scan_date] = day_signals

        reused_days = sum(1 for scan_date, signals in cache.items() if signals and scan_date not in candidates_by_date)
        self._debug(
            f"precompute_cache_ready cached_scan_dates={len(cache)} "
            f"fresh_signal_days={len(candidates_by_date)} reused_signal_days={reused_days}"
        )
        return cache

    def get_exit_conditions(
        self,
        position: dict[str, Any],
        df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> Optional[dict[str, Any]]:
        if df.empty:
            return None

        working = df.copy()
        if not isinstance(working.index, pd.DatetimeIndex):
            working.index = pd.to_datetime(working.index, errors="coerce")
            working = working[working.index.notna()]

        current_ts = pd.Timestamp(current_date)
        working = working[working.index <= current_ts].copy()
        if working.empty:
            return None

        ticker = str(position.get("ticker", "")).upper()
        if not ticker:
            return None

        raw_df = working.reset_index().rename(columns={"index": "Date"})
        raw_df["ticker"] = ticker
        scored = self.strategy.score_dataframe(raw_df)
        scored = self.strategy._augment_exit_support_columns(scored)
        current_row = scored.iloc[-1]

        entry_date = pd.Timestamp(position.get("entry_date"))
        trade_rows = scored[scored["Date"] >= entry_date]
        if trade_rows.empty:
            trade_rows = scored.iloc[[-1]]

        entry_score = float(position.get("entry_score", position.get("Score", current_row["score"])))
        setup_type = str(position.get("setup_type", position.get("SetupType", "none")))
        zone_support = float(
            position.get(
                "zone_support",
                position.get("ZoneSupport", self.strategy._entry_zone_support_level(current_row, setup_type)),
            )
        )
        backtest_position = _BacktestPosition(
            ticker=ticker,
            entry_date=entry_date,
            entry_price=float(position.get("entry_price", current_row["close"])),
            shares=float(position.get("shares", 0.0)),
            entry_score=entry_score,
            setup_type=setup_type,
            best_score=float(trade_rows["score"].max()),
            highest_close=float(trade_rows["close"].max()),
            has_new_high=bool((trade_rows["pct_from_20d_high"] >= 0).any()),
            score_improved=bool(float(trade_rows["score"].max()) > entry_score),
            days_held=max(len(trade_rows) - 1, 0),
            add_on_count=int(position.get("add_on_count", 0)),
            zone_support=zone_support,
        )

        exit_reason = self.strategy._exit_reason(current_row, backtest_position)
        if exit_reason is None:
            return None
        return {"reason": exit_reason, "exit_price": float(current_row["close"])}

    def _normalize_history_frame(
        self,
        ticker: str,
        df: pd.DataFrame,
        as_of_date: pd.Timestamp,
    ) -> tuple[Optional[pd.DataFrame], str]:
        if df is None or df.empty:
            return None, "missing_or_empty"

        local = df.copy()
        if not isinstance(local.index, pd.DatetimeIndex):
            local.index = pd.to_datetime(local.index, errors="coerce")
            local = local[local.index.notna()]

        local = local[local.index <= as_of_date].copy()
        if len(local) < self.min_history_bars:
            return None, "short_history"

        local = local.reset_index().rename(columns={"index": "Date"})
        local["ticker"] = ticker
        rename_map = {}
        for column in local.columns:
            lowered = str(column).strip().lower()
            if lowered in {"open", "high", "low", "close", "volume"}:
                rename_map[column] = lowered
        local = local.rename(columns=rename_map)
        required = {"Date", "ticker", "open", "high", "low", "close", "volume"}
        if not required.issubset(local.columns):
            return None, "missing_required_cols"
        return local[["Date", "ticker", "open", "high", "low", "close", "volume"]], "loaded"

    def _load_history_frame(self, tickers: list[str], as_of_date: pd.Timestamp) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        missing_or_empty = 0
        short_history = 0
        missing_required_cols = 0
        for ticker in tickers:
            df = get_historical_data(ticker)
            normalized, status = self._normalize_history_frame(ticker, df, as_of_date)
            if status == "missing_or_empty":
                missing_or_empty += 1
                continue
            if status == "short_history":
                short_history += 1
                continue
            if status == "missing_required_cols":
                missing_required_cols += 1
                continue
            if normalized is not None:
                frames.append(normalized)

        self._debug(
            "history_load "
            f"loaded={len(frames)} missing_or_empty={missing_or_empty} "
            f"short_history={short_history} missing_required_cols={missing_required_cols}"
        )
        if not frames:
            return pd.DataFrame(columns=["Date", "ticker", "open", "high", "low", "close", "volume"])
        return pd.concat(frames, ignore_index=True)

    def _latest_candidate_rows(self, raw_df: pd.DataFrame, scan_date: pd.Timestamp) -> pd.DataFrame:
        ranked = self.strategy.rank_candidates(raw_df)
        if ranked.empty:
            self._debug("rank_candidates returned no entry candidates")
            return ranked

        latest_signal_date = pd.Timestamp(ranked["Date"].max())
        if (scan_date - latest_signal_date).days > self.max_signal_age_days:
            self._debug(
                f"latest signal stale latest_signal_date={latest_signal_date.date()} "
                f"scan_date={scan_date.date()} max_signal_age_days={self.max_signal_age_days}"
            )
            return ranked.iloc[0:0]

        latest = ranked[ranked["Date"] == latest_signal_date].copy()
        setup_counts = latest["setup_type"].value_counts().to_dict() if "setup_type" in latest.columns else {}
        self._debug(
            f"ranked_candidates={len(ranked)} latest_date_candidates={len(latest)} "
            f"latest_signal_date={latest_signal_date.date()} setup_counts={setup_counts}"
        )
        return latest.sort_values(
            by=["setup_priority", "score", "volume_ratio_20", "ticker"],
            ascending=[True, False, False, True],
        ).reset_index(drop=True)

    def _signal_from_row(self, row: pd.Series) -> Optional[dict[str, Any]]:
        entry = float(row["close"])
        stop = float(row.get("entry_structural_support", 0.0))
        risk_per_share = float(row.get("entry_risk_per_share", 0.0))
        if entry <= 0:
            return None
        if stop <= 0 or stop >= entry:
            stop = entry - risk_per_share
        if stop <= 0 or stop >= entry:
            return None

        setup_type = str(row.get("setup_type", "none"))
        score = float(row["score"])
        leadership_stage = str(row.get("leadership_stage", "none"))
        if leadership_stage == "none":
            leadership_stage = "emerging" if setup_type.startswith("emerging_") else "confirmed"
        position_size_multiplier = self.strategy.starter_size_multiplier(setup_type)
        zone_support = float(self.strategy._entry_zone_support_level(row, setup_type))
        target = entry + (self.target_r_multiple * (entry - stop))
        signal_date = pd.Timestamp(row["Date"])

        return {
            "Ticker": str(row["ticker"]).upper(),
            "Strategy": self.name,
            "Direction": "LONG",
            "Priority": cfg.STRATEGY_PRIORITY.get(self.name, 3),
            "Price": round(entry, 2),
            "Close": round(entry, 2),
            "Entry": round(entry, 2),
            "StopLoss": round(stop, 2),
            "StopPrice": round(stop, 2),
            "Target": round(target, 2),
            "Score": round(score, 2),
            "EntryScore": round(score, 2),
            "SetupType": setup_type,
            "SignalType": setup_type,
            "LeadershipStage": leadership_stage,
            "PositionSizeMultiplier": round(position_size_multiplier, 4),
            "ZoneSupport": round(zone_support, 2),
            "Volume": int(float(row.get("volume", 0.0))),
            "Date": signal_date,
            "AsOfDate": signal_date,
            "MaxDays": self.max_days,
        }
