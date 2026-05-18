"""
Standalone daily ranking strategy for rally-pattern detection.

The strategy expects one row per ticker-date with precomputed technical columns.
It scores each row using the exact six-bucket model from the spec, derives labels
and pattern stages, emits entry/exit signals, ranks candidates cross-sectionally,
and runs a no-lookahead daily portfolio backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.trend_quality import (
    rolling_log_regression_r2,
    rolling_log_regression_slope,
    rolling_max_drawdown,
    rolling_positive_fraction,
)
from src.analysis.zone_structure import add_zone_columns, long_zone_broken


@dataclass(frozen=True)
class _BacktestPosition:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_score: float
    setup_type: str
    best_score: float
    highest_close: float
    has_new_high: bool
    score_improved: bool
    days_held: int
    add_on_count: int
    zone_support: float = 0.0


class RallyPatternStrategy:
    """Cross-sectional rally-ranking strategy operating on precomputed or raw features."""

    BENCHMARK_TICKERS: tuple[str, ...] = ("SPY", "QQQ")
    CONFIG_SECTION_PARAM_NAMES: frozenset[str] = frozenset(
        {"feature_config", "score_config", "entry_logic_config", "exit_logic_config", "ranking_config"}
    )
    COLUMN_DEFAULTS: dict[str, float] = {
        "close": 0.0,
        "close_vs_sma_10": 0.0,
        "close_vs_sma_20": 0.0,
        "close_vs_sma_50": 0.0,
        "close_vs_ema_10": 0.0,
        "close_vs_ema_20": 0.0,
        "close_vs_ema_50": 0.0,
        "trend_stack_bullish": 0.0,
        "trend_stack_bearish": 0.0,
        "pct_from_20d_high": -1.0,
        "pct_from_20d_low": 0.0,
        "donchian_pos_20": 0.0,
        "bb_pct_b_20": 0.0,
        "rsi_14": 0.0,
        "rsi_21": 0.0,
        "smoothed_rsi_ema21_rsi10": 0.0,
        "macd_line": 0.0,
        "macd_signal": 0.0,
        "macd_hist": 0.0,
        "stoch_k_14": 0.0,
        "stoch_d_3": 0.0,
        "williams_r_14": -100.0,
        "cci_20": 0.0,
        "adx_14": 0.0,
        "plus_di_14": 0.0,
        "minus_di_14": 0.0,
        "pct_chg": 0.0,
        "close_pos": 0.0,
        "body": 0.0,
        "tr": 0.0,
        "atr_14": 0.0,
        "atr_pct_14": 0.0,
        "realized_vol_20": 0.0,
        "volume_ratio_20": 0.0,
        "volume_ratio_50": 0.0,
        "volume_zscore_20": 0.0,
        "cmf_20": 0.0,
        "mfi_14": 0.0,
        "rs_spy_20": 0.0,
        "rs_spy_50": 0.0,
        "rs_qqq_20": 0.0,
        "rs_qqq_50": 0.0,
        "lr_slope_log_63": 0.0,
        "lr_slope_log_126": 0.0,
        "lr_r2_63": 0.0,
        "lr_r2_126": 0.0,
        "ema_50_slope_20": 0.0,
        "pct_above_ema_50_63": 0.0,
        "max_drawdown_63": 0.0,
        "base_width_20": 0.0,
        "base_tightness_20": 0.0,
        "rs_acceleration_3": 0.0,
        "confirmed_leader_score": 0.0,
        "confirmed_leader_regime_signal": 0.0,
        "emerging_leader_score": 0.0,
        "emerging_leader_regime_signal": 0.0,
        "leadership_score": 0.0,
        "leader_trend_score": 0.0,
        "leader_regime_signal": 0.0,
    }
    FEATURE_COLUMNS: tuple[str, ...] = tuple(COLUMN_DEFAULTS.keys())
    RAW_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    DEFAULT_FEATURE_CONFIG: dict[str, Any] = {
        "moving_average_periods": {"short": 10, "medium": 20, "long": 50},
        "volume_average_windows": {"short": 20, "long": 50},
        "breakout_window": 20,
        "donchian_window": 20,
        "bollinger_window": 20,
        "bollinger_std_mult": 2.0,
        "rsi_periods": {"fast": 14, "slow": 21},
        "smoothed_rsi": {"ema_span": 21, "rsi_period": 10},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "stochastic": {"window": 14, "signal": 3},
        "cci": {"window": 20, "constant": 0.015},
        "atr_window": 14,
        "directional_window": 14,
        "realized_vol": {"window": 20, "annualization": 252.0},
        "cmf_window": 20,
        "mfi_window": 14,
        "relative_strength_windows": {"short": 20, "long": 50},
        "zone_windows": {"short": 20, "long": 60},
        "structure_windows": {
            "prior_short": 3,
            "prior_medium": 5,
            "prior_long": 20,
            "tight_range": 5,
            "close_tightness": 3,
            "change_lookback": 3,
            "exit_break_low": 10,
            "confirmation_days": 2,
        },
        "broader_overhead_supply_buffer": 0.01,
        "trend_quality_windows": {
            "slope_short": 63,
            "slope_long": 126,
            "ema_slope": 20,
            "persistence": 63,
            "drawdown": 63,
        },
    }
    DEFAULT_SCORE_CONFIG: dict[str, Any] = {
        "trend": {
            "close_vs_sma_short_threshold": 0.0,
            "close_vs_sma_short_weight": 2,
            "close_vs_sma_medium_threshold": 0.0,
            "close_vs_sma_medium_weight": 4,
            "close_vs_sma_long_threshold": 0.0,
            "close_vs_sma_long_weight": 2,
            "close_vs_ema_short_threshold": 0.0,
            "close_vs_ema_short_weight": 2,
            "close_vs_ema_medium_threshold": 0.0,
            "close_vs_ema_medium_weight": 4,
            "close_vs_ema_long_threshold": 0.0,
            "close_vs_ema_long_weight": 2,
            "trend_stack_bullish_weight": 4,
            "trend_stack_bearish_penalty": 6,
            "max_points": 22,
        },
        "breakout": {
            "pct_from_high_pullback_min": -0.12,
            "pct_from_high_pullback_max": -0.01,
            "pct_from_high_pullback_weight": 5,
            "pct_from_high_breakout_min": -0.01,
            "pct_from_high_breakout_weight": 6,
            "pct_from_low_min": 0.05,
            "pct_from_low_weight": 2,
            "donchian_pos_min": 0.55,
            "donchian_pos_weight": 4,
            "donchian_pos_high_min": 0.80,
            "donchian_pos_high_weight": 2,
            "bb_pct_b_min": 0.55,
            "bb_pct_b_weight": 3,
            "bb_pct_b_high_min": 0.85,
            "bb_pct_b_high_weight": 2,
            "max_points": 18,
        },
        "momentum": {
            "rsi_fast_mid_min": 52.0,
            "rsi_fast_mid_weight": 3,
            "rsi_fast_high_min": 58.0,
            "rsi_fast_high_weight": 3,
            "rsi_fast_extreme_min": 65.0,
            "rsi_fast_extreme_weight": 2,
            "rsi_slow_min": 55.0,
            "rsi_slow_weight": 1,
            "smoothed_rsi_min": 55.0,
            "smoothed_rsi_weight": 1,
            "macd_hist_positive_weight": 3,
            "macd_line_above_signal_weight": 2,
            "stoch_k_mid_min": 65.0,
            "stoch_k_mid_weight": 1,
            "stoch_k_high_min": 80.0,
            "stoch_k_high_weight": 1,
            "stoch_d_min": 60.0,
            "stoch_d_weight": 1,
            "williams_r_min": -25.0,
            "williams_r_weight": 1,
            "cci_min": 50.0,
            "cci_weight": 1,
            "plus_di_above_minus_di_weight": 1,
            "adx_min": 20.0,
            "adx_weight": 1,
            "max_points": 18,
        },
        "flow": {
            "volume_ratio_short_min": 0.95,
            "volume_ratio_short_weight": 2,
            "volume_ratio_high_min": 1.15,
            "volume_ratio_high_weight": 3,
            "volume_ratio_long_min": 1.0,
            "volume_ratio_long_weight": 1,
            "volume_zscore_non_negative_weight": 1,
            "volume_zscore_high_min": 1.0,
            "volume_zscore_high_weight": 1,
            "cmf_non_negative_weight": 3,
            "cmf_high_min": 0.05,
            "cmf_high_weight": 2,
            "mfi_min": 50.0,
            "mfi_weight": 1,
            "pct_chg_positive_weight": 1,
            "close_pos_min": 0.60,
            "close_pos_weight": 1,
            "max_points": 16,
        },
        "relative_strength": {
            "rs_spy_short_positive_weight": 4,
            "rs_spy_long_positive_weight": 2,
            "rs_qqq_short_positive_weight": 4,
            "rs_qqq_long_positive_weight": 2,
            "rs_spy_short_strong_min": 0.03,
            "rs_spy_short_strong_weight": 2,
            "rs_qqq_short_strong_min": 0.03,
            "rs_qqq_short_strong_weight": 2,
            "max_points": 16,
        },
        "volatility": {
            "atr_pct_low_min": 0.012,
            "atr_pct_low_max": 0.040,
            "atr_pct_low_weight": 4,
            "atr_pct_high_max": 0.070,
            "atr_pct_high_weight": 2,
            "body_over_tr_min": 0.45,
            "body_over_tr_weight": 3,
            "close_pos_min": 0.70,
            "close_pos_weight": 2,
            "realized_vol_positive_weight": 1,
            "max_points": 10,
        },
        "penalties": {
            "trend_stack_bearish": 8,
            "minus_di_above_plus_di_and_rsi_below_50": 5,
            "deep_off_high_threshold": -0.15,
            "deep_off_high_with_negative_rs": 8,
            "weak_volume_ratio_threshold": 0.75,
            "weak_volume_and_negative_cmf": 4,
            "negative_macd_and_rsi_below_50": 6,
        },
        "label_cutoffs": {"A": 75, "B": 60, "C": 45},
        "pattern_stage": {"signal_score": 60, "breakout_pct_chg": 0.01},
    }
    DEFAULT_ENTRY_LOGIC_CONFIG: dict[str, Any] = {
        "breakout_base_min_rsi_14": 52.0,
        "breakout_base_min_donchian_pos": 0.75,
        "breakout_base_min_volume_ratio": 0.95,
        "breakout_strict_max_atr_pct_14": 0.055,
        "continuation_pct_from_20d_high_min": -0.10,
        "continuation_pct_from_20d_high_max": -0.003,
        "continuation_shelf_min_close_pos": 0.60,
        "continuation_shelf_min_pct_chg": 0.0,
        "continuation_pullback_min_volume_ratio_floor": 0.85,
        "continuation_pullback_max_volume_ratio_cap": 1.20,
        "continuation_pullback_min_pct_chg": -0.005,
        "continuation_strict_min_close_pos": 0.60,
        "emerging_shelf_min_close_pos": 0.58,
        "emerging_shelf_min_pct_chg": -0.002,
        "emerging_shelf_max_volume_ratio": 1.20,
        "late_stage_min_pct_chg": 0.0,
        "power_breakout_max_support_cluster_gap": 0.08,
        "power_breakout_strict_max_atr_pct_14": 0.08,
        "zone_entry_reclaim_buffer": 0.01,
    }
    DEFAULT_EXIT_LOGIC_CONFIG: dict[str, Any] = {
        "soft_score_fail_threshold": 35.0,
    }
    DEFAULT_RANKING_CONFIG: dict[str, Any] = {
        "setup_priority": {
            "power_breakout": 0,
            "expansion_leader": 1,
            "emerging_leader_ignition": 2,
            "leader_reentry": 3,
            "late_stage_leader": 4,
            "emerging_leader_breakout": 5,
            "breakout": 5,
            "emerging_leader_shelf": 6,
            "continuation_shelf": 7,
            "continuation_pullback": 8,
            "continuation": 9,
        }
    }

    @classmethod
    def default_strategy_config(cls) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for name, parameter in inspect.signature(cls.__init__).parameters.items():
            if name == "self" or name in cls.CONFIG_SECTION_PARAM_NAMES:
                continue
            if parameter.default is inspect.Signature.empty:
                raise ValueError(f"RallyPatternStrategy parameter '{name}' must define a default value.")
            defaults[name] = parameter.default
        return defaults

    def __init__(
        self,
        *,
        strict_entry: bool = False,
        use_atr_stop: bool = False,
        use_time_stop: bool = False,
        allocation_mode: str = "baseline",
        enable_risk_position_sizing: bool = False,
        enable_leader_reentry: bool = False,
        enable_late_stage_leaders: bool = False,
        reentry_cooldown_days: int = 3,
        time_stop_days: int = 15,
        setup_score_threshold: float = 70.0,
        trigger_window_days: int = 5,
        trigger_volume_ratio: float = 1.15,
        min_setup_days: int = 2,
        max_setup_extension: float = 0.06,
        breakout_min_combined_rs: float = 0.09,
        breakout_min_close_pos: float = 0.70,
        breakout_max_tight_range_5: float = 0.09,
        breakout_max_close_to_prior_20bar_high: float = 0.01,
        breakout_reclaim_exception_volume_ratio: float = 2.0,
        breakout_extended_max_close_vs_ema_20: float = 0.055,
        breakout_extended_min_volume_ratio: float = 2.0,
        breakout_extended_min_close_pos: float = 0.90,
        max_trigger_extension: float = 0.08,
        max_setup_rsi_14: float = 72.0,
        max_trigger_rsi_14: float = 75.0,
        max_setup_donchian_pos: float = 0.90,
        continuation_score_threshold: float = 75.0,
        continuation_trigger_volume_ratio: float = 0.90,
        continuation_min_close_vs_ema_20: float = 0.01,
        continuation_max_close_vs_ema_20: float = 0.05,
        continuation_min_rs_spy_20: float = 0.02,
        continuation_min_rs_qqq_20: float = 0.03,
        continuation_min_combined_rs: float = 0.07,
        continuation_min_rs_qqq_change_3: float = -1.0,
        continuation_max_atr_pct_change_3: float = 1.0,
        continuation_min_rsi_14: float = 55.0,
        continuation_max_rsi_14: float = 70.0,
        continuation_min_donchian_pos: float = 0.65,
        continuation_max_donchian_pos: float = 0.95,
        continuation_min_volume_ratio: float = 0.75,
        continuation_max_volume_ratio: float = 1.10,
        continuation_max_tight_range_5: float = 0.07,
        continuation_max_close_tightness_3: float = 0.0155,
        continuation_max_atr_pct_14: float = 0.055,
        continuation_max_trigger_extension: float = 0.06,
        continuation_shelf_max_close_vs_ema_20: float = 0.04,
        continuation_shelf_max_tight_range_5: float = 0.055,
        continuation_shelf_max_close_tightness_3: float = 0.0125,
        continuation_shelf_max_atr_pct_14: float = 0.04,
        continuation_shelf_max_support_gap: float = 0.045,
        continuation_pullback_score_threshold: float = 80.0,
        continuation_pullback_min_atr_pct_14: float = 0.03,
        continuation_pullback_max_tight_range_5: float = 0.085,
        continuation_pullback_max_close_tightness_3: float = 0.02,
        continuation_pullback_min_close_pos: float = 0.55,
        continuation_pullback_max_support_gap: float = 0.06,
        super_leader_score_threshold: float = 90.0,
        super_leader_min_combined_rs: float = 0.12,
        super_leader_min_close_vs_sma_50: float = 0.03,
        super_leader_min_donchian_pos: float = 0.85,
        super_leader_lookback_days: int = 20,
        leader_min_combined_rs: float = 0.06,
        leader_min_lr_slope_log_63: float = 0.0015,
        leader_min_lr_slope_log_126: float = 0.0010,
        leader_min_lr_r2_63: float = 0.55,
        leader_min_lr_r2_126: float = 0.45,
        leader_min_ema_50_slope_20: float = 0.0008,
        leader_min_pct_above_ema_50_63: float = 0.65,
        leader_max_drawdown_63: float = 0.18,
        leader_regime_min_score: float = 10.0,
        emerging_leader_score_threshold: float = 65.0,
        emerging_leader_min_combined_rs: float = 0.03,
        emerging_leader_min_lr_slope_log_63: float = 0.0002,
        emerging_leader_min_lr_r2_63: float = 0.15,
        emerging_leader_min_ema_50_slope_20: float = -0.0002,
        emerging_leader_min_pct_above_ema_50_63: float = 0.35,
        emerging_leader_max_drawdown_63: float = 0.28,
        emerging_leader_min_rs_acceleration_3: float = 0.001,
        emerging_leader_min_score: float = 7.0,
        emerging_leader_max_base_width_20: float = 0.22,
        emerging_leader_max_base_tightness_20: float = 0.90,
        emerging_leader_breakout_min_volume_ratio: float = 1.15,
        emerging_leader_breakout_min_pct_chg: float = 0.010,
        emerging_leader_breakout_max_close_vs_ema_20: float = 0.08,
        emerging_leader_breakout_min_close_vs_sma_50: float = -0.02,
        emerging_leader_breakout_min_close_pos: float = 0.72,
        emerging_leader_ignition_score_threshold: float = 80.0,
        emerging_leader_ignition_min_combined_rs: float = 0.08,
        emerging_leader_ignition_min_pct_chg: float = 0.04,
        emerging_leader_ignition_min_volume_ratio: float = 1.75,
        emerging_leader_ignition_min_close_pos: float = 0.75,
        emerging_leader_ignition_min_close_vs_ema_20: float = 0.08,
        emerging_leader_ignition_max_close_vs_ema_20: float = 0.22,
        emerging_leader_ignition_min_close_vs_sma_50: float = 0.0,
        emerging_leader_ignition_min_donchian_pos: float = 0.88,
        emerging_leader_ignition_max_support_gap: float = 0.20,
        emerging_leader_ignition_max_close_to_prior_20bar_high: float = 0.12,
        emerging_leader_shelf_min_close_vs_ema_20: float = -0.01,
        emerging_leader_shelf_max_close_vs_ema_20: float = 0.06,
        emerging_leader_shelf_min_close_vs_sma_50: float = -0.02,
        emerging_leader_shelf_max_tight_range_5: float = 0.07,
        emerging_leader_shelf_max_close_tightness_3: float = 0.02,
        emerging_leader_shelf_max_support_gap: float = 0.06,
        emerging_leader_shelf_trigger_volume_ratio: float = 0.95,
        emerging_leader_max_trigger_extension: float = 0.08,
        expansion_leader_score_threshold: float = 95.0,
        expansion_leader_min_combined_rs: float = 0.18,
        expansion_leader_min_close_vs_ema_20: float = 0.06,
        expansion_leader_max_close_vs_ema_20: float = 0.22,
        expansion_leader_min_pct_chg: float = 0.035,
        expansion_leader_min_volume_ratio: float = 1.20,
        expansion_leader_min_close_pos: float = 0.75,
        expansion_leader_min_donchian_pos: float = 0.92,
        expansion_leader_min_atr_pct_14: float = 0.025,
        expansion_leader_max_tight_range_5: float = 0.18,
        expansion_leader_max_close_to_prior_20bar_high: float = 0.02,
        expansion_leader_reclaim_exception_volume_ratio: float = 2.0,
        leader_reentry_score_threshold: float = 90.0,
        leader_reentry_min_combined_rs: float = 0.12,
        leader_reentry_min_close_vs_ema_20: float = 0.02,
        leader_reentry_max_close_vs_ema_20: float = 0.08,
        leader_reentry_min_pct_chg: float = 0.010,
        leader_reentry_min_volume_ratio: float = 1.05,
        leader_reentry_min_close_pos: float = 0.70,
        leader_reentry_min_donchian_pos: float = 0.84,
        leader_reentry_max_tight_range_5: float = 0.08,
        leader_reentry_max_support_gap: float = 0.045,
        leader_reentry_min_close_to_prior_20bar_high: float = -0.015,
        late_stage_leader_score_threshold: float = 90.0,
        late_stage_leader_min_combined_rs: float = 0.14,
        late_stage_leader_min_close_vs_ema_20: float = 0.03,
        late_stage_leader_max_close_vs_ema_20: float = 0.13,
        late_stage_leader_min_close_pos: float = 0.62,
        late_stage_leader_min_donchian_pos: float = 0.82,
        late_stage_leader_min_volume_ratio: float = 0.85,
        late_stage_leader_max_tight_range_5: float = 0.09,
        late_stage_leader_max_close_tightness_3: float = 0.02,
        late_stage_leader_max_support_gap: float = 0.04,
        late_stage_leader_min_close_to_prior_20bar_high: float = -0.01,
        power_breakout_score_threshold: float = 85.0,
        power_breakout_trigger_volume_ratio: float = 1.30,
        power_breakout_min_close_vs_ema_20: float = 0.025,
        power_breakout_max_close_vs_ema_20: float = 0.12,
        power_breakout_min_rsi_14: float = 60.0,
        power_breakout_max_rsi_14: float = 82.0,
        power_breakout_min_donchian_pos: float = 0.88,
        power_breakout_min_close_pos: float = 0.70,
        power_breakout_min_pct_chg: float = 0.012,
        power_breakout_min_rs_spy_20: float = 0.015,
        power_breakout_min_rs_qqq_20: float = 0.015,
        power_breakout_min_pct_from_20d_high: float = -0.01,
        power_breakout_min_close_pos_strict: float = 0.80,
        power_breakout_min_combined_rs: float = 0.10,
        power_breakout_max_tight_range_5: float = 0.11,
        power_breakout_calm_max_close_vs_ema_20: float = 0.07,
        power_breakout_calm_max_atr_pct_14: float = 0.045,
        power_breakout_explosive_min_pct_chg: float = 0.020,
        power_breakout_explosive_min_volume_ratio: float = 1.75,
        zone_max_width_20: float = 0.20,
        zone_reclaim_distance_20: float = 0.04,
        zone_seller_fraction: float = 0.2,
        zone_demand_fraction: float = 0.2,
        zone_exit_tolerance_pct: float = 0.002,
        breakout_min_room_to_60bar_high: float = 0.01,
        continuation_shelf_min_room_to_60bar_high: float = 0.03,
        continuation_pullback_min_room_to_60bar_high: float = 0.02,
        expansion_leader_min_room_to_60bar_high: float = 0.0,
        leader_reentry_min_room_to_60bar_high: float = 0.01,
        late_stage_leader_min_room_to_60bar_high: float = 0.01,
        power_breakout_min_room_to_60bar_high: float = 0.0,
        emerging_leader_breakout_min_room_to_60bar_high: float = 0.02,
        emerging_leader_ignition_min_room_to_60bar_high: float = 0.0,
        emerging_leader_shelf_min_room_to_60bar_high: float = 0.02,
        enable_aggressive_early_failure: bool = False,
        enable_bb_micro_failure: bool = False,
        enable_medium_confirm_failure: bool = False,
        enable_aggressive_starter_sizing: bool = False,
        aggressive_starter_fraction: float = 0.5,
        emerging_leader_ignition_starter_fraction: float = 0.5,
        aggressive_add_on_size_fraction: float = 1.0,
        max_add_ons_per_ticker: int = 1,
        add_on_profit_threshold: float = 0.08,
        add_on_size_fraction: float = 0.50,
        add_on_min_score: float = 85.0,
        trend_hold_entry_score_threshold: float = 90.0,
        trend_hold_gain_threshold: float = 0.10,
        trend_hold_min_score: float = 75.0,
        trend_hold_min_days: int = 20,
        trend_hold_trailing_atr_multiple: float = 4.0,
        trend_hold_relative_weak_exit_score: float = 60.0,
        breakout_early_failure_max_days: int = 4,
        breakout_early_failure_max_score: float = 70.0,
        breakout_early_failure_min_close_vs_ema_20: float = 0.02,
        breakout_early_failure_max_open_gain: float = 0.01,
        expansion_leader_early_failure_max_days: int = 3,
        expansion_leader_early_failure_max_score: float = 80.0,
        expansion_leader_early_failure_min_close_vs_ema_20: float = 0.06,
        expansion_leader_early_failure_max_open_gain: float = 0.0,
        expansion_leader_early_failure_min_close_pos: float = 0.60,
        power_breakout_early_failure_max_days: int = 3,
        power_breakout_early_failure_max_score: float = 80.0,
        power_breakout_early_failure_min_close_vs_ema_20: float = 0.03,
        power_breakout_early_failure_max_open_gain: float = 0.0,
        power_breakout_early_failure_min_close_pos: float = 0.55,
        bb_micro_failure_max_days: int = 5,
        bb_micro_failure_max_close_pos: float = 0.50,
        bb_micro_failure_max_bb_pct_b_20: float = 0.60,
        medium_confirm_failure_max_days: int = 5,
        medium_confirm_failure_max_close_pos: float = 0.45,
        trailing_stop_min_gain_to_arm: float = 0.08,
        trailing_stop_min_days_to_arm: int = 5,
        breakout_trailing_atr_multiple: float = 3.0,
        continuation_shelf_trailing_atr_multiple: float = 2.75,
        continuation_pullback_trailing_atr_multiple: float = 3.25,
        expansion_leader_trailing_atr_multiple: float = 5.0,
        power_breakout_trailing_atr_multiple: float = 4.5,
        portfolio_risk_per_trade: float = 2_000.0,
        max_allocation_per_stock: float = 50_000.0,
        max_position_weight: float = 1.0,
        equal_weight_allocation_cap: float = 0.35,
        tiered_weight_power_breakout: float = 0.35,
        tiered_weight_expansion_leader: float = 0.35,
        tiered_weight_late_stage_leader: float = 0.28,
        tiered_weight_breakout: float = 0.22,
        tiered_weight_continuation_shelf: float = 0.18,
        tiered_weight_continuation_pullback: float = 0.16,
        tiered_weight_leader_reentry: float = 0.18,
        tiered_weight_emerging_leader_ignition: float = 0.18,
        tiered_weight_emerging_leader_breakout: float = 0.16,
        tiered_weight_emerging_leader_shelf: float = 0.14,
        tiered_weight_default: float = 0.20,
        min_stop_atr_multiple: float = 1.75,
        min_stop_pct: float = 0.035,
        feature_config: dict[str, Any] | None = None,
        score_config: dict[str, Any] | None = None,
        entry_logic_config: dict[str, Any] | None = None,
        exit_logic_config: dict[str, Any] | None = None,
        ranking_config: dict[str, Any] | None = None,
    ) -> None:
        self.strict_entry = strict_entry
        self.use_atr_stop = use_atr_stop
        self.use_time_stop = use_time_stop
        self.allocation_mode = allocation_mode
        self.enable_risk_position_sizing = enable_risk_position_sizing
        self.enable_leader_reentry = enable_leader_reentry
        self.enable_late_stage_leaders = enable_late_stage_leaders
        self.reentry_cooldown_days = reentry_cooldown_days
        self.time_stop_days = time_stop_days
        self.setup_score_threshold = setup_score_threshold
        self.trigger_window_days = trigger_window_days
        self.trigger_volume_ratio = trigger_volume_ratio
        self.min_setup_days = min_setup_days
        self.max_setup_extension = max_setup_extension
        self.breakout_min_combined_rs = breakout_min_combined_rs
        self.breakout_min_close_pos = breakout_min_close_pos
        self.breakout_max_tight_range_5 = breakout_max_tight_range_5
        self.breakout_max_close_to_prior_20bar_high = breakout_max_close_to_prior_20bar_high
        self.breakout_reclaim_exception_volume_ratio = breakout_reclaim_exception_volume_ratio
        self.breakout_extended_max_close_vs_ema_20 = breakout_extended_max_close_vs_ema_20
        self.breakout_extended_min_volume_ratio = breakout_extended_min_volume_ratio
        self.breakout_extended_min_close_pos = breakout_extended_min_close_pos
        self.max_trigger_extension = max_trigger_extension
        self.max_setup_rsi_14 = max_setup_rsi_14
        self.max_trigger_rsi_14 = max_trigger_rsi_14
        self.max_setup_donchian_pos = max_setup_donchian_pos
        self.continuation_score_threshold = continuation_score_threshold
        self.continuation_trigger_volume_ratio = continuation_trigger_volume_ratio
        self.continuation_min_close_vs_ema_20 = continuation_min_close_vs_ema_20
        self.continuation_max_close_vs_ema_20 = continuation_max_close_vs_ema_20
        self.continuation_min_rs_spy_20 = continuation_min_rs_spy_20
        self.continuation_min_rs_qqq_20 = continuation_min_rs_qqq_20
        self.continuation_min_combined_rs = continuation_min_combined_rs
        self.continuation_min_rs_qqq_change_3 = continuation_min_rs_qqq_change_3
        self.continuation_max_atr_pct_change_3 = continuation_max_atr_pct_change_3
        self.continuation_min_rsi_14 = continuation_min_rsi_14
        self.continuation_max_rsi_14 = continuation_max_rsi_14
        self.continuation_min_donchian_pos = continuation_min_donchian_pos
        self.continuation_max_donchian_pos = continuation_max_donchian_pos
        self.continuation_min_volume_ratio = continuation_min_volume_ratio
        self.continuation_max_volume_ratio = continuation_max_volume_ratio
        self.continuation_max_tight_range_5 = continuation_max_tight_range_5
        self.continuation_max_close_tightness_3 = continuation_max_close_tightness_3
        self.continuation_max_atr_pct_14 = continuation_max_atr_pct_14
        self.continuation_max_trigger_extension = continuation_max_trigger_extension
        self.continuation_shelf_max_close_vs_ema_20 = continuation_shelf_max_close_vs_ema_20
        self.continuation_shelf_max_tight_range_5 = continuation_shelf_max_tight_range_5
        self.continuation_shelf_max_close_tightness_3 = continuation_shelf_max_close_tightness_3
        self.continuation_shelf_max_atr_pct_14 = continuation_shelf_max_atr_pct_14
        self.continuation_shelf_max_support_gap = continuation_shelf_max_support_gap
        self.continuation_pullback_score_threshold = continuation_pullback_score_threshold
        self.continuation_pullback_min_atr_pct_14 = continuation_pullback_min_atr_pct_14
        self.continuation_pullback_max_tight_range_5 = continuation_pullback_max_tight_range_5
        self.continuation_pullback_max_close_tightness_3 = continuation_pullback_max_close_tightness_3
        self.continuation_pullback_min_close_pos = continuation_pullback_min_close_pos
        self.continuation_pullback_max_support_gap = continuation_pullback_max_support_gap
        self.super_leader_score_threshold = super_leader_score_threshold
        self.super_leader_min_combined_rs = super_leader_min_combined_rs
        self.super_leader_min_close_vs_sma_50 = super_leader_min_close_vs_sma_50
        self.super_leader_min_donchian_pos = super_leader_min_donchian_pos
        self.super_leader_lookback_days = super_leader_lookback_days
        self.leader_min_combined_rs = leader_min_combined_rs
        self.leader_min_lr_slope_log_63 = leader_min_lr_slope_log_63
        self.leader_min_lr_slope_log_126 = leader_min_lr_slope_log_126
        self.leader_min_lr_r2_63 = leader_min_lr_r2_63
        self.leader_min_lr_r2_126 = leader_min_lr_r2_126
        self.leader_min_ema_50_slope_20 = leader_min_ema_50_slope_20
        self.leader_min_pct_above_ema_50_63 = leader_min_pct_above_ema_50_63
        self.leader_max_drawdown_63 = leader_max_drawdown_63
        self.leader_regime_min_score = leader_regime_min_score
        self.emerging_leader_score_threshold = emerging_leader_score_threshold
        self.emerging_leader_min_combined_rs = emerging_leader_min_combined_rs
        self.emerging_leader_min_lr_slope_log_63 = emerging_leader_min_lr_slope_log_63
        self.emerging_leader_min_lr_r2_63 = emerging_leader_min_lr_r2_63
        self.emerging_leader_min_ema_50_slope_20 = emerging_leader_min_ema_50_slope_20
        self.emerging_leader_min_pct_above_ema_50_63 = emerging_leader_min_pct_above_ema_50_63
        self.emerging_leader_max_drawdown_63 = emerging_leader_max_drawdown_63
        self.emerging_leader_min_rs_acceleration_3 = emerging_leader_min_rs_acceleration_3
        self.emerging_leader_min_score = emerging_leader_min_score
        self.emerging_leader_max_base_width_20 = emerging_leader_max_base_width_20
        self.emerging_leader_max_base_tightness_20 = emerging_leader_max_base_tightness_20
        self.emerging_leader_breakout_min_volume_ratio = emerging_leader_breakout_min_volume_ratio
        self.emerging_leader_breakout_min_pct_chg = emerging_leader_breakout_min_pct_chg
        self.emerging_leader_breakout_max_close_vs_ema_20 = emerging_leader_breakout_max_close_vs_ema_20
        self.emerging_leader_breakout_min_close_vs_sma_50 = emerging_leader_breakout_min_close_vs_sma_50
        self.emerging_leader_breakout_min_close_pos = emerging_leader_breakout_min_close_pos
        self.emerging_leader_ignition_score_threshold = emerging_leader_ignition_score_threshold
        self.emerging_leader_ignition_min_combined_rs = emerging_leader_ignition_min_combined_rs
        self.emerging_leader_ignition_min_pct_chg = emerging_leader_ignition_min_pct_chg
        self.emerging_leader_ignition_min_volume_ratio = emerging_leader_ignition_min_volume_ratio
        self.emerging_leader_ignition_min_close_pos = emerging_leader_ignition_min_close_pos
        self.emerging_leader_ignition_min_close_vs_ema_20 = emerging_leader_ignition_min_close_vs_ema_20
        self.emerging_leader_ignition_max_close_vs_ema_20 = emerging_leader_ignition_max_close_vs_ema_20
        self.emerging_leader_ignition_min_close_vs_sma_50 = emerging_leader_ignition_min_close_vs_sma_50
        self.emerging_leader_ignition_min_donchian_pos = emerging_leader_ignition_min_donchian_pos
        self.emerging_leader_ignition_max_support_gap = emerging_leader_ignition_max_support_gap
        self.emerging_leader_ignition_max_close_to_prior_20bar_high = emerging_leader_ignition_max_close_to_prior_20bar_high
        self.emerging_leader_shelf_min_close_vs_ema_20 = emerging_leader_shelf_min_close_vs_ema_20
        self.emerging_leader_shelf_max_close_vs_ema_20 = emerging_leader_shelf_max_close_vs_ema_20
        self.emerging_leader_shelf_min_close_vs_sma_50 = emerging_leader_shelf_min_close_vs_sma_50
        self.emerging_leader_shelf_max_tight_range_5 = emerging_leader_shelf_max_tight_range_5
        self.emerging_leader_shelf_max_close_tightness_3 = emerging_leader_shelf_max_close_tightness_3
        self.emerging_leader_shelf_max_support_gap = emerging_leader_shelf_max_support_gap
        self.emerging_leader_shelf_trigger_volume_ratio = emerging_leader_shelf_trigger_volume_ratio
        self.emerging_leader_max_trigger_extension = emerging_leader_max_trigger_extension
        self.expansion_leader_score_threshold = expansion_leader_score_threshold
        self.expansion_leader_min_combined_rs = expansion_leader_min_combined_rs
        self.expansion_leader_min_close_vs_ema_20 = expansion_leader_min_close_vs_ema_20
        self.expansion_leader_max_close_vs_ema_20 = expansion_leader_max_close_vs_ema_20
        self.expansion_leader_min_pct_chg = expansion_leader_min_pct_chg
        self.expansion_leader_min_volume_ratio = expansion_leader_min_volume_ratio
        self.expansion_leader_min_close_pos = expansion_leader_min_close_pos
        self.expansion_leader_min_donchian_pos = expansion_leader_min_donchian_pos
        self.expansion_leader_min_atr_pct_14 = expansion_leader_min_atr_pct_14
        self.expansion_leader_max_tight_range_5 = expansion_leader_max_tight_range_5
        self.expansion_leader_max_close_to_prior_20bar_high = expansion_leader_max_close_to_prior_20bar_high
        self.expansion_leader_reclaim_exception_volume_ratio = expansion_leader_reclaim_exception_volume_ratio
        self.leader_reentry_score_threshold = leader_reentry_score_threshold
        self.leader_reentry_min_combined_rs = leader_reentry_min_combined_rs
        self.leader_reentry_min_close_vs_ema_20 = leader_reentry_min_close_vs_ema_20
        self.leader_reentry_max_close_vs_ema_20 = leader_reentry_max_close_vs_ema_20
        self.leader_reentry_min_pct_chg = leader_reentry_min_pct_chg
        self.leader_reentry_min_volume_ratio = leader_reentry_min_volume_ratio
        self.leader_reentry_min_close_pos = leader_reentry_min_close_pos
        self.leader_reentry_min_donchian_pos = leader_reentry_min_donchian_pos
        self.leader_reentry_max_tight_range_5 = leader_reentry_max_tight_range_5
        self.leader_reentry_max_support_gap = leader_reentry_max_support_gap
        self.leader_reentry_min_close_to_prior_20bar_high = leader_reentry_min_close_to_prior_20bar_high
        self.late_stage_leader_score_threshold = late_stage_leader_score_threshold
        self.late_stage_leader_min_combined_rs = late_stage_leader_min_combined_rs
        self.late_stage_leader_min_close_vs_ema_20 = late_stage_leader_min_close_vs_ema_20
        self.late_stage_leader_max_close_vs_ema_20 = late_stage_leader_max_close_vs_ema_20
        self.late_stage_leader_min_close_pos = late_stage_leader_min_close_pos
        self.late_stage_leader_min_donchian_pos = late_stage_leader_min_donchian_pos
        self.late_stage_leader_min_volume_ratio = late_stage_leader_min_volume_ratio
        self.late_stage_leader_max_tight_range_5 = late_stage_leader_max_tight_range_5
        self.late_stage_leader_max_close_tightness_3 = late_stage_leader_max_close_tightness_3
        self.late_stage_leader_max_support_gap = late_stage_leader_max_support_gap
        self.late_stage_leader_min_close_to_prior_20bar_high = late_stage_leader_min_close_to_prior_20bar_high
        self.power_breakout_score_threshold = power_breakout_score_threshold
        self.power_breakout_trigger_volume_ratio = power_breakout_trigger_volume_ratio
        self.power_breakout_min_close_vs_ema_20 = power_breakout_min_close_vs_ema_20
        self.power_breakout_max_close_vs_ema_20 = power_breakout_max_close_vs_ema_20
        self.power_breakout_min_rsi_14 = power_breakout_min_rsi_14
        self.power_breakout_max_rsi_14 = power_breakout_max_rsi_14
        self.power_breakout_min_donchian_pos = power_breakout_min_donchian_pos
        self.power_breakout_min_close_pos = power_breakout_min_close_pos
        self.power_breakout_min_pct_chg = power_breakout_min_pct_chg
        self.power_breakout_min_rs_spy_20 = power_breakout_min_rs_spy_20
        self.power_breakout_min_rs_qqq_20 = power_breakout_min_rs_qqq_20
        self.power_breakout_min_pct_from_20d_high = power_breakout_min_pct_from_20d_high
        self.power_breakout_min_close_pos_strict = power_breakout_min_close_pos_strict
        self.power_breakout_min_combined_rs = power_breakout_min_combined_rs
        self.power_breakout_max_tight_range_5 = power_breakout_max_tight_range_5
        self.power_breakout_calm_max_close_vs_ema_20 = power_breakout_calm_max_close_vs_ema_20
        self.power_breakout_calm_max_atr_pct_14 = power_breakout_calm_max_atr_pct_14
        self.power_breakout_explosive_min_pct_chg = power_breakout_explosive_min_pct_chg
        self.power_breakout_explosive_min_volume_ratio = power_breakout_explosive_min_volume_ratio
        self.zone_max_width_20 = zone_max_width_20
        self.zone_reclaim_distance_20 = zone_reclaim_distance_20
        self.zone_seller_fraction = zone_seller_fraction
        self.zone_demand_fraction = zone_demand_fraction
        self.zone_exit_tolerance_pct = zone_exit_tolerance_pct
        self.breakout_min_room_to_60bar_high = breakout_min_room_to_60bar_high
        self.continuation_shelf_min_room_to_60bar_high = continuation_shelf_min_room_to_60bar_high
        self.continuation_pullback_min_room_to_60bar_high = continuation_pullback_min_room_to_60bar_high
        self.expansion_leader_min_room_to_60bar_high = expansion_leader_min_room_to_60bar_high
        self.leader_reentry_min_room_to_60bar_high = leader_reentry_min_room_to_60bar_high
        self.late_stage_leader_min_room_to_60bar_high = late_stage_leader_min_room_to_60bar_high
        self.power_breakout_min_room_to_60bar_high = power_breakout_min_room_to_60bar_high
        self.emerging_leader_breakout_min_room_to_60bar_high = emerging_leader_breakout_min_room_to_60bar_high
        self.emerging_leader_ignition_min_room_to_60bar_high = emerging_leader_ignition_min_room_to_60bar_high
        self.emerging_leader_shelf_min_room_to_60bar_high = emerging_leader_shelf_min_room_to_60bar_high
        self.enable_aggressive_early_failure = enable_aggressive_early_failure
        self.enable_bb_micro_failure = enable_bb_micro_failure
        self.enable_medium_confirm_failure = enable_medium_confirm_failure
        self.enable_aggressive_starter_sizing = enable_aggressive_starter_sizing
        self.aggressive_starter_fraction = aggressive_starter_fraction
        self.emerging_leader_ignition_starter_fraction = emerging_leader_ignition_starter_fraction
        self.aggressive_add_on_size_fraction = aggressive_add_on_size_fraction
        self.max_add_ons_per_ticker = max_add_ons_per_ticker
        self.add_on_profit_threshold = add_on_profit_threshold
        self.add_on_size_fraction = add_on_size_fraction
        self.add_on_min_score = add_on_min_score
        self.trend_hold_entry_score_threshold = trend_hold_entry_score_threshold
        self.trend_hold_gain_threshold = trend_hold_gain_threshold
        self.trend_hold_min_score = trend_hold_min_score
        self.trend_hold_min_days = trend_hold_min_days
        self.trend_hold_trailing_atr_multiple = trend_hold_trailing_atr_multiple
        self.trend_hold_relative_weak_exit_score = trend_hold_relative_weak_exit_score
        self.breakout_early_failure_max_days = breakout_early_failure_max_days
        self.breakout_early_failure_max_score = breakout_early_failure_max_score
        self.breakout_early_failure_min_close_vs_ema_20 = breakout_early_failure_min_close_vs_ema_20
        self.breakout_early_failure_max_open_gain = breakout_early_failure_max_open_gain
        self.expansion_leader_early_failure_max_days = expansion_leader_early_failure_max_days
        self.expansion_leader_early_failure_max_score = expansion_leader_early_failure_max_score
        self.expansion_leader_early_failure_min_close_vs_ema_20 = expansion_leader_early_failure_min_close_vs_ema_20
        self.expansion_leader_early_failure_max_open_gain = expansion_leader_early_failure_max_open_gain
        self.expansion_leader_early_failure_min_close_pos = expansion_leader_early_failure_min_close_pos
        self.power_breakout_early_failure_max_days = power_breakout_early_failure_max_days
        self.power_breakout_early_failure_max_score = power_breakout_early_failure_max_score
        self.power_breakout_early_failure_min_close_vs_ema_20 = power_breakout_early_failure_min_close_vs_ema_20
        self.power_breakout_early_failure_max_open_gain = power_breakout_early_failure_max_open_gain
        self.power_breakout_early_failure_min_close_pos = power_breakout_early_failure_min_close_pos
        self.bb_micro_failure_max_days = bb_micro_failure_max_days
        self.bb_micro_failure_max_close_pos = bb_micro_failure_max_close_pos
        self.bb_micro_failure_max_bb_pct_b_20 = bb_micro_failure_max_bb_pct_b_20
        self.medium_confirm_failure_max_days = medium_confirm_failure_max_days
        self.medium_confirm_failure_max_close_pos = medium_confirm_failure_max_close_pos
        self.trailing_stop_min_gain_to_arm = trailing_stop_min_gain_to_arm
        self.trailing_stop_min_days_to_arm = trailing_stop_min_days_to_arm
        self.breakout_trailing_atr_multiple = breakout_trailing_atr_multiple
        self.continuation_shelf_trailing_atr_multiple = continuation_shelf_trailing_atr_multiple
        self.continuation_pullback_trailing_atr_multiple = continuation_pullback_trailing_atr_multiple
        self.expansion_leader_trailing_atr_multiple = expansion_leader_trailing_atr_multiple
        self.power_breakout_trailing_atr_multiple = power_breakout_trailing_atr_multiple
        self.portfolio_risk_per_trade = portfolio_risk_per_trade
        self.max_allocation_per_stock = max_allocation_per_stock
        self.max_position_weight = max_position_weight
        self.equal_weight_allocation_cap = equal_weight_allocation_cap
        self.tiered_weight_power_breakout = tiered_weight_power_breakout
        self.tiered_weight_expansion_leader = tiered_weight_expansion_leader
        self.tiered_weight_late_stage_leader = tiered_weight_late_stage_leader
        self.tiered_weight_breakout = tiered_weight_breakout
        self.tiered_weight_continuation_shelf = tiered_weight_continuation_shelf
        self.tiered_weight_continuation_pullback = tiered_weight_continuation_pullback
        self.tiered_weight_leader_reentry = tiered_weight_leader_reentry
        self.tiered_weight_emerging_leader_ignition = tiered_weight_emerging_leader_ignition
        self.tiered_weight_emerging_leader_breakout = tiered_weight_emerging_leader_breakout
        self.tiered_weight_emerging_leader_shelf = tiered_weight_emerging_leader_shelf
        self.tiered_weight_default = tiered_weight_default
        self.min_stop_atr_multiple = min_stop_atr_multiple
        self.min_stop_pct = min_stop_pct
        self.feature_config = self._deep_merge_dicts(self.DEFAULT_FEATURE_CONFIG, feature_config or {})
        self.score_config = self._deep_merge_dicts(self.DEFAULT_SCORE_CONFIG, score_config or {})
        self.entry_logic_config = self._deep_merge_dicts(self.DEFAULT_ENTRY_LOGIC_CONFIG, entry_logic_config or {})
        self.exit_logic_config = self._deep_merge_dicts(self.DEFAULT_EXIT_LOGIC_CONFIG, exit_logic_config or {})
        self.ranking_config = self._deep_merge_dicts(self.DEFAULT_RANKING_CONFIG, ranking_config or {})

        ma_periods = self.feature_config["moving_average_periods"]
        volume_windows = self.feature_config["volume_average_windows"]
        rsi_periods = self.feature_config["rsi_periods"]
        smoothed_rsi = self.feature_config["smoothed_rsi"]
        macd_cfg = self.feature_config["macd"]
        stochastic_cfg = self.feature_config["stochastic"]
        cci_cfg = self.feature_config["cci"]
        realized_vol_cfg = self.feature_config["realized_vol"]
        rs_windows = self.feature_config["relative_strength_windows"]
        zone_windows = self.feature_config["zone_windows"]
        structure_windows = self.feature_config["structure_windows"]
        trend_quality_windows = self.feature_config["trend_quality_windows"]

        self.ma_short_period = int(ma_periods["short"])
        self.ma_medium_period = int(ma_periods["medium"])
        self.ma_long_period = int(ma_periods["long"])
        self.volume_short_window = int(volume_windows["short"])
        self.volume_long_window = int(volume_windows["long"])
        self.breakout_window = int(self.feature_config["breakout_window"])
        self.donchian_window = int(self.feature_config["donchian_window"])
        self.bollinger_window = int(self.feature_config["bollinger_window"])
        self.bollinger_std_mult = float(self.feature_config["bollinger_std_mult"])
        self.rsi_fast_period = int(rsi_periods["fast"])
        self.rsi_slow_period = int(rsi_periods["slow"])
        self.smoothed_rsi_ema_span = int(smoothed_rsi["ema_span"])
        self.smoothed_rsi_period = int(smoothed_rsi["rsi_period"])
        self.macd_fast_period = int(macd_cfg["fast"])
        self.macd_slow_period = int(macd_cfg["slow"])
        self.macd_signal_period = int(macd_cfg["signal"])
        self.stochastic_window = int(stochastic_cfg["window"])
        self.stochastic_signal_window = int(stochastic_cfg["signal"])
        self.cci_window = int(cci_cfg["window"])
        self.cci_constant = float(cci_cfg["constant"])
        self.atr_window = int(self.feature_config["atr_window"])
        self.directional_window = int(self.feature_config["directional_window"])
        self.realized_vol_window = int(realized_vol_cfg["window"])
        self.realized_vol_annualization = float(realized_vol_cfg["annualization"])
        self.cmf_window = int(self.feature_config["cmf_window"])
        self.mfi_window = int(self.feature_config["mfi_window"])
        self.rs_short_window = int(rs_windows["short"])
        self.rs_long_window = int(rs_windows["long"])
        self.zone_short_window = int(zone_windows["short"])
        self.zone_long_window = int(zone_windows["long"])
        self.structure_prior_short_window = int(structure_windows["prior_short"])
        self.structure_prior_medium_window = int(structure_windows["prior_medium"])
        self.structure_prior_long_window = int(structure_windows["prior_long"])
        self.tight_range_window = int(structure_windows["tight_range"])
        self.close_tightness_window = int(structure_windows["close_tightness"])
        self.change_lookback_window = int(structure_windows["change_lookback"])
        self.exit_break_low_window = int(structure_windows["exit_break_low"])
        self.confirmation_days = int(structure_windows["confirmation_days"])
        self.broader_overhead_supply_buffer = float(self.feature_config["broader_overhead_supply_buffer"])
        self.trend_slope_short_window = int(trend_quality_windows["slope_short"])
        self.trend_slope_long_window = int(trend_quality_windows["slope_long"])
        self.trend_ema_slope_window = int(trend_quality_windows["ema_slope"])
        self.trend_persistence_window = int(trend_quality_windows["persistence"])
        self.trend_drawdown_window = int(trend_quality_windows["drawdown"])

    def build_feature_dataframe(
        self,
        df: pd.DataFrame,
        *,
        spy_df: pd.DataFrame | None = None,
        qqq_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build the full feature set from raw OHLCV input.

        Input can be a multi-ticker frame. If `spy_df` / `qqq_df` are not passed,
        SPY/QQQ rows will be taken from the same input when available.
        """
        working = self._standardize_dataframe(df)
        self._require_columns(working, ("Date", "ticker", *self.RAW_PRICE_COLUMNS))
        working = self._coerce_numeric_columns(working, self.RAW_PRICE_COLUMNS)

        spy_series = self._resolve_benchmark_close(working, "SPY", spy_df)
        qqq_series = self._resolve_benchmark_close(working, "QQQ", qqq_df)

        feature_frames: list[pd.DataFrame] = []
        for _, ticker_df in working.groupby("ticker", sort=False):
            feature_frames.append(
                self._build_single_ticker_features(
                    ticker_df.copy(),
                    spy_close=spy_series,
                    qqq_close=qqq_series,
                )
            )

        if not feature_frames:
            return working.copy()
        return pd.concat(feature_frames, ignore_index=True)

    def score_row(self, row: pd.Series) -> dict[str, Any]:
        """Score a single ticker-date row using the exact bucket rules."""
        values = self._coerce_row(row)
        trend_cfg = self.score_config["trend"]
        breakout_cfg = self.score_config["breakout"]
        momentum_cfg = self.score_config["momentum"]
        flow_cfg = self.score_config["flow"]
        rs_cfg = self.score_config["relative_strength"]
        volatility_cfg = self.score_config["volatility"]
        penalty_cfg = self.score_config["penalties"]
        label_cfg = self.score_config["label_cutoffs"]
        pattern_stage_cfg = self.score_config["pattern_stage"]

        trend_points = self._clamp(
            (
                trend_cfg["close_vs_sma_short_weight"]
                if values["close_vs_sma_10"] > trend_cfg["close_vs_sma_short_threshold"]
                else 0
            )
            + (
                trend_cfg["close_vs_sma_medium_weight"]
                if values["close_vs_sma_20"] > trend_cfg["close_vs_sma_medium_threshold"]
                else 0
            )
            + (
                trend_cfg["close_vs_sma_long_weight"]
                if values["close_vs_sma_50"] > trend_cfg["close_vs_sma_long_threshold"]
                else 0
            )
            + (
                trend_cfg["close_vs_ema_short_weight"]
                if values["close_vs_ema_10"] > trend_cfg["close_vs_ema_short_threshold"]
                else 0
            )
            + (
                trend_cfg["close_vs_ema_medium_weight"]
                if values["close_vs_ema_20"] > trend_cfg["close_vs_ema_medium_threshold"]
                else 0
            )
            + (
                trend_cfg["close_vs_ema_long_weight"]
                if values["close_vs_ema_50"] > trend_cfg["close_vs_ema_long_threshold"]
                else 0
            )
            + (trend_cfg["trend_stack_bullish_weight"] if values["trend_stack_bullish"] == 1 else 0)
            - (trend_cfg["trend_stack_bearish_penalty"] if values["trend_stack_bearish"] == 1 else 0),
            0,
            trend_cfg["max_points"],
        )

        breakout_points = self._clamp(
            (
                breakout_cfg["pct_from_high_pullback_weight"]
                if breakout_cfg["pct_from_high_pullback_min"]
                <= values["pct_from_20d_high"]
                <= breakout_cfg["pct_from_high_pullback_max"]
                else 0
            )
            + (
                breakout_cfg["pct_from_high_breakout_weight"]
                if values["pct_from_20d_high"] > breakout_cfg["pct_from_high_breakout_min"]
                else 0
            )
            + (
                breakout_cfg["pct_from_low_weight"]
                if values["pct_from_20d_low"] > breakout_cfg["pct_from_low_min"]
                else 0
            )
            + (
                breakout_cfg["donchian_pos_weight"]
                if values["donchian_pos_20"] >= breakout_cfg["donchian_pos_min"]
                else 0
            )
            + (
                breakout_cfg["donchian_pos_high_weight"]
                if values["donchian_pos_20"] >= breakout_cfg["donchian_pos_high_min"]
                else 0
            )
            + (
                breakout_cfg["bb_pct_b_weight"]
                if values["bb_pct_b_20"] >= breakout_cfg["bb_pct_b_min"]
                else 0
            )
            + (
                breakout_cfg["bb_pct_b_high_weight"]
                if values["bb_pct_b_20"] >= breakout_cfg["bb_pct_b_high_min"]
                else 0
            ),
            0,
            breakout_cfg["max_points"],
        )

        momentum_points = self._clamp(
            (
                momentum_cfg["rsi_fast_mid_weight"]
                if values["rsi_14"] >= momentum_cfg["rsi_fast_mid_min"]
                else 0
            )
            + (
                momentum_cfg["rsi_fast_high_weight"]
                if values["rsi_14"] >= momentum_cfg["rsi_fast_high_min"]
                else 0
            )
            + (
                momentum_cfg["rsi_fast_extreme_weight"]
                if values["rsi_14"] >= momentum_cfg["rsi_fast_extreme_min"]
                else 0
            )
            + (
                momentum_cfg["rsi_slow_weight"]
                if values["rsi_21"] >= momentum_cfg["rsi_slow_min"]
                else 0
            )
            + (
                momentum_cfg["smoothed_rsi_weight"]
                if values["smoothed_rsi_ema21_rsi10"] >= momentum_cfg["smoothed_rsi_min"]
                else 0
            )
            + (momentum_cfg["macd_hist_positive_weight"] if values["macd_hist"] > 0 else 0)
            + (
                momentum_cfg["macd_line_above_signal_weight"]
                if values["macd_line"] > values["macd_signal"]
                else 0
            )
            + (
                momentum_cfg["stoch_k_mid_weight"]
                if values["stoch_k_14"] >= momentum_cfg["stoch_k_mid_min"]
                else 0
            )
            + (
                momentum_cfg["stoch_k_high_weight"]
                if values["stoch_k_14"] >= momentum_cfg["stoch_k_high_min"]
                else 0
            )
            + (
                momentum_cfg["stoch_d_weight"]
                if values["stoch_d_3"] >= momentum_cfg["stoch_d_min"]
                else 0
            )
            + (
                momentum_cfg["williams_r_weight"]
                if values["williams_r_14"] >= momentum_cfg["williams_r_min"]
                else 0
            )
            + (momentum_cfg["cci_weight"] if values["cci_20"] >= momentum_cfg["cci_min"] else 0)
            + (
                momentum_cfg["plus_di_above_minus_di_weight"]
                if values["plus_di_14"] > values["minus_di_14"]
                else 0
            )
            + (momentum_cfg["adx_weight"] if values["adx_14"] >= momentum_cfg["adx_min"] else 0),
            0,
            momentum_cfg["max_points"],
        )

        flow_points = self._clamp(
            (
                flow_cfg["volume_ratio_short_weight"]
                if values["volume_ratio_20"] >= flow_cfg["volume_ratio_short_min"]
                else 0
            )
            + (
                flow_cfg["volume_ratio_high_weight"]
                if values["volume_ratio_20"] >= flow_cfg["volume_ratio_high_min"]
                else 0
            )
            + (
                flow_cfg["volume_ratio_long_weight"]
                if values["volume_ratio_50"] >= flow_cfg["volume_ratio_long_min"]
                else 0
            )
            + (flow_cfg["volume_zscore_non_negative_weight"] if values["volume_zscore_20"] >= 0 else 0)
            + (
                flow_cfg["volume_zscore_high_weight"]
                if values["volume_zscore_20"] >= flow_cfg["volume_zscore_high_min"]
                else 0
            )
            + (flow_cfg["cmf_non_negative_weight"] if values["cmf_20"] >= 0 else 0)
            + (flow_cfg["cmf_high_weight"] if values["cmf_20"] >= flow_cfg["cmf_high_min"] else 0)
            + (flow_cfg["mfi_weight"] if values["mfi_14"] >= flow_cfg["mfi_min"] else 0)
            + (flow_cfg["pct_chg_positive_weight"] if values["pct_chg"] > 0 else 0)
            + (flow_cfg["close_pos_weight"] if values["close_pos"] >= flow_cfg["close_pos_min"] else 0),
            0,
            flow_cfg["max_points"],
        )

        rs_points = self._clamp(
            (rs_cfg["rs_spy_short_positive_weight"] if values["rs_spy_20"] > 0 else 0)
            + (rs_cfg["rs_spy_long_positive_weight"] if values["rs_spy_50"] > 0 else 0)
            + (rs_cfg["rs_qqq_short_positive_weight"] if values["rs_qqq_20"] > 0 else 0)
            + (rs_cfg["rs_qqq_long_positive_weight"] if values["rs_qqq_50"] > 0 else 0)
            + (
                rs_cfg["rs_spy_short_strong_weight"]
                if values["rs_spy_20"] > rs_cfg["rs_spy_short_strong_min"]
                else 0
            )
            + (
                rs_cfg["rs_qqq_short_strong_weight"]
                if values["rs_qqq_20"] > rs_cfg["rs_qqq_short_strong_min"]
                else 0
            ),
            0,
            rs_cfg["max_points"],
        )

        body_over_tr = (values["body"] / values["tr"]) if values["tr"] > 0 else 0.0
        volatility_points = self._clamp(
            (
                volatility_cfg["atr_pct_low_weight"]
                if volatility_cfg["atr_pct_low_min"] <= values["atr_pct_14"] <= volatility_cfg["atr_pct_low_max"]
                else 0
            )
            + (
                volatility_cfg["atr_pct_high_weight"]
                if volatility_cfg["atr_pct_low_max"] < values["atr_pct_14"] <= volatility_cfg["atr_pct_high_max"]
                else 0
            )
            + (
                volatility_cfg["body_over_tr_weight"]
                if values["tr"] > 0 and body_over_tr >= volatility_cfg["body_over_tr_min"]
                else 0
            )
            + (
                volatility_cfg["close_pos_weight"]
                if values["close_pos"] >= volatility_cfg["close_pos_min"]
                else 0
            )
            + (volatility_cfg["realized_vol_positive_weight"] if values["realized_vol_20"] > 0 else 0),
            0,
            volatility_cfg["max_points"],
        )

        penalty = (
            (penalty_cfg["trend_stack_bearish"] if values["trend_stack_bearish"] == 1 else 0)
            + (
                penalty_cfg["minus_di_above_plus_di_and_rsi_below_50"]
                if values["minus_di_14"] > values["plus_di_14"] and values["rsi_14"] < 50
                else 0
            )
            + (
                penalty_cfg["deep_off_high_with_negative_rs"]
                if values["pct_from_20d_high"] < penalty_cfg["deep_off_high_threshold"]
                and values["rs_spy_20"] < 0
                and values["rs_qqq_20"] < 0
                else 0
            )
            + (
                penalty_cfg["weak_volume_and_negative_cmf"]
                if values["volume_ratio_20"] < penalty_cfg["weak_volume_ratio_threshold"] and values["cmf_20"] < 0
                else 0
            )
            + (
                penalty_cfg["negative_macd_and_rsi_below_50"]
                if values["macd_line"] < values["macd_signal"]
                and values["macd_hist"] < 0
                and values["rsi_14"] < 50
                else 0
            )
        )

        score = self._clamp(
            trend_points
            + breakout_points
            + momentum_points
            + flow_points
            + rs_points
            + volatility_points
            - penalty,
            0,
            100,
        )

        if score >= label_cfg["A"]:
            label = "A"
        elif score >= label_cfg["B"]:
            label = "B"
        elif score >= label_cfg["C"]:
            label = "C"
        else:
            label = "D"

        if score >= pattern_stage_cfg["signal_score"] and values["pct_chg"] <= pattern_stage_cfg["breakout_pct_chg"]:
            pattern_stage = "launchpad_or_early_trigger"
        elif score >= pattern_stage_cfg["signal_score"] and values["pct_chg"] > pattern_stage_cfg["breakout_pct_chg"]:
            pattern_stage = "breakout_or_power_trend"
        else:
            pattern_stage = "non_signal"

        return {
            "trend_points": trend_points,
            "breakout_points": breakout_points,
            "momentum_points": momentum_points,
            "flow_points": flow_points,
            "rs_points": rs_points,
            "volatility_points": volatility_points,
            "penalty": penalty,
            "score": score,
            "label": label,
            "pattern_stage": pattern_stage,
        }

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a scored DataFrame with debug bucket columns."""
        working = self._standardize_dataframe(df)
        if not self._has_feature_columns(working):
            if self._has_raw_price_columns(working):
                working = self.build_feature_dataframe(working)
            else:
                working = self._fill_feature_defaults(working)
        else:
            working = self._fill_feature_defaults(working)
        scored = working.apply(self.score_row, axis=1, result_type="expand")
        combined = pd.concat([working, scored], axis=1)
        return self._augment_entry_support_columns(combined)

    def generate_entries(self, df: pd.DataFrame) -> pd.Series:
        """Return stateful setup-trigger entry signals."""
        scored = self._augment_entry_support_columns(self._ensure_scored(df))
        return scored["entry_signal"].rename("entry_signal")

    def generate_exits(self, df: pd.DataFrame) -> pd.Series:
        """Return confirmed trend-failure exits, excluding entry-aware trailing stops."""
        scored = self._augment_exit_support_columns(self._ensure_scored(df))
        exit_signal = (
            scored["soft_score_fail_2d"]
            | scored["close_below_ema20_2d"]
            | scored["close_below_sma50"]
            | scored["relative_weak_2d"]
        )
        return exit_signal.rename("exit_signal")

    def rank_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank baseline entry candidates for each date."""
        scored = self._augment_entry_support_columns(self._ensure_scored(df).copy())
        candidates = scored[scored["entry_signal"]].copy()
        candidates = candidates[~candidates["ticker"].isin(self.BENCHMARK_TICKERS)].copy()
        if candidates.empty:
            return candidates

        setup_priority = self.ranking_config["setup_priority"]
        candidates["setup_priority"] = candidates["setup_type"].map(setup_priority).fillna(3)
        allocation_mode = self._resolved_allocation_mode()
        if allocation_mode == "hybrid_risk_capped":
            sort_columns = [
                "Date",
                "setup_priority",
                "leadership_score",
                "entry_tight_stop_penalty",
                "score",
                "entry_breathing_room_ratio",
                "rs_qqq_20",
                "rs_spy_20",
                "volume_ratio_20",
                "ticker",
            ]
            ascending = [True, True, False, True, False, False, False, False, False, True]
        else:
            sort_columns = [
                "Date",
                "setup_priority",
                "leadership_score",
                "score",
                "rs_qqq_20",
                "rs_spy_20",
                "volume_ratio_20",
                "ticker",
            ]
            ascending = [True, True, False, False, False, False, False, True]
        candidates = candidates.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)
        candidates["candidate_rank"] = candidates.groupby("Date").cumcount() + 1
        return candidates

    def backtest(
        self,
        df: pd.DataFrame,
        max_positions: int = 0,
        initial_capital: float = 100_000.0,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        trade_start_date: str | pd.Timestamp | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Run a no-lookahead daily backtest.

        Signals are evaluated at each row's close. Exits and new entries are both
        executed at that close, and new positions participate from the next bar.
        """
        scored = self._ensure_scored(df).copy()
        if start_date is not None:
            scored = scored[scored["Date"] >= pd.Timestamp(start_date)].copy()
        if end_date is not None:
            scored = scored[scored["Date"] <= pd.Timestamp(end_date)].copy()
        scored = self._augment_entry_support_columns(scored)
        scored = self._augment_exit_support_columns(scored)
        scored["exit_signal"] = self.generate_exits(scored)
        scored = scored.sort_values(["Date", "ticker"]).reset_index(drop=True)
        trade_start_ts = pd.Timestamp(trade_start_date) if trade_start_date is not None else None
        ranked_all = self.rank_candidates(scored)
        uncapped_positions = max_positions <= 0

        positions: dict[str, _BacktestPosition] = {}
        last_exit_date_index: dict[str, int] = {}
        cash = float(initial_capital)
        trades: list[dict[str, Any]] = []
        holdings_rows: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []

        grouped = list(scored.groupby("Date", sort=True))
        for date_index, (current_date, day_df) in enumerate(grouped):
            day_df = day_df.sort_values(["score", "rs_spy_20", "volume_ratio_20", "ticker"], ascending=[False, False, False, True])
            day_rows = {row["ticker"]: row for _, row in day_df.iterrows()}

            for ticker, position in list(positions.items()):
                row = day_rows.get(ticker)
                if row is None:
                    continue

                updated_position = _BacktestPosition(
                    ticker=position.ticker,
                    entry_date=position.entry_date,
                    entry_price=position.entry_price,
                    shares=position.shares,
                    entry_score=position.entry_score,
                    setup_type=position.setup_type,
                    best_score=max(position.best_score, float(row["score"])),
                    highest_close=max(position.highest_close, float(row["close"])),
                    has_new_high=position.has_new_high or float(row["pct_from_20d_high"]) >= 0,
                    score_improved=position.score_improved or float(row["score"]) > position.entry_score,
                    days_held=position.days_held + 1,
                    add_on_count=position.add_on_count,
                    zone_support=position.zone_support,
                )
                positions[ticker] = updated_position

                exit_reason = self._exit_reason(row, updated_position)
                if exit_reason is None:
                    continue

                exit_price = float(row["close"])
                cash += updated_position.shares * exit_price
                trades.append(
                    {
                        "ticker": ticker,
                        "entry_date": updated_position.entry_date,
                        "exit_date": current_date,
                        "entry_price": updated_position.entry_price,
                        "exit_price": exit_price,
                        "shares": updated_position.shares,
                        "setup_type": updated_position.setup_type,
                        "entry_score": updated_position.entry_score,
                        "exit_score": float(row["score"]),
                        "holding_days": updated_position.days_held,
                        "return_pct": (exit_price / updated_position.entry_price) - 1.0
                        if updated_position.entry_price > 0
                        else 0.0,
                        "pnl": updated_position.shares * (exit_price - updated_position.entry_price),
                        "exit_reason": exit_reason,
                    }
                )
                last_exit_date_index[ticker] = date_index
                del positions[ticker]

            for ticker, position in list(positions.items()):
                row = day_rows.get(ticker)
                if row is None:
                    continue
                if position.add_on_count >= self.max_add_ons_per_ticker:
                    continue
                current_setup_type = str(row.get("setup_type", "none"))
                aggressive_starter_setup = self._uses_starter_sizing(position.setup_type)
                has_confirmation_signal = (
                    bool(row.get("zone_reentry_signal", False))
                    if aggressive_starter_setup
                    else bool(row.get("entry_signal", False))
                )
                if not has_confirmation_signal:
                    continue
                if not (
                    self._is_continuation_setup_type(current_setup_type)
                    or current_setup_type == "power_breakout"
                    or (aggressive_starter_setup and current_setup_type in {"power_breakout", "expansion_leader", "none"})
                ):
                    continue
                open_gain = (
                    (float(row["close"]) / position.entry_price) - 1.0
                    if position.entry_price > 0
                    else 0.0
                )
                if open_gain < self.add_on_profit_threshold:
                    continue
                if not bool(row.get("zone_reentry_signal", False)):
                    continue

                price = float(row["close"])
                if price <= 0:
                    continue
                remaining_stock_capacity = self._remaining_stock_capacity(position.shares * price)
                add_on_fraction = (
                    self.aggressive_add_on_size_fraction
                    if aggressive_starter_setup
                    else self.add_on_size_fraction
                )
                allocation = min(cash, position.shares * price * add_on_fraction, remaining_stock_capacity)
                if allocation <= 0:
                    continue

                add_on_shares = allocation / price
                if add_on_shares <= 0:
                    continue

                total_shares = position.shares + add_on_shares
                blended_entry = (
                    ((position.entry_price * position.shares) + (price * add_on_shares)) / total_shares
                    if total_shares > 0
                    else position.entry_price
                )
                cash -= allocation
                positions[ticker] = _BacktestPosition(
                    ticker=position.ticker,
                    entry_date=position.entry_date,
                    entry_price=blended_entry,
                    shares=total_shares,
                    entry_score=max(position.entry_score, float(row["score"])),
                    setup_type=position.setup_type,
                    best_score=max(position.best_score, float(row["score"])),
                    highest_close=max(position.highest_close, price),
                    has_new_high=position.has_new_high or float(row["pct_from_20d_high"]) >= 0,
                    score_improved=True,
                    days_held=position.days_held,
                    add_on_count=position.add_on_count + 1,
                    zone_support=max(position.zone_support, self._entry_zone_support_level(row, position.setup_type)),
                )

            ranked_candidates = ranked_all[ranked_all["Date"] == current_date].copy()
            if trade_start_ts is not None and current_date < trade_start_ts:
                ranked_candidates = ranked_candidates.iloc[0:0]

            if not ranked_candidates.empty:
                eligible_candidates: list[pd.Series] = []
                for _, row in ranked_candidates.iterrows():
                    ticker = row["ticker"]
                    if ticker in positions:
                        continue

                    last_exit_idx = last_exit_date_index.get(ticker)
                    if (
                        last_exit_idx is not None
                        and (date_index - last_exit_idx) <= self.reentry_cooldown_days
                        and float(row["score"]) < 75
                        and not bool(row.get("zone_reentry_signal", False))
                    ):
                        continue

                    if float(row["close"]) <= 0:
                        continue
                    eligible_candidates.append(row)

                available_slots = len(eligible_candidates)
                if not uncapped_positions:
                    available_slots = min(available_slots, max(max_positions - len(positions), 0))

            else:
                eligible_candidates = []
                available_slots = 0

            if available_slots > 0:
                total_equity = cash + sum(
                    positions[ticker].shares * float(day_rows[ticker]["close"])
                    for ticker in positions
                    if ticker in day_rows
                )
                for _, row in enumerate(eligible_candidates[:available_slots]):
                    price = float(row["close"])
                    stock_allocation_cap = self._remaining_stock_capacity(0.0)
                    allocation_mode = self._resolved_allocation_mode()
                    if allocation_mode == "hybrid_risk_capped":
                        risk_per_share = float(row.get("entry_risk_per_share", 0.0))
                        if price <= 0 or risk_per_share <= 0:
                            continue

                        desired_shares = self.portfolio_risk_per_trade / risk_per_share
                        desired_allocation = desired_shares * price
                        position_value_cap = total_equity * min(
                            self.max_position_weight,
                            self._setup_target_weight(str(row.get("setup_type", "none"))),
                        )
                        allocation = min(cash, desired_allocation, position_value_cap, stock_allocation_cap)
                        if allocation <= 0 or allocation < price:
                            break
                    elif allocation_mode == "setup_tiered_cap":
                        if price <= 0:
                            continue
                        position_value_cap = total_equity * min(
                            self.max_position_weight,
                            self._setup_target_weight(str(row.get("setup_type", "none"))),
                        )
                        allocation = min(cash, position_value_cap, stock_allocation_cap)
                        if allocation <= 0 or allocation < price:
                            break
                    elif allocation_mode == "equal_weight_cap":
                        if price <= 0:
                            continue
                        remaining_entries = available_slots
                        equal_weight_allocation = cash / remaining_entries if remaining_entries > 0 else 0.0
                        allocation_cap = total_equity * min(self.max_position_weight, self.equal_weight_allocation_cap)
                        allocation = min(cash, equal_weight_allocation, allocation_cap, stock_allocation_cap)
                        if allocation <= 0 or allocation < price:
                            break
                    else:
                        if price <= 0:
                            continue
                        if uncapped_positions:
                            remaining_entries = available_slots
                            allocation = cash / remaining_entries if remaining_entries > 0 else 0.0
                        else:
                            target_position_value = total_equity / max_positions if max_positions > 0 else 0.0
                            allocation = min(cash, target_position_value)
                        allocation = min(allocation, stock_allocation_cap)
                        if allocation <= 0:
                            break

                    starter_multiplier = self.starter_size_multiplier(str(row.get("setup_type", "none")))
                    if starter_multiplier < 1.0:
                        allocation *= starter_multiplier
                        if allocation <= 0:
                            continue

                    shares = allocation / price
                    if shares <= 0:
                        continue

                    cash -= allocation
                    ticker = row["ticker"]
                    positions[ticker] = _BacktestPosition(
                        ticker=ticker,
                        entry_date=current_date,
                        entry_price=price,
                        shares=shares,
                        entry_score=float(row["score"]),
                        setup_type=str(row.get("setup_type", "none")),
                        best_score=float(row["score"]),
                        highest_close=price,
                        has_new_high=float(row["pct_from_20d_high"]) >= 0,
                        score_improved=False,
                        days_held=0,
                        add_on_count=0,
                        zone_support=self._entry_zone_support_level(row, str(row.get("setup_type", "none"))),
                    )
                    available_slots -= 1

            invested_value = 0.0
            holding_snapshots: list[dict[str, Any]] = []
            for ticker, position in positions.items():
                row = day_rows.get(ticker)
                if row is None:
                    continue
                market_value = position.shares * float(row["close"])
                invested_value += market_value
                holding_snapshots.append(
                    {
                        "Date": current_date,
                        "ticker": ticker,
                        "shares": position.shares,
                        "close": float(row["close"]),
                        "market_value": market_value,
                        "entry_date": position.entry_date,
                        "entry_price": position.entry_price,
                        "entry_score": position.entry_score,
                        "current_score": float(row["score"]),
                        "days_held": position.days_held,
                    }
                )

            total_equity = cash + invested_value
            for snapshot in holding_snapshots:
                snapshot["weight"] = (
                    snapshot["market_value"] / total_equity if total_equity > 0 else 0.0
                )
                holdings_rows.append(snapshot)
            equity_rows.append(
                {
                    "Date": current_date,
                    "cash": cash,
                    "invested_value": invested_value,
                    "total_equity": total_equity,
                    "num_positions": len(positions),
                }
            )

        return {
            "trades": pd.DataFrame(trades),
            "daily_holdings": pd.DataFrame(holdings_rows),
            "equity_curve": pd.DataFrame(equity_rows),
            "scored_data": scored,
        }

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        working = self._fill_feature_defaults(self._standardize_dataframe(df))
        high_col = "high" if "high" in working.columns else "close"
        low_col = "low" if "low" in working.columns else "close"
        working = add_zone_columns(
            working,
            high_col=high_col,
            low_col=low_col,
            close_col="close",
            group_col="ticker",
            windows=(self.zone_short_window, self.zone_long_window),
            seller_zone_fraction=self.zone_seller_fraction,
            demand_zone_fraction=self.zone_demand_fraction,
        )
        working["prior_20bar_high"] = working[f"prior_{self.zone_short_window}bar_high"]
        working["zone_width_20"] = working[f"zone_width_{self.zone_short_window}"]
        working["prior_60bar_high"] = working[f"prior_{self.zone_long_window}bar_high"]
        working["room_to_60bar_high"] = working[f"room_to_{self.zone_long_window}bar_high"]
        working["in_60bar_seller_zone"] = working[f"in_{self.zone_long_window}bar_seller_zone"]
        working["prior_3bar_high"] = (
            working.groupby("ticker", sort=False)[high_col]
            .transform(
                lambda series: series.shift(1).rolling(self.structure_prior_short_window, min_periods=1).max()
            )
            .fillna(0.0)
        )
        working["prior_5bar_high"] = (
            working.groupby("ticker", sort=False)[high_col]
            .transform(
                lambda series: series.shift(1).rolling(self.structure_prior_medium_window, min_periods=1).max()
            )
            .fillna(0.0)
        )
        working["prior_3bar_low"] = (
            working.groupby("ticker", sort=False)[low_col]
            .transform(
                lambda series: series.shift(1).rolling(self.structure_prior_short_window, min_periods=1).min()
            )
            .fillna(0.0)
        )
        working["prior_5bar_low"] = (
            working.groupby("ticker", sort=False)[low_col]
            .transform(
                lambda series: series.shift(1).rolling(self.structure_prior_medium_window, min_periods=1).min()
            )
            .fillna(0.0)
        )
        working["prior_20bar_low"] = (
            working.groupby("ticker", sort=False)[low_col]
            .transform(
                lambda series: series.shift(1).rolling(self.structure_prior_long_window, min_periods=1).min()
            )
            .fillna(0.0)
        )
        rolling_high_5 = working.groupby("ticker", sort=False)[high_col].transform(
            lambda series: series.rolling(self.tight_range_window, min_periods=1).max()
        )
        rolling_low_5 = working.groupby("ticker", sort=False)[low_col].transform(
            lambda series: series.rolling(self.tight_range_window, min_periods=1).min()
        )
        working["tight_range_5"] = self._safe_divide(
            rolling_high_5 - rolling_low_5,
            working["close"].abs(),
            0.0,
        )
        close_std_3 = working.groupby("ticker", sort=False)["close"].transform(
            lambda series: series.rolling(self.close_tightness_window, min_periods=1).std(ddof=0)
        )
        close_mean_3 = working.groupby("ticker", sort=False)["close"].transform(
            lambda series: series.rolling(self.close_tightness_window, min_periods=1).mean()
        )
        working["close_tightness_3"] = self._safe_divide(close_std_3, close_mean_3.abs(), 0.0)
        working["close_to_prior_20bar_high"] = self._safe_divide(
            working["close"] - working["prior_20bar_high"],
            working["close"].abs(),
            0.0,
        )
        working["close_to_prior_20bar_low"] = self._safe_divide(
            working["close"] - working["prior_20bar_low"],
            working["close"].abs(),
            0.0,
        )
        working["rs_spy_20_change_3"] = (
            working.groupby("ticker", sort=False)["rs_spy_20"].transform(
                lambda series: series.diff(self.change_lookback_window)
            )
        ).fillna(0.0)
        working["rs_qqq_20_change_3"] = (
            working.groupby("ticker", sort=False)["rs_qqq_20"].transform(
                lambda series: series.diff(self.change_lookback_window)
            )
        ).fillna(0.0)
        working["atr_pct_14_change_3"] = (
            working.groupby("ticker", sort=False)["atr_pct_14"].transform(
                lambda series: series.diff(self.change_lookback_window)
            )
        ).fillna(0.0)
        working["support_cluster_gap"] = np.minimum(
            working["close_vs_ema_20"].abs(),
            working["close_vs_sma_50"].abs(),
        )
        working["base_width_20"] = self._safe_divide(
            working["prior_20bar_high"] - working["prior_20bar_low"],
            working["close"].abs(),
            0.0,
        )
        working["base_tightness_20"] = self._safe_divide(
            working["tight_range_5"],
            working["base_width_20"].abs(),
            0.0,
        )
        working["rs_acceleration_3"] = working["rs_spy_20_change_3"] + working["rs_qqq_20_change_3"]
        working["confirmed_leader_score"] = self._confirmed_leader_score(working)
        working["confirmed_leader_regime_signal"] = self._confirmed_leader_regime_signal(working).astype(int)
        working["emerging_leader_score"] = self._emerging_leader_score(working)
        working["emerging_leader_regime_signal"] = self._emerging_leader_regime_signal(working).astype(int)
        working["leadership_score"] = np.where(
            working["confirmed_leader_regime_signal"].astype(bool),
            working["confirmed_leader_score"] + 2.0,
            working["emerging_leader_score"],
        )
        working["leadership_stage"] = np.where(
            working["confirmed_leader_regime_signal"].astype(bool),
            "confirmed",
            np.where(working["emerging_leader_regime_signal"].astype(bool), "emerging", "none"),
        )
        working["leader_trend_score"] = working["confirmed_leader_score"]
        working["leader_regime_signal"] = working["confirmed_leader_regime_signal"]
        return working

    def _ensure_scored(self, df: pd.DataFrame) -> pd.DataFrame:
        if {"score", "trend_points", "penalty", "pattern_stage", "label"}.issubset(df.columns):
            return self._prepare_dataframe(df)
        return self.score_dataframe(df)

    def _augment_exit_support_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        working = self._prepare_dataframe(df).copy()
        low_source = "low" if "low" in working.columns else "close"

        working["roll_low_10"] = (
            working.groupby("ticker", sort=False)[low_source]
            .transform(lambda series: series.rolling(self.exit_break_low_window, min_periods=1).min())
        )
        working["soft_score_fail"] = working["score"] < self.exit_logic_config["soft_score_fail_threshold"]
        working["close_below_ema20"] = working["close_vs_ema_20"] < 0
        working["close_below_sma50"] = working["close_vs_sma_50"] < 0
        working["relative_weak"] = (working["rs_spy_20"] < 0) & (working["rs_qqq_20"] < 0)
        working["micro_support_break"] = working[low_source] < working["prior_3bar_low"]
        prior_break = (
            working.groupby("ticker", sort=False)["micro_support_break"]
            .shift(1)
            .fillna(False)
            .astype(bool)
        )
        working["bb_micro_support_fail"] = (
            prior_break
            & (working["close"] <= working["prior_3bar_low"])
            & (working["close_pos"] <= self.bb_micro_failure_max_close_pos)
            & (working["bb_pct_b_20"] <= self.bb_micro_failure_max_bb_pct_b_20)
        )
        working["medium_confirm_failure"] = (
            prior_break
            & (working["close"] <= working["prior_3bar_low"])
            & (
                (working["close_below_ema20"])
                | (working["relative_weak"])
                | (working["close_pos"] <= self.medium_confirm_failure_max_close_pos)
            )
        )

        for source, target in (
            ("soft_score_fail", "soft_score_fail_2d"),
            ("close_below_ema20", "close_below_ema20_2d"),
            ("relative_weak", "relative_weak_2d"),
        ):
            working[target] = (
                working.groupby("ticker", sort=False)[source]
                .transform(
                    lambda series: (
                        series.astype(int)
                        .rolling(self.confirmation_days, min_periods=self.confirmation_days)
                        .sum()
                        >= self.confirmation_days
                    )
                )
            )

        return working

    def _augment_entry_support_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        working = self._prepare_dataframe(df).copy()
        combined_rs = working["rs_spy_20"] + working["rs_qqq_20"]
        ema20_price = self._level_from_relative_close(working["close"], working["close_vs_ema_20"])
        sma50_price = self._level_from_relative_close(working["close"], working["close_vs_sma_50"])
        structural_support = np.maximum.reduce(
            [
                np.nan_to_num(working["prior_20bar_low"].to_numpy(dtype=float), nan=0.0),
                np.nan_to_num(ema20_price.to_numpy(dtype=float), nan=0.0),
                np.nan_to_num(sma50_price.to_numpy(dtype=float), nan=0.0),
            ]
        )
        structural_support = np.minimum(structural_support, working["close"].to_numpy(dtype=float))
        structural_distance = np.maximum(working["close"].to_numpy(dtype=float) - structural_support, 0.0)
        min_breathing_distance = np.maximum(
            self.min_stop_atr_multiple * working["atr_14"].to_numpy(dtype=float),
            self.min_stop_pct * working["close"].to_numpy(dtype=float),
        )
        entry_risk_per_share = np.maximum(structural_distance, min_breathing_distance)
        working["entry_structural_support"] = structural_support
        working["entry_structural_distance"] = structural_distance
        working["entry_min_breathing_distance"] = min_breathing_distance
        working["entry_risk_per_share"] = entry_risk_per_share
        working["entry_stop_price"] = np.maximum(working["close"].to_numpy(dtype=float) - entry_risk_per_share, 0.0)
        working["entry_risk_pct"] = self._safe_divide(working["entry_risk_per_share"], working["close"].abs(), 0.0)
        working["entry_tight_stop_penalty"] = (
            working["entry_structural_distance"] < working["entry_min_breathing_distance"]
        )
        working["entry_breathing_room_ratio"] = self._safe_divide(
            working["entry_structural_distance"],
            working["entry_min_breathing_distance"],
            0.0,
        )
        overlays_enabled = self.enable_leader_reentry or self.enable_late_stage_leaders
        if overlays_enabled:
            super_leader_signal = self._super_leader_signal(working)
            recent_super_leader = (
                super_leader_signal.groupby(working["ticker"], sort=False)
                .transform(
                    lambda series: series.shift(1).rolling(self.super_leader_lookback_days, min_periods=1).max()
                )
                .fillna(0.0)
                .astype(bool)
            )
        else:
            super_leader_signal = pd.Series(False, index=working.index)
            recent_super_leader = pd.Series(False, index=working.index)
        breakout_setup_signal = self._breakout_setup_signal(working)
        continuation_shelf_setup_signal = self._continuation_shelf_setup_signal(working)
        continuation_pullback_setup_signal = self._continuation_pullback_setup_signal(working)
        continuation_setup_signal = continuation_shelf_setup_signal | continuation_pullback_setup_signal
        expansion_leader_setup_signal = self._expansion_leader_setup_signal(working)
        power_breakout_setup_signal = self._power_breakout_setup_signal(working)
        emerging_leader_ignition_setup_signal = self._emerging_leader_ignition_setup_signal(working)
        emerging_leader_breakout_setup_signal = self._emerging_leader_breakout_setup_signal(working)
        emerging_leader_shelf_setup_signal = self._emerging_leader_shelf_setup_signal(working)
        leader_reentry_setup_signal = (
            self._leader_reentry_setup_signal(
                working,
                recent_super_leader=recent_super_leader,
            )
            if self.enable_leader_reentry
            else pd.Series(False, index=working.index)
        )
        late_stage_leader_setup_signal = (
            self._late_stage_leader_setup_signal(
                working,
                recent_super_leader=recent_super_leader,
            )
            if self.enable_late_stage_leaders
            else pd.Series(False, index=working.index)
        )
        zone_reentry_signal = self._zone_reentry_signal(working)
        base_setup_signal = (
            breakout_setup_signal
            | continuation_setup_signal
            | expansion_leader_setup_signal
            | leader_reentry_setup_signal
            | late_stage_leader_setup_signal
            | power_breakout_setup_signal
            | emerging_leader_ignition_setup_signal
            | emerging_leader_breakout_setup_signal
            | emerging_leader_shelf_setup_signal
        )
        previous_setup_signal = base_setup_signal.groupby(working["ticker"], sort=False).shift(1)
        previous_setup_signal = previous_setup_signal.where(previous_setup_signal.notna(), False).astype(bool)
        fresh_setup_signal = base_setup_signal & (~previous_setup_signal)

        working["setup_signal"] = False
        working["setup_active"] = False
        working["setup_age"] = 0
        working["setup_high"] = np.nan
        working["trigger_pivot_high"] = np.nan
        working["trigger_level"] = np.nan
        working["entry_signal"] = False
        working["setup_cancelled"] = False
        working["setup_expired"] = False
        working["setup_state"] = "no_setup"
        working["setup_type"] = "none"
        working["zone_reentry_signal"] = zone_reentry_signal
        working["super_leader_signal"] = super_leader_signal
        working["recent_super_leader"] = recent_super_leader

        for _, index_group in working.groupby("ticker", sort=False).groups.items():
            active = False
            setup_age = 0
            setup_high = np.nan
            active_setup_type = "none"

            for idx in index_group:
                row = working.loc[idx]
                entry_triggered = False
                cancelled = False
                expired = False
                state = "no_setup"
                pivot_high = float(row.get("prior_3bar_high", 0.0))
                trigger_level = np.nan
                current_setup_type = active_setup_type
                has_expansion_leader = bool(expansion_leader_setup_signal.loc[idx])
                has_leader_reentry = bool(leader_reentry_setup_signal.loc[idx])
                has_late_stage_leader = bool(late_stage_leader_setup_signal.loc[idx])
                has_power_breakout = bool(power_breakout_setup_signal.loc[idx])
                has_emerging_leader_ignition = bool(emerging_leader_ignition_setup_signal.loc[idx])
                has_emerging_leader_breakout = bool(emerging_leader_breakout_setup_signal.loc[idx])

                if active:
                    setup_age += 1
                    pivot_high = float(
                        row.get("prior_5bar_high", 0.0)
                        if self._uses_medium_trigger_window(active_setup_type)
                        else row.get("prior_3bar_high", 0.0)
                    )
                    trigger_level = max(float(setup_high), pivot_high)
                    min_volume_ratio = (
                        self.continuation_trigger_volume_ratio
                        if self._is_continuation_setup_type(active_setup_type)
                        else (
                            self.emerging_leader_shelf_trigger_volume_ratio
                            if active_setup_type == "emerging_leader_shelf"
                            else self.trigger_volume_ratio
                        )
                    )
                    max_trigger_extension = (
                        self.continuation_max_trigger_extension
                        if self._is_continuation_setup_type(active_setup_type)
                        else (
                            self.emerging_leader_max_trigger_extension
                            if active_setup_type == "emerging_leader_shelf"
                            else self.max_trigger_extension
                        )
                    )
                    continuation_trigger_quality_ok = (
                        (
                            active_setup_type == "emerging_leader_shelf"
                            and float(row["rs_acceleration_3"]) >= self.emerging_leader_min_rs_acceleration_3
                        )
                        or (
                            not self._is_continuation_setup_type(active_setup_type)
                            and active_setup_type != "emerging_leader_shelf"
                        )
                        or (
                            float(row["rs_qqq_20_change_3"]) >= self.continuation_min_rs_qqq_change_3
                            and float(row["atr_pct_14_change_3"]) <= self.continuation_max_atr_pct_change_3
                        )
                    )
                    if not self._setup_regime_active(row, active_setup_type):
                        active = False
                        cancelled = True
                        state = "setup_cancelled"
                    elif (
                        active_setup_type != "emerging_leader_shelf"
                        and float(row["close_vs_ema_20"]) < 0
                    ):
                        active = False
                        cancelled = True
                        state = "setup_cancelled"
                    elif (
                        active_setup_type == "emerging_leader_shelf"
                        and float(row["close_vs_sma_50"]) < self.emerging_leader_shelf_min_close_vs_sma_50
                    ):
                        active = False
                        cancelled = True
                        state = "setup_cancelled"
                    elif has_expansion_leader:
                        working.at[idx, "setup_signal"] = True
                        working.at[idx, "entry_signal"] = True
                        entry_triggered = True
                        active = False
                        active_setup_type = "none"
                        setup_age = 0
                        setup_high = np.nan
                        current_setup_type = "expansion_leader"
                        state = "entered"
                    elif has_power_breakout:
                        working.at[idx, "setup_signal"] = True
                        working.at[idx, "entry_signal"] = True
                        entry_triggered = True
                        active = False
                        active_setup_type = "none"
                        setup_age = 0
                        setup_high = np.nan
                        current_setup_type = "power_breakout"
                        state = "entered"
                    elif has_emerging_leader_ignition:
                        working.at[idx, "setup_signal"] = True
                        working.at[idx, "entry_signal"] = True
                        entry_triggered = True
                        active = False
                        active_setup_type = "none"
                        setup_age = 0
                        setup_high = np.nan
                        current_setup_type = "emerging_leader_ignition"
                        state = "entered"
                    elif has_emerging_leader_breakout:
                        working.at[idx, "setup_signal"] = True
                        working.at[idx, "entry_signal"] = True
                        entry_triggered = True
                        active = False
                        active_setup_type = "none"
                        setup_age = 0
                        setup_high = np.nan
                        current_setup_type = "emerging_leader_breakout"
                        state = "entered"
                    elif has_leader_reentry:
                        working.at[idx, "setup_signal"] = True
                        working.at[idx, "entry_signal"] = True
                        entry_triggered = True
                        active = False
                        active_setup_type = "none"
                        setup_age = 0
                        setup_high = np.nan
                        current_setup_type = "leader_reentry"
                        state = "entered"
                    elif has_late_stage_leader:
                        working.at[idx, "setup_signal"] = True
                        working.at[idx, "entry_signal"] = True
                        entry_triggered = True
                        active = False
                        active_setup_type = "none"
                        setup_age = 0
                        setup_high = np.nan
                        current_setup_type = "late_stage_leader"
                        state = "entered"
                    elif setup_age > self.trigger_window_days:
                        active = False
                        expired = True
                        state = "setup_expired"
                    elif (
                        setup_age >= self.min_setup_days
                        and float(row["close"]) > trigger_level
                        and float(row["volume_ratio_20"]) >= min_volume_ratio
                        and float(row["close_vs_ema_20"])
                        >= (
                            self.emerging_leader_shelf_min_close_vs_ema_20
                            if active_setup_type == "emerging_leader_shelf"
                            else 0
                        )
                        and float(row["close_vs_ema_20"]) <= max_trigger_extension
                        and float(row["rsi_14"]) <= self.max_trigger_rsi_14
                        and continuation_trigger_quality_ok
                    ):
                        working.at[idx, "entry_signal"] = True
                        entry_triggered = True
                        active = False
                        state = "entered"
                    else:
                        state = "setup_ready"

                working.at[idx, "setup_cancelled"] = cancelled
                working.at[idx, "setup_expired"] = expired

                if (
                    has_expansion_leader
                    and not active
                    and not entry_triggered
                ):
                    current_setup_type = "expansion_leader"
                    working.at[idx, "setup_signal"] = True
                    working.at[idx, "entry_signal"] = True
                    working.at[idx, "setup_state"] = "entered"
                    working.at[idx, "setup_type"] = current_setup_type
                    working.at[idx, "trigger_pivot_high"] = pivot_high
                    working.at[idx, "trigger_level"] = float(row.get("prior_20bar_high", 0.0))
                    continue

                if (
                    has_power_breakout
                    and not active
                    and not entry_triggered
                ):
                    current_setup_type = "power_breakout"
                    working.at[idx, "setup_signal"] = True
                    working.at[idx, "entry_signal"] = True
                    working.at[idx, "setup_state"] = "entered"
                    working.at[idx, "setup_type"] = current_setup_type
                    working.at[idx, "trigger_pivot_high"] = pivot_high
                    working.at[idx, "trigger_level"] = float(row.get("prior_3bar_high", 0.0))
                    continue

                if (
                    has_emerging_leader_ignition
                    and not active
                    and not entry_triggered
                ):
                    current_setup_type = "emerging_leader_ignition"
                    working.at[idx, "setup_signal"] = True
                    working.at[idx, "entry_signal"] = True
                    working.at[idx, "setup_state"] = "entered"
                    working.at[idx, "setup_type"] = current_setup_type
                    working.at[idx, "trigger_pivot_high"] = pivot_high
                    working.at[idx, "trigger_level"] = float(row.get("prior_5bar_high", 0.0))
                    continue

                if (
                    has_emerging_leader_breakout
                    and not active
                    and not entry_triggered
                ):
                    current_setup_type = "emerging_leader_breakout"
                    working.at[idx, "setup_signal"] = True
                    working.at[idx, "entry_signal"] = True
                    working.at[idx, "setup_state"] = "entered"
                    working.at[idx, "setup_type"] = current_setup_type
                    working.at[idx, "trigger_pivot_high"] = pivot_high
                    working.at[idx, "trigger_level"] = float(row.get("prior_5bar_high", 0.0))
                    continue

                if (
                    has_leader_reentry
                    and not active
                    and not entry_triggered
                ):
                    current_setup_type = "leader_reentry"
                    working.at[idx, "setup_signal"] = True
                    working.at[idx, "entry_signal"] = True
                    working.at[idx, "setup_state"] = "entered"
                    working.at[idx, "setup_type"] = current_setup_type
                    working.at[idx, "trigger_pivot_high"] = float(row.get("prior_5bar_high", 0.0))
                    working.at[idx, "trigger_level"] = float(row.get("prior_5bar_high", 0.0))
                    continue

                if (
                    has_late_stage_leader
                    and not active
                    and not entry_triggered
                ):
                    current_setup_type = "late_stage_leader"
                    working.at[idx, "setup_signal"] = True
                    working.at[idx, "entry_signal"] = True
                    working.at[idx, "setup_state"] = "entered"
                    working.at[idx, "setup_type"] = current_setup_type
                    working.at[idx, "trigger_pivot_high"] = float(row.get("prior_5bar_high", 0.0))
                    working.at[idx, "trigger_level"] = float(row.get("prior_5bar_high", 0.0))
                    continue

                if fresh_setup_signal.loc[idx] and not active and not entry_triggered:
                    active = True
                    setup_age = 0
                    if emerging_leader_shelf_setup_signal.loc[idx]:
                        active_setup_type = "emerging_leader_shelf"
                    elif continuation_shelf_setup_signal.loc[idx]:
                        active_setup_type = "continuation_shelf"
                    elif continuation_pullback_setup_signal.loc[idx]:
                        active_setup_type = "continuation_pullback"
                    elif expansion_leader_setup_signal.loc[idx]:
                        active_setup_type = "expansion_leader"
                    elif leader_reentry_setup_signal.loc[idx]:
                        active_setup_type = "leader_reentry"
                    elif late_stage_leader_setup_signal.loc[idx]:
                        active_setup_type = "late_stage_leader"
                    elif breakout_setup_signal.loc[idx]:
                        active_setup_type = "breakout"
                    elif continuation_setup_signal.loc[idx]:
                        active_setup_type = "continuation"
                    else:
                        active_setup_type = "continuation"
                    current_setup_type = active_setup_type
                    setup_high = (
                        float(row["high"])
                        if "high" in working.columns and pd.notna(row.get("high"))
                        else float(row["close"])
                    )
                    working.at[idx, "setup_signal"] = True
                    state = "setup_ready"

                if active:
                    working.at[idx, "setup_active"] = True
                    working.at[idx, "setup_age"] = setup_age
                    working.at[idx, "setup_high"] = setup_high
                    working.at[idx, "trigger_pivot_high"] = pivot_high
                    working.at[idx, "trigger_level"] = max(float(setup_high), pivot_high)
                else:
                    working.at[idx, "setup_age"] = 0
                    working.at[idx, "setup_high"] = np.nan
                    working.at[idx, "trigger_pivot_high"] = pivot_high
                    working.at[idx, "trigger_level"] = trigger_level
                    if not (cancelled or expired or entry_triggered):
                        current_setup_type = "none"

                working.at[idx, "setup_type"] = current_setup_type
                working.at[idx, "setup_state"] = state

        return working

    def _breakout_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        entry_cfg = self.entry_logic_config
        leader_regime = scored["leader_regime_signal"].astype(bool)
        breakout_reclaim_ok = (
            (scored["prior_20bar_high"] <= 0)
            | (scored["close_to_prior_20bar_high"] <= self.breakout_max_close_to_prior_20bar_high)
            | (scored["volume_ratio_20"] >= self.breakout_reclaim_exception_volume_ratio)
        )
        breakout_extension_ok = (
            (scored["close_vs_ema_20"] <= self.breakout_extended_max_close_vs_ema_20)
            | (scored["volume_ratio_20"] >= self.breakout_extended_min_volume_ratio)
            | (scored["close_pos"] >= self.breakout_extended_min_close_pos)
        )
        setup_signal = (
            leader_regime
            & (scored["score"] >= self.setup_score_threshold)
            & (scored["trend_stack_bullish"] == 1)
            & ((scored["rs_spy_20"] > 0) | (scored["rs_qqq_20"] > 0))
            & ((scored["rs_spy_20"] + scored["rs_qqq_20"]) >= self.breakout_min_combined_rs)
            & (scored["close_vs_ema_20"] > 0)
            & (scored["close_vs_ema_20"] <= self.max_setup_extension)
            & breakout_extension_ok
            & breakout_reclaim_ok
            & (scored["rsi_14"] >= entry_cfg["breakout_base_min_rsi_14"])
            & (scored["rsi_14"] <= self.max_setup_rsi_14)
            & (scored["donchian_pos_20"] >= entry_cfg["breakout_base_min_donchian_pos"])
            & (scored["donchian_pos_20"] <= self.max_setup_donchian_pos)
            & (scored["close_pos"] >= self.breakout_min_close_pos)
            & (scored["tight_range_5"] <= self.breakout_max_tight_range_5)
            & (scored["volume_ratio_20"] >= entry_cfg["breakout_base_min_volume_ratio"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.breakout_min_room_to_60bar_high,
                allow_breakout=True,
            )
            & (scored["macd_hist"] > 0)
        )

        if self.strict_entry:
            setup_signal &= (
                (scored["cmf_20"] >= 0)
                & (scored["atr_pct_14"] <= entry_cfg["breakout_strict_max_atr_pct_14"])
            )

        return setup_signal

    def _continuation_common_signal(self, scored: pd.DataFrame) -> pd.Series:
        entry_cfg = self.entry_logic_config
        return (
            scored["leader_regime_signal"].astype(bool)
            & (scored["trend_stack_bullish"] == 1)
            & (scored["score"] >= self.continuation_score_threshold)
            & (scored["rs_spy_20"] >= self.continuation_min_rs_spy_20)
            & (scored["rs_qqq_20"] >= self.continuation_min_rs_qqq_20)
            & ((scored["rs_spy_20"] + scored["rs_qqq_20"]) >= self.continuation_min_combined_rs)
            & (scored["close_vs_ema_20"] >= self.continuation_min_close_vs_ema_20)
            & (scored["close_vs_ema_20"] <= self.continuation_max_close_vs_ema_20)
            & (scored["rsi_14"] >= self.continuation_min_rsi_14)
            & (scored["rsi_14"] <= self.continuation_max_rsi_14)
            & (scored["donchian_pos_20"] >= self.continuation_min_donchian_pos)
            & (scored["donchian_pos_20"] <= self.continuation_max_donchian_pos)
            & (scored["pct_from_20d_high"] <= entry_cfg["continuation_pct_from_20d_high_max"])
            & (scored["pct_from_20d_high"] >= entry_cfg["continuation_pct_from_20d_high_min"])
            & (scored["volume_ratio_20"] >= self.continuation_min_volume_ratio)
            & (scored["volume_ratio_20"] <= self.continuation_max_volume_ratio)
            & (scored["atr_pct_14"] <= self.continuation_max_atr_pct_14)
            & (scored["zone_width_20"] <= self.zone_max_width_20)
            & (scored["close_to_prior_20bar_high"] >= (-1.0 * self.zone_reclaim_distance_20))
        )

    def _continuation_shelf_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        entry_cfg = self.entry_logic_config
        setup_signal = (
            self._continuation_common_signal(scored)
            & (scored["close_vs_ema_20"] <= self.continuation_shelf_max_close_vs_ema_20)
            & (scored["tight_range_5"] <= self.continuation_shelf_max_tight_range_5)
            & (scored["close_tightness_3"] <= self.continuation_shelf_max_close_tightness_3)
            & (scored["atr_pct_14"] <= self.continuation_shelf_max_atr_pct_14)
            & (scored["support_cluster_gap"] <= self.continuation_shelf_max_support_gap)
            & (scored["close_pos"] >= entry_cfg["continuation_shelf_min_close_pos"])
            & (scored["pct_chg"] >= entry_cfg["continuation_shelf_min_pct_chg"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.continuation_shelf_min_room_to_60bar_high,
                allow_breakout=False,
            )
        )

        if self.strict_entry:
            setup_signal &= (
                (scored["cmf_20"] >= 0)
                & (scored["close_pos"] >= entry_cfg["continuation_strict_min_close_pos"])
            )

        return setup_signal

    def _continuation_pullback_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        entry_cfg = self.entry_logic_config
        setup_signal = (
            self._continuation_common_signal(scored)
            & (scored["score"] >= self.continuation_pullback_score_threshold)
            & (scored["atr_pct_14"] >= self.continuation_pullback_min_atr_pct_14)
            & (scored["tight_range_5"] <= self.continuation_pullback_max_tight_range_5)
            & (scored["close_tightness_3"] <= self.continuation_pullback_max_close_tightness_3)
            & (scored["close_pos"] >= self.continuation_pullback_min_close_pos)
            & (scored["support_cluster_gap"] <= self.continuation_pullback_max_support_gap)
            & (
                scored["volume_ratio_20"]
                >= max(self.continuation_min_volume_ratio, entry_cfg["continuation_pullback_min_volume_ratio_floor"])
            )
            & (
                scored["volume_ratio_20"]
                <= max(self.continuation_max_volume_ratio, entry_cfg["continuation_pullback_max_volume_ratio_cap"])
            )
            & (scored["pct_chg"] > entry_cfg["continuation_pullback_min_pct_chg"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.continuation_pullback_min_room_to_60bar_high,
                allow_breakout=False,
            )
        )

        if self.strict_entry:
            setup_signal &= (
                (scored["cmf_20"] >= 0)
                & (scored["close_pos"] >= entry_cfg["continuation_strict_min_close_pos"])
            )

        return setup_signal

    def _continuation_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        return self._continuation_shelf_setup_signal(scored) | self._continuation_pullback_setup_signal(scored)

    def _emerging_leader_breakout_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        return (
            scored["emerging_leader_regime_signal"].astype(bool)
            & (scored["score"] >= self.emerging_leader_score_threshold)
            & (scored["rs_acceleration_3"] >= self.emerging_leader_min_rs_acceleration_3)
            & (scored["close"] >= scored["prior_20bar_high"])
            & (scored["close"] > scored["prior_5bar_high"])
            & (scored["pct_chg"] >= self.emerging_leader_breakout_min_pct_chg)
            & (scored["volume_ratio_20"] >= self.emerging_leader_breakout_min_volume_ratio)
            & (scored["close_pos"] >= self.emerging_leader_breakout_min_close_pos)
            & (scored["close_vs_ema_20"] >= self.emerging_leader_shelf_min_close_vs_ema_20)
            & (scored["close_vs_ema_20"] <= self.emerging_leader_breakout_max_close_vs_ema_20)
            & (scored["close_vs_sma_50"] >= self.emerging_leader_breakout_min_close_vs_sma_50)
            & (scored["base_width_20"] <= self.emerging_leader_max_base_width_20)
            & (scored["support_cluster_gap"] <= self.entry_logic_config["power_breakout_max_support_cluster_gap"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.emerging_leader_breakout_min_room_to_60bar_high,
                allow_breakout=True,
            )
            & (scored["macd_hist"] > 0)
        )

    def _emerging_leader_ignition_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        combined_rs = scored["rs_spy_20"] + scored["rs_qqq_20"]
        explosive_extension = (
            (scored["close_vs_ema_20"] > self.emerging_leader_breakout_max_close_vs_ema_20)
            | (scored["support_cluster_gap"] > self.entry_logic_config["power_breakout_max_support_cluster_gap"])
        )
        return (
            scored["emerging_leader_regime_signal"].astype(bool)
            & explosive_extension
            & (scored["score"] >= self.emerging_leader_ignition_score_threshold)
            & (combined_rs >= self.emerging_leader_ignition_min_combined_rs)
            & (scored["rs_acceleration_3"] >= self.emerging_leader_min_rs_acceleration_3)
            & (scored["close"] >= scored["prior_20bar_high"])
            & (scored["close"] > scored["prior_5bar_high"])
            & (scored["pct_chg"] >= self.emerging_leader_ignition_min_pct_chg)
            & (scored["volume_ratio_20"] >= self.emerging_leader_ignition_min_volume_ratio)
            & (scored["close_pos"] >= self.emerging_leader_ignition_min_close_pos)
            & (scored["close_vs_ema_20"] >= self.emerging_leader_ignition_min_close_vs_ema_20)
            & (scored["close_vs_ema_20"] <= self.emerging_leader_ignition_max_close_vs_ema_20)
            & (scored["close_vs_sma_50"] >= self.emerging_leader_ignition_min_close_vs_sma_50)
            & (scored["donchian_pos_20"] >= self.emerging_leader_ignition_min_donchian_pos)
            & (scored["support_cluster_gap"] <= self.emerging_leader_ignition_max_support_gap)
            & (scored["close_to_prior_20bar_high"] <= self.emerging_leader_ignition_max_close_to_prior_20bar_high)
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.emerging_leader_ignition_min_room_to_60bar_high,
                allow_breakout=True,
            )
            & (scored["macd_hist"] > 0)
        )

    def _emerging_leader_shelf_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        entry_cfg = self.entry_logic_config
        return (
            scored["emerging_leader_regime_signal"].astype(bool)
            & (scored["score"] >= self.emerging_leader_score_threshold)
            & (scored["rs_acceleration_3"] >= self.emerging_leader_min_rs_acceleration_3)
            & (scored["close_to_prior_20bar_high"] >= -0.03)
            & (scored["close_vs_ema_20"] >= self.emerging_leader_shelf_min_close_vs_ema_20)
            & (scored["close_vs_ema_20"] <= self.emerging_leader_shelf_max_close_vs_ema_20)
            & (scored["close_vs_sma_50"] >= self.emerging_leader_shelf_min_close_vs_sma_50)
            & (scored["tight_range_5"] <= self.emerging_leader_shelf_max_tight_range_5)
            & (scored["close_tightness_3"] <= self.emerging_leader_shelf_max_close_tightness_3)
            & (scored["support_cluster_gap"] <= self.emerging_leader_shelf_max_support_gap)
            & (scored["base_width_20"] <= self.emerging_leader_max_base_width_20)
            & (scored["base_tightness_20"] <= self.emerging_leader_max_base_tightness_20)
            & (scored["volume_ratio_20"] >= self.continuation_min_volume_ratio)
            & (scored["volume_ratio_20"] <= entry_cfg["emerging_shelf_max_volume_ratio"])
            & (scored["close_pos"] >= entry_cfg["emerging_shelf_min_close_pos"])
            & (scored["pct_chg"] >= entry_cfg["emerging_shelf_min_pct_chg"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.emerging_leader_shelf_min_room_to_60bar_high,
                allow_breakout=False,
            )
        )

    def _expansion_leader_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        expansion_reclaim_ok = (
            (scored["prior_20bar_high"] <= 0)
            | (scored["close_to_prior_20bar_high"] <= self.expansion_leader_max_close_to_prior_20bar_high)
            | (scored["volume_ratio_20"] >= self.expansion_leader_reclaim_exception_volume_ratio)
        )
        return (
            scored["leader_regime_signal"].astype(bool)
            & (scored["score"] >= self.expansion_leader_score_threshold)
            & (scored["trend_stack_bullish"] == 1)
            & ((scored["rs_spy_20"] + scored["rs_qqq_20"]) >= self.expansion_leader_min_combined_rs)
            & (scored["close_vs_ema_20"] >= self.expansion_leader_min_close_vs_ema_20)
            & (scored["close_vs_ema_20"] <= self.expansion_leader_max_close_vs_ema_20)
            & (scored["pct_chg"] >= self.expansion_leader_min_pct_chg)
            & (scored["volume_ratio_20"] >= self.expansion_leader_min_volume_ratio)
            & (scored["close_pos"] >= self.expansion_leader_min_close_pos)
            & (scored["donchian_pos_20"] >= self.expansion_leader_min_donchian_pos)
            & (scored["atr_pct_14"] >= self.expansion_leader_min_atr_pct_14)
            & (scored["tight_range_5"] <= self.expansion_leader_max_tight_range_5)
            & expansion_reclaim_ok
            & (scored["close"] >= scored["prior_20bar_high"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.expansion_leader_min_room_to_60bar_high,
                allow_breakout=True,
            )
            & (scored["macd_hist"] > 0)
        )

    def _leader_reentry_setup_signal(
        self,
        scored: pd.DataFrame,
        *,
        recent_super_leader: pd.Series,
    ) -> pd.Series:
        combined_rs = scored["rs_spy_20"] + scored["rs_qqq_20"]
        return (
            scored["leader_regime_signal"].astype(bool)
            & recent_super_leader
            & (scored["score"] >= self.leader_reentry_score_threshold)
            & (scored["trend_stack_bullish"] == 1)
            & (combined_rs >= self.leader_reentry_min_combined_rs)
            & (scored["close_vs_sma_50"] > 0)
            & (scored["close_vs_ema_20"] >= self.leader_reentry_min_close_vs_ema_20)
            & (scored["close_vs_ema_20"] <= self.leader_reentry_max_close_vs_ema_20)
            & (scored["pct_chg"] >= self.leader_reentry_min_pct_chg)
            & (scored["volume_ratio_20"] >= self.leader_reentry_min_volume_ratio)
            & (scored["close_pos"] >= self.leader_reentry_min_close_pos)
            & (scored["donchian_pos_20"] >= self.leader_reentry_min_donchian_pos)
            & (scored["tight_range_5"] <= self.leader_reentry_max_tight_range_5)
            & (scored["support_cluster_gap"] <= self.leader_reentry_max_support_gap)
            & (scored["close_to_prior_20bar_high"] >= self.leader_reentry_min_close_to_prior_20bar_high)
            & (scored["close"] > scored["prior_5bar_high"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.leader_reentry_min_room_to_60bar_high,
                allow_breakout=True,
            )
            & (scored["macd_hist"] > 0)
            & (~self._expansion_leader_setup_signal(scored))
            & (~self._power_breakout_setup_signal(scored))
        )

    def _late_stage_leader_setup_signal(
        self,
        scored: pd.DataFrame,
        *,
        recent_super_leader: pd.Series,
    ) -> pd.Series:
        entry_cfg = self.entry_logic_config
        combined_rs = scored["rs_spy_20"] + scored["rs_qqq_20"]
        return (
            scored["leader_regime_signal"].astype(bool)
            & recent_super_leader
            & (scored["score"] >= self.late_stage_leader_score_threshold)
            & (scored["trend_stack_bullish"] == 1)
            & (combined_rs >= self.late_stage_leader_min_combined_rs)
            & (scored["close_vs_sma_50"] > 0)
            & (scored["close_vs_ema_20"] >= self.late_stage_leader_min_close_vs_ema_20)
            & (scored["close_vs_ema_20"] <= self.late_stage_leader_max_close_vs_ema_20)
            & (scored["close_pos"] >= self.late_stage_leader_min_close_pos)
            & (scored["donchian_pos_20"] >= self.late_stage_leader_min_donchian_pos)
            & (scored["volume_ratio_20"] >= self.late_stage_leader_min_volume_ratio)
            & (scored["tight_range_5"] <= self.late_stage_leader_max_tight_range_5)
            & (scored["close_tightness_3"] <= self.late_stage_leader_max_close_tightness_3)
            & (scored["support_cluster_gap"] <= self.late_stage_leader_max_support_gap)
            & (scored["close_to_prior_20bar_high"] >= self.late_stage_leader_min_close_to_prior_20bar_high)
            & (scored["close"] > scored["prior_5bar_high"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.late_stage_leader_min_room_to_60bar_high,
                allow_breakout=True,
            )
            & (scored["pct_chg"] >= entry_cfg["late_stage_min_pct_chg"])
            & (scored["macd_hist"] > 0)
            & (~self._expansion_leader_setup_signal(scored))
            & (~self._power_breakout_setup_signal(scored))
            & (~self._leader_reentry_setup_signal(scored, recent_super_leader=recent_super_leader))
        )

    def _zone_reentry_signal(self, scored: pd.DataFrame) -> pd.Series:
        leadership_reentry_ok = (
            scored["confirmed_leader_regime_signal"].astype(bool)
            | scored["emerging_leader_regime_signal"].astype(bool)
        )
        return (
            leadership_reentry_ok
            & (scored["score"] >= self.add_on_min_score)
            & (scored["trend_stack_bullish"] == 1)
            & (scored["close_vs_ema_20"] > 0)
            & (scored["close_vs_sma_50"] > 0)
            & ((scored["rs_spy_20"] > 0) | (scored["rs_qqq_20"] > 0))
            & (scored["zone_width_20"] <= self.zone_max_width_20)
            & (scored["close_to_prior_20bar_high"] >= (-1.0 * self.zone_reclaim_distance_20))
        )

    def _zone_entry_ok(
        self,
        scored: pd.DataFrame,
        *,
        min_room_to_high: float,
        allow_breakout: bool,
    ) -> pd.Series:
        prior_high = scored.get("prior_60bar_high", pd.Series(0.0, index=scored.index))
        short_high = scored.get("prior_20bar_high", pd.Series(0.0, index=scored.index))
        room_to_high = scored.get("room_to_60bar_high", pd.Series(0.0, index=scored.index))
        in_seller_zone = scored.get("in_60bar_seller_zone", pd.Series(False, index=scored.index)).astype(bool)
        broader_overhead_supply = prior_high > (
            short_high * (1.0 + self.entry_logic_config["zone_entry_reclaim_buffer"])
        )
        breaking_out = allow_breakout & (scored["close"] >= prior_high)
        return (
            (prior_high <= 0)
            | (~broader_overhead_supply)
            | breaking_out
            | ((room_to_high >= min_room_to_high) & (~in_seller_zone))
        )

    def _entry_zone_support_level(self, row: pd.Series, setup_type: str) -> float:
        if setup_type == "emerging_leader_shelf":
            return float(row.get("prior_5bar_low", 0.0))
        if setup_type == "continuation_shelf":
            return float(row.get("prior_5bar_low", 0.0))
        if setup_type == "continuation_pullback":
            return float(row.get("prior_20bar_low", 0.0))
        if setup_type in {
            "breakout",
            "power_breakout",
            "expansion_leader",
            "emerging_leader_breakout",
            "emerging_leader_ignition",
        }:
            return float(max(row.get("prior_20bar_high", 0.0), row.get("prior_5bar_low", 0.0)))
        if setup_type in {"leader_reentry", "late_stage_leader"}:
            return float(max(row.get("prior_5bar_low", 0.0), row.get("prior_20bar_low", 0.0)))
        return float(row.get("prior_20bar_low", 0.0))

    def _power_breakout_setup_signal(self, scored: pd.DataFrame) -> pd.Series:
        entry_cfg = self.entry_logic_config
        calm_breakout = (
            (scored["atr_pct_14"] <= self.power_breakout_calm_max_atr_pct_14)
            & (scored["close_vs_ema_20"] <= self.power_breakout_calm_max_close_vs_ema_20)
            & (scored["tight_range_5"] <= self.power_breakout_max_tight_range_5)
        )
        explosive_breakout = (
            (scored["pct_chg"] >= self.power_breakout_explosive_min_pct_chg)
            & (scored["volume_ratio_20"] >= self.power_breakout_explosive_min_volume_ratio)
        )
        setup_signal = (
            scored["leader_regime_signal"].astype(bool)
            & (scored["score"] >= self.power_breakout_score_threshold)
            & (scored["trend_stack_bullish"] == 1)
            & (scored["rs_spy_20"] >= self.power_breakout_min_rs_spy_20)
            & (scored["rs_qqq_20"] >= self.power_breakout_min_rs_qqq_20)
            & ((scored["rs_spy_20"] + scored["rs_qqq_20"]) >= self.power_breakout_min_combined_rs)
            & (scored["close_vs_ema_20"] >= self.power_breakout_min_close_vs_ema_20)
            & (scored["close_vs_ema_20"] <= self.power_breakout_max_close_vs_ema_20)
            & (scored["rsi_14"] >= self.power_breakout_min_rsi_14)
            & (scored["rsi_14"] <= self.power_breakout_max_rsi_14)
            & (scored["donchian_pos_20"] >= self.power_breakout_min_donchian_pos)
            & (scored["volume_ratio_20"] >= self.power_breakout_trigger_volume_ratio)
            & (scored["close_pos"] >= self.power_breakout_min_close_pos_strict)
            & (scored["pct_chg"] >= self.power_breakout_min_pct_chg)
            & (scored["pct_from_20d_high"] >= self.power_breakout_min_pct_from_20d_high)
            & (scored["macd_hist"] > 0)
            & (scored["close"] > scored["prior_3bar_high"])
            & (scored["close"] >= scored["prior_20bar_high"])
            & self._zone_entry_ok(
                scored,
                min_room_to_high=self.power_breakout_min_room_to_60bar_high,
                allow_breakout=True,
            )
            & (scored["support_cluster_gap"] <= entry_cfg["power_breakout_max_support_cluster_gap"])
            & (calm_breakout | explosive_breakout)
        )

        if self.strict_entry:
            setup_signal &= (
                (scored["cmf_20"] >= 0)
                & (scored["atr_pct_14"] <= entry_cfg["power_breakout_strict_max_atr_pct_14"])
            )

        return setup_signal

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        working = df.copy()
        rename_map = {}
        for column in working.columns:
            if column == "Date":
                rename_map[column] = "Date"
            else:
                rename_map[column] = self._normalize_name(column)
        working = working.rename(columns=rename_map)

        if "date" in working.columns and "Date" not in working.columns:
            working = working.rename(columns={"date": "Date"})
        if "symbol" in working.columns and "ticker" not in working.columns:
            working = working.rename(columns={"symbol": "ticker"})

        if "Date" not in working.columns:
            raise ValueError("Input DataFrame must include a Date column.")
        if "ticker" not in working.columns:
            raise ValueError("Input DataFrame must include a ticker column.")

        working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
        working = working[working["Date"].notna()].copy()
        working["ticker"] = working["ticker"].astype(str).str.upper()
        return working.sort_values(["ticker", "Date"]).reset_index(drop=True)

    def _fill_feature_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        working = df.copy()
        for column, default_value in self.COLUMN_DEFAULTS.items():
            if column not in working.columns:
                working[column] = default_value
            working[column] = pd.to_numeric(working[column], errors="coerce").fillna(default_value)
        return working.sort_values(["ticker", "Date"]).reset_index(drop=True)

    def _has_feature_columns(self, df: pd.DataFrame) -> bool:
        return all(column in df.columns for column in self.FEATURE_COLUMNS)

    def _has_raw_price_columns(self, df: pd.DataFrame) -> bool:
        return all(column in df.columns for column in self.RAW_PRICE_COLUMNS)

    def _require_columns(self, df: pd.DataFrame, columns: tuple[str, ...]) -> None:
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _coerce_numeric_columns(self, df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
        working = df.copy()
        for column in columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        return working

    def _resolve_benchmark_close(
        self,
        working: pd.DataFrame,
        benchmark_ticker: str,
        benchmark_df: pd.DataFrame | None,
    ) -> pd.Series | None:
        if benchmark_df is not None:
            benchmark = self._standardize_dataframe(benchmark_df)
            self._require_columns(benchmark, ("Date", "ticker", "close"))
            benchmark = self._coerce_numeric_columns(benchmark, ("close",))
            benchmark = benchmark[benchmark["ticker"] == benchmark_ticker]
            if benchmark.empty:
                return None
            return benchmark.set_index("Date")["close"].sort_index()

        embedded = working[working["ticker"] == benchmark_ticker]
        if embedded.empty or "close" not in embedded.columns:
            return None
        embedded = self._coerce_numeric_columns(embedded, ("close",))
        return embedded.set_index("Date")["close"].sort_index()

    def _build_single_ticker_features(
        self,
        ticker_df: pd.DataFrame,
        *,
        spy_close: pd.Series | None,
        qqq_close: pd.Series | None,
    ) -> pd.DataFrame:
        ticker_df = ticker_df.sort_values("Date").reset_index(drop=True)
        ticker_df = self._coerce_numeric_columns(ticker_df, self.RAW_PRICE_COLUMNS)

        close = ticker_df["close"]
        open_ = ticker_df["open"]
        high = ticker_df["high"]
        low = ticker_df["low"]
        volume = ticker_df["volume"]
        typical_price = (high + low + close) / 3.0

        ticker_df["avg_vol_20"] = volume.rolling(self.volume_short_window, min_periods=1).mean()
        ticker_df["avg_vol_50"] = volume.rolling(self.volume_long_window, min_periods=1).mean()

        moving_average_periods = (
            (self.ma_short_period, 10),
            (self.ma_medium_period, 20),
            (self.ma_long_period, 50),
        )
        for period, alias in moving_average_periods:
            sma_series = close.rolling(period, min_periods=1).mean()
            ema_series = close.ewm(span=period, adjust=False).mean()
            ticker_df[f"sma_{alias}"] = sma_series
            ticker_df[f"ema_{alias}"] = ema_series
            ticker_df[f"close_vs_sma_{alias}"] = self._safe_divide(close - sma_series, sma_series, 0.0)
            ticker_df[f"close_vs_ema_{alias}"] = self._safe_divide(close - ema_series, ema_series, 0.0)

        ticker_df["trend_stack_bullish"] = (
            (close > ticker_df["ema_10"])
            & (ticker_df["ema_10"] > ticker_df["ema_20"])
            & (ticker_df["ema_20"] > ticker_df["ema_50"])
        ).astype(int)
        ticker_df["trend_stack_bearish"] = (
            (close < ticker_df["ema_10"])
            & (ticker_df["ema_10"] < ticker_df["ema_20"])
            & (ticker_df["ema_20"] < ticker_df["ema_50"])
        ).astype(int)
        ticker_df["lr_slope_log_63"] = rolling_log_regression_slope(close, self.trend_slope_short_window)
        ticker_df["lr_slope_log_126"] = rolling_log_regression_slope(close, self.trend_slope_long_window)
        ticker_df["lr_r2_63"] = rolling_log_regression_r2(close, self.trend_slope_short_window)
        ticker_df["lr_r2_126"] = rolling_log_regression_r2(close, self.trend_slope_long_window)
        ticker_df["ema_50_slope_20"] = rolling_log_regression_slope(
            ticker_df["ema_50"],
            self.trend_ema_slope_window,
        )
        ticker_df["pct_above_ema_50_63"] = rolling_positive_fraction(
            close - ticker_df["ema_50"],
            self.trend_persistence_window,
        )
        ticker_df["max_drawdown_63"] = rolling_max_drawdown(close, self.trend_drawdown_window)

        ticker_df["roll_high_20"] = high.rolling(self.breakout_window, min_periods=1).max()
        ticker_df["roll_low_20"] = low.rolling(self.breakout_window, min_periods=1).min()
        ticker_df["pct_from_20d_high"] = self._safe_divide(
            close - ticker_df["roll_high_20"],
            ticker_df["roll_high_20"],
            0.0,
        )
        ticker_df["pct_from_20d_low"] = self._safe_divide(
            close - ticker_df["roll_low_20"],
            ticker_df["roll_low_20"],
            0.0,
        )

        ticker_df["donchian_high_20"] = high.rolling(self.donchian_window, min_periods=1).max()
        ticker_df["donchian_low_20"] = low.rolling(self.donchian_window, min_periods=1).min()
        ticker_df["donchian_pos_20"] = self._safe_divide(
            close - ticker_df["donchian_low_20"],
            ticker_df["donchian_high_20"] - ticker_df["donchian_low_20"],
            0.5,
        )

        ticker_df["bb_mid_20"] = close.rolling(self.bollinger_window, min_periods=1).mean()
        bb_std_20 = close.rolling(self.bollinger_window, min_periods=1).std()
        ticker_df["bb_upper_20"] = ticker_df["bb_mid_20"] + (self.bollinger_std_mult * bb_std_20)
        ticker_df["bb_lower_20"] = ticker_df["bb_mid_20"] - (self.bollinger_std_mult * bb_std_20)
        ticker_df["bb_pct_b_20"] = self._safe_divide(
            close - ticker_df["bb_lower_20"],
            ticker_df["bb_upper_20"] - ticker_df["bb_lower_20"],
            0.5,
        )

        ticker_df["rsi_14"] = self._rsi(close, self.rsi_fast_period)
        ticker_df["rsi_21"] = self._rsi(close, self.rsi_slow_period)
        ticker_df["smoothed_rsi_ema21_rsi10"] = self._rsi(
            close.ewm(span=self.smoothed_rsi_ema_span, adjust=False).mean(),
            self.smoothed_rsi_period,
        )

        ema12 = close.ewm(span=self.macd_fast_period, adjust=False).mean()
        ema26 = close.ewm(span=self.macd_slow_period, adjust=False).mean()
        ticker_df["macd_line"] = ema12 - ema26
        ticker_df["macd_signal"] = ticker_df["macd_line"].ewm(span=self.macd_signal_period, adjust=False).mean()
        ticker_df["macd_hist"] = ticker_df["macd_line"] - ticker_df["macd_signal"]

        rolling_low_14 = low.rolling(self.stochastic_window, min_periods=1).min()
        rolling_high_14 = high.rolling(self.stochastic_window, min_periods=1).max()
        stochastic_range = rolling_high_14 - rolling_low_14
        ticker_df["stoch_k_14"] = 100.0 * self._safe_divide(close - rolling_low_14, stochastic_range, 0.0)
        ticker_df["stoch_d_3"] = ticker_df["stoch_k_14"].rolling(self.stochastic_signal_window, min_periods=1).mean()
        ticker_df["williams_r_14"] = -100.0 * self._safe_divide(rolling_high_14 - close, stochastic_range, 0.0)

        typical_sma_20 = typical_price.rolling(self.cci_window, min_periods=1).mean()
        mean_deviation = typical_price.rolling(self.cci_window, min_periods=1).apply(
            lambda values: np.mean(np.abs(values - values.mean())),
            raw=True,
        )
        ticker_df["cci_20"] = self._safe_divide(
            typical_price - typical_sma_20,
            self.cci_constant * mean_deviation,
            0.0,
        )

        prev_close = close.shift(1)
        ticker_df["tr"] = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        ticker_df["atr_14"] = ticker_df["tr"].rolling(self.atr_window, min_periods=1).mean()
        ticker_df["atr_pct_14"] = self._safe_divide(ticker_df["atr_14"], close, 0.0)

        plus_di, minus_di, adx = self._directional_indicators(high, low, close, self.directional_window)
        ticker_df["plus_di_14"] = plus_di
        ticker_df["minus_di_14"] = minus_di
        ticker_df["adx_14"] = adx

        ticker_df["pct_chg"] = close.pct_change().fillna(0.0)
        ticker_df["close_pos"] = self._safe_divide(close - low, high - low, 0.0)
        ticker_df["body"] = (close - open_).abs()
        ticker_df["realized_vol_20"] = (
            close.pct_change().rolling(self.realized_vol_window, min_periods=1).std(ddof=0)
            * np.sqrt(self.realized_vol_annualization)
        )

        ticker_df["volume_ratio_20"] = self._safe_divide(volume, ticker_df["avg_vol_20"], 0.0)
        ticker_df["volume_ratio_50"] = self._safe_divide(volume, ticker_df["avg_vol_50"], 0.0)
        volume_std_20 = volume.rolling(self.volume_short_window, min_periods=1).std(ddof=0)
        ticker_df["volume_zscore_20"] = self._safe_divide(
            volume - ticker_df["avg_vol_20"],
            volume_std_20,
            0.0,
        )

        money_flow_multiplier = self._safe_divide(
            ((close - low) - (high - close)),
            (high - low),
            0.0,
        )
        money_flow_volume = money_flow_multiplier * volume
        ticker_df["cmf_20"] = self._safe_divide(
            money_flow_volume.rolling(self.cmf_window, min_periods=1).sum(),
            volume.rolling(self.cmf_window, min_periods=1).sum(),
            0.0,
        )

        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0.0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0.0)
        ticker_df["mfi_14"] = 100.0 - (
            100.0
            / (
                1.0
                + self._safe_divide(
                    positive_flow.rolling(self.mfi_window, min_periods=1).sum(),
                    negative_flow.rolling(self.mfi_window, min_periods=1).sum(),
                    0.0,
                )
            )
        )

        self._add_relative_strength_features(ticker_df, "spy", spy_close)
        self._add_relative_strength_features(ticker_df, "qqq", qqq_close)

        ticker_df = self._fill_feature_defaults(ticker_df)
        return ticker_df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    def _add_relative_strength_features(
        self,
        ticker_df: pd.DataFrame,
        benchmark_name: str,
        benchmark_close: pd.Series | None,
    ) -> None:
        if benchmark_close is None:
            ticker_df[f"price_to_{benchmark_name}"] = 0.0
            ticker_df[f"rs_{benchmark_name}_20"] = 0.0
            ticker_df[f"rs_{benchmark_name}_50"] = 0.0
            return

        aligned = benchmark_close.reindex(ticker_df["Date"]).ffill()
        aligned = pd.Series(aligned.values, index=ticker_df.index, dtype=float)
        price_to_benchmark = self._safe_divide(ticker_df["close"], aligned, 0.0)
        ticker_df[f"price_to_{benchmark_name}"] = price_to_benchmark
        ticker_df[f"rs_{benchmark_name}_20"] = price_to_benchmark.pct_change(self.rs_short_window).fillna(0.0)
        ticker_df[f"rs_{benchmark_name}_50"] = price_to_benchmark.pct_change(self.rs_long_window).fillna(0.0)

    def _coerce_row(self, row: pd.Series) -> dict[str, float]:
        coerced: dict[str, float] = {}
        for column, default_value in self.COLUMN_DEFAULTS.items():
            value = row[column] if column in row.index else default_value
            coerced[column] = default_value if pd.isna(value) else float(value)
        return coerced

    def _exit_reason(self, row: pd.Series, position: _BacktestPosition) -> str | None:
        open_gain = (
            (float(row["close"]) / position.entry_price) - 1.0
            if position.entry_price > 0
            else 0.0
        )
        trend_hold = (
            position.entry_score >= self.trend_hold_entry_score_threshold
            or open_gain >= self.trend_hold_gain_threshold
            or (
                position.days_held >= self.trend_hold_min_days
                and float(row["score"]) >= self.trend_hold_min_score
            )
        )
        trend_strength_intact = (
            float(row["score"]) >= self.trend_hold_min_score
            and float(row["close_vs_ema_20"]) > 0
            and ((float(row["rs_spy_20"]) > 0) or (float(row["rs_qqq_20"]) > 0))
        )
        continuation_setup = self._is_continuation_setup_type(position.setup_type)
        if long_zone_broken(float(row["close"]), position.zone_support, self.zone_exit_tolerance_pct):
            return "zone_support_fail"
        if float(row.get("roll_low_10", 0.0)) > 0 and float(row["close"]) <= float(row["roll_low_10"]):
            return "break_10d_low"

        if bool(row.get("close_below_sma50", False)):
            return "structure_sma50_fail"

        if continuation_setup and bool(row.get("close_below_ema20", False)):
            return "structure_ema20_fail"

        if position.setup_type == "breakout" and position.days_held <= self.breakout_early_failure_max_days:
            breakout_followthrough_failed = (
                open_gain <= self.breakout_early_failure_max_open_gain
                and (
                    float(row["score"]) <= self.breakout_early_failure_max_score
                    or float(row["close_vs_ema_20"]) <= self.breakout_early_failure_min_close_vs_ema_20
                    or bool(row.get("relative_weak", False))
                )
            )
            if breakout_followthrough_failed:
                return "breakout_failed_followthrough"

        if position.setup_type == "expansion_leader" and position.days_held <= self.expansion_leader_early_failure_max_days:
            expansion_followthrough_failed = self._early_followthrough_failed(
                row,
                open_gain=open_gain,
                max_open_gain=self.expansion_leader_early_failure_max_open_gain,
                max_score=self.expansion_leader_early_failure_max_score,
                min_close_vs_ema_20=self.expansion_leader_early_failure_min_close_vs_ema_20,
                min_close_pos=(
                    self.expansion_leader_early_failure_min_close_pos
                    if self.enable_aggressive_early_failure
                    else None
                ),
            )
            if expansion_followthrough_failed:
                return "expansion_failed_followthrough"

        if (
            self.enable_aggressive_early_failure
            and position.setup_type in {"power_breakout", "emerging_leader_ignition"}
            and position.days_held <= self.power_breakout_early_failure_max_days
        ):
            power_breakout_followthrough_failed = self._early_followthrough_failed(
                row,
                open_gain=open_gain,
                max_open_gain=self.power_breakout_early_failure_max_open_gain,
                max_score=self.power_breakout_early_failure_max_score,
                min_close_vs_ema_20=self.power_breakout_early_failure_min_close_vs_ema_20,
                min_close_pos=self.power_breakout_early_failure_min_close_pos,
            )
            if power_breakout_followthrough_failed:
                return "power_breakout_failed_followthrough"

        if (
            self.enable_bb_micro_failure
            and position.setup_type in {"power_breakout", "expansion_leader", "emerging_leader_ignition"}
            and position.days_held <= self.bb_micro_failure_max_days
            and bool(row.get("bb_micro_support_fail", False))
        ):
            return "bb_micro_support_fail"

        if (
            self.enable_medium_confirm_failure
            and position.setup_type in {"power_breakout", "expansion_leader", "emerging_leader_ignition"}
            and position.days_held <= self.medium_confirm_failure_max_days
            and bool(row.get("medium_confirm_failure", False))
        ):
            return "medium_confirm_failure"

        if bool(row.get("close_below_ema20_2d", False)) and (
            not trend_hold
            or float(row["score"]) < self.trend_hold_min_score
            or float(row["close_vs_sma_50"]) < 0
        ):
            return "structure_ema20_fail"

        if bool(row.get("soft_score_fail_2d", False)):
            return "soft_score_fail"

        if continuation_setup and bool(row.get("relative_weak", False)) and (
            float(row["score"]) < max(self.trend_hold_relative_weak_exit_score, position.entry_score - 10.0)
            or float(row["close_vs_ema_20"]) <= 0
        ):
            return "relative_weakness"

        if bool(row.get("relative_weak_2d", False)) and (
            not trend_hold
            or float(row["score"]) < self.trend_hold_relative_weak_exit_score
            or float(row["close_vs_ema_20"]) <= 0
        ):
            return "relative_weakness"

        trailing_stop_armed = (
            open_gain >= self.trailing_stop_min_gain_to_arm
            or position.days_held >= self.trailing_stop_min_days_to_arm
        )
        if trend_hold and trend_strength_intact:
            trailing_atr_multiple = self.trend_hold_trailing_atr_multiple
        else:
            trailing_atr_multiple = {
                "breakout": self.breakout_trailing_atr_multiple,
                "continuation": self.continuation_shelf_trailing_atr_multiple,
                "continuation_shelf": self.continuation_shelf_trailing_atr_multiple,
                "continuation_pullback": self.continuation_pullback_trailing_atr_multiple,
                "expansion_leader": self.expansion_leader_trailing_atr_multiple,
                "leader_reentry": self.expansion_leader_trailing_atr_multiple,
                "late_stage_leader": self.power_breakout_trailing_atr_multiple,
                "emerging_leader_ignition": self.power_breakout_trailing_atr_multiple,
                "emerging_leader_breakout": self.breakout_trailing_atr_multiple,
                "emerging_leader_shelf": self.continuation_shelf_trailing_atr_multiple,
                "power_breakout": self.power_breakout_trailing_atr_multiple,
            }.get(position.setup_type, self.breakout_trailing_atr_multiple)
        trailing_stop = position.highest_close - (trailing_atr_multiple * float(row["atr_14"]))
        if trailing_stop_armed and float(row["close"]) <= trailing_stop:
            return "atr_trailing_stop"

        if self.use_atr_stop:
            atr_stop = position.entry_price - (2.0 * float(row["atr_14"]))
            if float(row["close"]) <= atr_stop:
                return "legacy_atr_stop"

        if self.use_time_stop:
            if (
                position.days_held >= self.time_stop_days
                and not position.has_new_high
                and not position.score_improved
            ):
                return "time_stop"

        return None

    @staticmethod
    def _early_followthrough_failed(
        row: pd.Series,
        *,
        open_gain: float,
        max_open_gain: float,
        max_score: float,
        min_close_vs_ema_20: float,
        min_close_pos: float | None = None,
    ) -> bool:
        if open_gain > max_open_gain:
            return False
        if float(row["score"]) <= max_score:
            return True
        if float(row["close_vs_ema_20"]) <= min_close_vs_ema_20:
            return True
        if min_close_pos is not None and float(row.get("close_pos", 1.0)) <= min_close_pos:
            return True
        return bool(row.get("relative_weak", False))

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return float(max(low, min(high, value)))

    @staticmethod
    def _is_continuation_setup_type(setup_type: str) -> bool:
        return str(setup_type).startswith("continuation")

    @staticmethod
    def _is_aggressive_setup_type(setup_type: str) -> bool:
        return str(setup_type) in {"power_breakout", "expansion_leader", "emerging_leader_ignition"}

    @staticmethod
    def _is_emerging_setup_type(setup_type: str) -> bool:
        return str(setup_type) in {"emerging_leader_breakout", "emerging_leader_shelf", "emerging_leader_ignition"}

    def _uses_starter_sizing(self, setup_type: str) -> bool:
        normalized = str(setup_type)
        if normalized == "emerging_leader_ignition":
            return True
        return self.enable_aggressive_starter_sizing and self._is_aggressive_setup_type(normalized)

    def starter_size_multiplier(self, setup_type: str) -> float:
        normalized = str(setup_type)
        if normalized == "emerging_leader_ignition":
            return self.emerging_leader_ignition_starter_fraction
        if self.enable_aggressive_starter_sizing and self._is_aggressive_setup_type(normalized):
            return self.aggressive_starter_fraction
        return 1.0

    @classmethod
    def _uses_medium_trigger_window(cls, setup_type: str) -> bool:
        return cls._is_continuation_setup_type(setup_type) or str(setup_type) == "emerging_leader_shelf"

    def _confirmed_leader_score(self, scored: pd.DataFrame) -> pd.Series:
        return (
            (scored["trend_stack_bullish"] == 1).astype(int) * 2
            + (scored["close_vs_ema_50"] > 0).astype(int)
            + (scored["close_vs_sma_50"] > 0).astype(int)
            + (scored["ema_50_slope_20"] >= self.leader_min_ema_50_slope_20).astype(int) * 2
            + (scored["lr_slope_log_63"] >= self.leader_min_lr_slope_log_63).astype(int) * 2
            + (scored["lr_slope_log_126"] >= self.leader_min_lr_slope_log_126).astype(int) * 2
            + (scored["lr_r2_63"] >= self.leader_min_lr_r2_63).astype(int)
            + (scored["lr_r2_126"] >= self.leader_min_lr_r2_126).astype(int)
            + (scored["pct_above_ema_50_63"] >= self.leader_min_pct_above_ema_50_63).astype(int) * 2
            + (scored["max_drawdown_63"] >= (-1.0 * self.leader_max_drawdown_63)).astype(int)
            + (scored["rs_spy_20"] > 0).astype(int)
            + (scored["rs_qqq_20"] > 0).astype(int)
        ).astype(float)

    def _confirmed_leader_regime_signal(self, scored: pd.DataFrame) -> pd.Series:
        combined_rs = scored["rs_spy_20"] + scored["rs_qqq_20"]
        leader_score = (
            scored["confirmed_leader_score"]
            if "confirmed_leader_score" in scored.columns
            else self._confirmed_leader_score(scored)
        )
        return (
            (scored["trend_stack_bullish"] == 1)
            & (scored["close_vs_ema_50"] > 0)
            & (scored["close_vs_sma_50"] > 0)
            & (combined_rs >= self.leader_min_combined_rs)
            & (leader_score >= self.leader_regime_min_score)
        )

    def _emerging_leader_score(self, scored: pd.DataFrame) -> pd.Series:
        combined_rs = scored["rs_spy_20"] + scored["rs_qqq_20"]
        base_score = scored.get("score", pd.Series(0.0, index=scored.index, dtype=float))
        return (
            (base_score >= self.emerging_leader_score_threshold).astype(int) * 2
            + (combined_rs >= self.emerging_leader_min_combined_rs).astype(int) * 2
            + (scored["rs_acceleration_3"] >= self.emerging_leader_min_rs_acceleration_3).astype(int) * 2
            + (scored["lr_slope_log_63"] >= self.emerging_leader_min_lr_slope_log_63).astype(int)
            + (scored["lr_r2_63"] >= self.emerging_leader_min_lr_r2_63).astype(int)
            + (scored["ema_50_slope_20"] >= self.emerging_leader_min_ema_50_slope_20).astype(int)
            + (scored["pct_above_ema_50_63"] >= self.emerging_leader_min_pct_above_ema_50_63).astype(int)
            + (scored["max_drawdown_63"] >= (-1.0 * self.emerging_leader_max_drawdown_63)).astype(int)
            + (scored["base_width_20"] <= self.emerging_leader_max_base_width_20).astype(int)
            + (scored["base_tightness_20"] <= self.emerging_leader_max_base_tightness_20).astype(int)
            + (scored["close_vs_ema_20"] >= self.emerging_leader_shelf_min_close_vs_ema_20).astype(int)
            + (scored["close_vs_sma_50"] >= self.emerging_leader_shelf_min_close_vs_sma_50).astype(int)
            + (scored["close_to_prior_20bar_high"] >= -0.04).astype(int)
        ).astype(float)

    def _emerging_leader_regime_signal(self, scored: pd.DataFrame) -> pd.Series:
        combined_rs = scored["rs_spy_20"] + scored["rs_qqq_20"]
        base_score = scored.get("score", pd.Series(0.0, index=scored.index, dtype=float))
        emerging_score = (
            scored["emerging_leader_score"]
            if "emerging_leader_score" in scored.columns
            else self._emerging_leader_score(scored)
        )
        confirmed = (
            scored["confirmed_leader_regime_signal"].astype(bool)
            if "confirmed_leader_regime_signal" in scored.columns
            else self._confirmed_leader_regime_signal(scored)
        )
        return (
            (~confirmed)
            & (base_score >= self.emerging_leader_score_threshold)
            & (combined_rs >= self.emerging_leader_min_combined_rs)
            & (scored["rs_acceleration_3"] >= self.emerging_leader_min_rs_acceleration_3)
            & (scored["close_vs_ema_20"] >= self.emerging_leader_shelf_min_close_vs_ema_20)
            & (scored["close_vs_sma_50"] >= self.emerging_leader_shelf_min_close_vs_sma_50)
            & (scored["max_drawdown_63"] >= (-1.0 * self.emerging_leader_max_drawdown_63))
            & (emerging_score >= self.emerging_leader_min_score)
        )

    def _setup_regime_active(self, row: pd.Series, setup_type: str) -> bool:
        if self._is_emerging_setup_type(setup_type):
            return bool(row.get("emerging_leader_regime_signal", False))
        return bool(row.get("leader_regime_signal", False))

    def _super_leader_signal(self, scored: pd.DataFrame) -> pd.Series:
        combined_rs = scored["rs_spy_20"] + scored["rs_qqq_20"]
        return (
            scored["leader_regime_signal"].astype(bool)
            & (scored["score"] >= self.super_leader_score_threshold)
            & (scored["trend_stack_bullish"] == 1)
            & (combined_rs >= self.super_leader_min_combined_rs)
            & (scored["close_vs_sma_50"] >= self.super_leader_min_close_vs_sma_50)
            & (scored["donchian_pos_20"] >= self.super_leader_min_donchian_pos)
        )

    @staticmethod
    def _level_from_relative_close(close: pd.Series, relative: pd.Series) -> pd.Series:
        denominator = 1.0 + pd.to_numeric(relative, errors="coerce").fillna(0.0)
        denominator = denominator.where(denominator > 0, np.nan)
        level = pd.to_numeric(close, errors="coerce").fillna(0.0) / denominator
        return level.fillna(0.0)

    def _resolved_allocation_mode(self) -> str:
        if self.enable_risk_position_sizing and self.allocation_mode == "baseline":
            return "hybrid_risk_capped"
        return str(self.allocation_mode)

    @classmethod
    def _deep_merge_dicts(cls, base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for key, value in base.items():
            if isinstance(value, dict):
                override_value = overrides.get(key, {})
                if isinstance(override_value, dict):
                    merged[key] = cls._deep_merge_dicts(value, override_value)
                else:
                    merged[key] = value
            else:
                merged[key] = overrides.get(key, value)
        for key, value in overrides.items():
            if key not in merged:
                merged[key] = value
        return merged

    def _setup_target_weight(self, setup_type: str) -> float:
        return {
            "power_breakout": self.tiered_weight_power_breakout,
            "expansion_leader": self.tiered_weight_expansion_leader,
            "late_stage_leader": self.tiered_weight_late_stage_leader,
            "emerging_leader_ignition": self.tiered_weight_emerging_leader_ignition,
            "emerging_leader_breakout": self.tiered_weight_emerging_leader_breakout,
            "breakout": self.tiered_weight_breakout,
            "emerging_leader_shelf": self.tiered_weight_emerging_leader_shelf,
            "continuation_shelf": self.tiered_weight_continuation_shelf,
            "continuation_pullback": self.tiered_weight_continuation_pullback,
            "leader_reentry": self.tiered_weight_leader_reentry,
        }.get(str(setup_type), self.tiered_weight_default)

    def _remaining_stock_capacity(self, current_market_value: float) -> float:
        if self.max_allocation_per_stock <= 0:
            return float("inf")
        return max(self.max_allocation_per_stock - current_market_value, 0.0)

    @staticmethod
    def _normalize_name(name: str) -> str:
        return (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    @staticmethod
    def _safe_divide(
        numerator: pd.Series | float,
        denominator: pd.Series | float,
        default: float,
    ) -> pd.Series | float:
        if isinstance(denominator, pd.Series):
            denominator_clean = denominator.replace(0, np.nan)
            result = numerator / denominator_clean
            return result.replace([np.inf, -np.inf], np.nan).fillna(default)
        if denominator == 0 or pd.isna(denominator):
            return default
        result = numerator / denominator
        if pd.isna(result) or result in (np.inf, -np.inf):
            return default
        return result

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(0.0)

    @staticmethod
    def _directional_indicators(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=high.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=high.index,
        )
        true_range = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        smoothed_tr = true_range.rolling(period, min_periods=1).mean()
        plus_di = (100.0 * plus_dm.rolling(period, min_periods=1).mean() / smoothed_tr.replace(0, np.nan)).fillna(0.0)
        minus_di = (100.0 * minus_dm.rolling(period, min_periods=1).mean() / smoothed_tr.replace(0, np.nan)).fillna(0.0)
        dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
        adx = dx.rolling(period, min_periods=1).mean().fillna(0.0)
        return plus_di, minus_di, adx
