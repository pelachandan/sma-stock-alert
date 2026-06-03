import pandas as pd

from src.analysis.rally_pattern_strategy import RallyPatternStrategy, _BacktestPosition


def _strong_row(date: str, ticker: str, close: float = 100.0) -> dict:
    return {
        "ticker": ticker,
        "Date": pd.Timestamp(date),
        "close": close,
        "close_vs_sma_10": 0.01,
        "close_vs_sma_20": 0.02,
        "close_vs_sma_50": 0.03,
        "close_vs_ema_10": 0.01,
        "close_vs_ema_20": 0.02,
        "close_vs_ema_50": 0.03,
        "trend_stack_bullish": 1,
        "trend_stack_bearish": 0,
        "pct_from_20d_high": -0.005,
        "pct_from_20d_low": 0.10,
        "donchian_pos_20": 0.85,
        "bb_pct_b_20": 0.90,
        "rsi_14": 66.0,
        "rsi_21": 56.0,
        "smoothed_rsi_ema21_rsi10": 60.0,
        "macd_line": 1.50,
        "macd_signal": 0.50,
        "macd_hist": 1.00,
        "stoch_k_14": 85.0,
        "stoch_d_3": 65.0,
        "williams_r_14": -20.0,
        "cci_20": 55.0,
        "adx_14": 25.0,
        "plus_di_14": 30.0,
        "minus_di_14": 15.0,
        "pct_chg": 0.015,
        "close_pos": 0.80,
        "body": 1.50,
        "tr": 3.00,
        "atr_14": 3.00,
        "atr_pct_14": 0.03,
        "realized_vol_20": 0.20,
        "volume_ratio_20": 1.20,
        "volume_ratio_50": 1.05,
        "volume_zscore_20": 1.20,
        "cmf_20": 0.06,
        "mfi_14": 55.0,
        "rs_spy_20": 0.05,
        "rs_spy_50": 0.02,
        "rs_qqq_20": 0.05,
        "rs_qqq_50": 0.02,
        "lr_slope_log_63": 0.0035,
        "lr_slope_log_126": 0.0020,
        "lr_r2_63": 0.82,
        "lr_r2_126": 0.74,
        "ema_50_slope_20": 0.0018,
        "pct_above_ema_50_63": 0.90,
        "max_drawdown_63": -0.08,
    }


def _moderate_entry_row(date: str, ticker: str, close: float = 100.0) -> dict:
    row = _strong_row(date, ticker, close)
    row.update(
        {
            "close_vs_sma_50": -0.01,
            "close_vs_ema_50": -0.01,
            "pct_from_20d_high": -0.05,
            "donchian_pos_20": 0.56,
            "bb_pct_b_20": 0.56,
            "rsi_14": 58.0,
            "rsi_21": 55.0,
            "smoothed_rsi_ema21_rsi10": 55.0,
            "stoch_k_14": 70.0,
            "stoch_d_3": 50.0,
            "williams_r_14": -30.0,
            "cci_20": 40.0,
            "adx_14": 19.0,
            "volume_ratio_20": 1.16,
            "volume_zscore_20": 0.10,
            "cmf_20": 0.01,
            "mfi_14": 49.0,
            "rs_spy_20": 0.01,
            "rs_spy_50": 0.0,
            "rs_qqq_20": 0.01,
            "rs_qqq_50": 0.0,
            "atr_pct_14": 0.045,
        }
    )
    return row


def _emerging_row(date: str, ticker: str, close: float = 100.0) -> dict:
    row = _strong_row(date, ticker, close)
    row.update(
        {
            "close_vs_sma_50": -0.01,
            "close_vs_ema_50": -0.005,
            "close_vs_ema_20": 0.015,
            "pct_from_20d_high": -0.015,
            "donchian_pos_20": 0.78,
            "bb_pct_b_20": 0.72,
            "rsi_14": 60.0,
            "rsi_21": 54.0,
            "smoothed_rsi_ema21_rsi10": 56.0,
            "pct_chg": 0.009,
            "close_pos": 0.78,
            "volume_ratio_20": 1.08,
            "volume_ratio_50": 1.00,
            "volume_zscore_20": 0.55,
            "cmf_20": 0.04,
            "rs_spy_20": 0.02,
            "rs_spy_50": 0.0,
            "rs_qqq_20": 0.03,
            "rs_qqq_50": 0.0,
            "lr_slope_log_63": 0.0006,
            "lr_slope_log_126": 0.0001,
            "lr_r2_63": 0.28,
            "lr_r2_126": 0.18,
            "ema_50_slope_20": 0.0001,
            "pct_above_ema_50_63": 0.45,
            "max_drawdown_63": -0.16,
            "tight_range_5": 0.045,
            "close_tightness_3": 0.012,
            "atr_pct_14": 0.032,
            "realized_vol_20": 0.17,
        }
    )
    return row


def _emerging_ignition_row(date: str, ticker: str, close: float = 100.0) -> dict:
    row = _emerging_row(date, ticker, close)
    row.update(
        {
            "close_vs_sma_50": 0.12,
            "close_vs_ema_20": 0.16,
            "pct_from_20d_high": 0.0,
            "donchian_pos_20": 0.94,
            "bb_pct_b_20": 0.98,
            "pct_chg": 0.065,
            "close_pos": 0.92,
            "volume_ratio_20": 2.10,
            "volume_ratio_50": 1.40,
            "volume_zscore_20": 2.40,
            "rs_spy_20": 0.10,
            "rs_qqq_20": 0.11,
            "lr_r2_63": 0.38,
            "base_width_20": 0.18,
            "base_tightness_20": 0.70,
            "close_to_prior_20bar_high": 0.055,
            "support_cluster_gap": 0.16,
        }
    )
    row["open"] = close - 4.5
    row["high"] = close + 1.0
    row["low"] = close - 5.0
    return row


def _continuation_row(date: str, ticker: str, close: float = 100.0) -> dict:
    row = _strong_row(date, ticker, close)
    row.update(
        {
            "close_vs_sma_10": 0.015,
            "close_vs_sma_20": 0.022,
            "close_vs_sma_50": 0.040,
            "close_vs_ema_10": 0.012,
            "close_vs_ema_20": 0.025,
            "close_vs_ema_50": 0.045,
            "pct_from_20d_high": -0.02,
            "pct_from_20d_low": 0.12,
            "donchian_pos_20": 0.80,
            "bb_pct_b_20": 0.72,
            "rsi_14": 62.0,
            "rsi_21": 57.0,
            "smoothed_rsi_ema21_rsi10": 58.0,
            "macd_hist": 0.15,
            "volume_ratio_20": 0.90,
            "volume_ratio_50": 0.92,
            "volume_zscore_20": -0.20,
            "cmf_20": 0.03,
            "atr_pct_14": 0.035,
            "realized_vol_20": 0.16,
            "rs_spy_20": 0.03,
            "rs_qqq_20": 0.04,
        }
    )
    row["open"] = close - 0.2
    row["high"] = close + 0.8
    row["low"] = close - 0.8
    return row


def _continuation_pullback_row(date: str, ticker: str, close: float = 100.0) -> dict:
    row = _continuation_row(date, ticker, close)
    row.update(
        {
            "close_vs_ema_20": 0.045,
            "rsi_14": 64.0,
            "donchian_pos_20": 0.82,
            "pct_from_20d_high": -0.04,
            "volume_ratio_20": 0.92,
            "atr_pct_14": 0.038,
            "close_tightness_3": 0.018,
            "tight_range_5": 0.075,
            "close_pos": 0.64,
            "pct_chg": 0.004,
        }
    )
    row["open"] = close - 0.4
    row["high"] = close + 0.9
    row["low"] = close - 1.2
    return row


def _power_breakout_row(date: str, ticker: str, close: float = 100.0) -> dict:
    row = _strong_row(date, ticker, close)
    row.update(
        {
            "close_vs_sma_10": 0.03,
            "close_vs_sma_20": 0.05,
            "close_vs_sma_50": 0.08,
            "close_vs_ema_10": 0.03,
            "close_vs_ema_20": 0.055,
            "close_vs_ema_50": 0.09,
            "pct_from_20d_high": 0.0,
            "pct_from_20d_low": 0.18,
            "donchian_pos_20": 0.95,
            "bb_pct_b_20": 0.98,
            "rsi_14": 74.0,
            "rsi_21": 64.0,
            "smoothed_rsi_ema21_rsi10": 66.0,
            "pct_chg": 0.03,
            "close_pos": 0.90,
            "volume_ratio_20": 1.65,
            "volume_ratio_50": 1.30,
            "volume_zscore_20": 2.00,
            "cmf_20": 0.10,
            "rs_spy_20": 0.07,
            "rs_qqq_20": 0.07,
        }
    )
    row["open"] = close - 1.5
    row["high"] = close + 0.8
    row["low"] = close - 1.8
    return row


def _expansion_leader_row(date: str, ticker: str, close: float = 100.0) -> dict:
    row = _strong_row(date, ticker, close)
    row.update(
        {
            "close_vs_sma_10": 0.08,
            "close_vs_sma_20": 0.12,
            "close_vs_sma_50": 0.18,
            "close_vs_ema_10": 0.07,
            "close_vs_ema_20": 0.16,
            "close_vs_ema_50": 0.20,
            "pct_from_20d_high": 0.0,
            "pct_from_20d_low": 0.26,
            "donchian_pos_20": 0.99,
            "bb_pct_b_20": 1.00,
            "rsi_14": 79.0,
            "rsi_21": 70.0,
            "smoothed_rsi_ema21_rsi10": 71.0,
            "pct_chg": 0.065,
            "close_pos": 0.93,
            "volume_ratio_20": 1.55,
            "volume_ratio_50": 1.25,
            "volume_zscore_20": 2.30,
            "cmf_20": 0.10,
            "atr_pct_14": 0.038,
            "tight_range_5": 0.12,
            "rs_spy_20": 0.22,
            "rs_qqq_20": 0.20,
        }
    )
    row["open"] = close - 4.0
    row["high"] = close + 1.5
    row["low"] = close - 4.5
    return row


def _exit_row(date: str, ticker: str, close: float = 95.0) -> dict:
    row = _strong_row(date, ticker, close)
    row.update(
        {
            "close_vs_sma_10": -0.02,
            "close_vs_sma_20": -0.03,
            "close_vs_sma_50": -0.05,
            "close_vs_ema_10": -0.02,
            "close_vs_ema_20": -0.03,
            "close_vs_ema_50": -0.05,
            "trend_stack_bullish": 0,
            "trend_stack_bearish": 1,
            "pct_from_20d_high": -0.20,
            "pct_from_20d_low": -0.05,
            "donchian_pos_20": 0.20,
            "bb_pct_b_20": 0.20,
            "rsi_14": 40.0,
            "rsi_21": 45.0,
            "smoothed_rsi_ema21_rsi10": 40.0,
            "macd_line": -1.00,
            "macd_signal": -0.50,
            "macd_hist": -0.50,
            "stoch_k_14": 30.0,
            "stoch_d_3": 35.0,
            "williams_r_14": -80.0,
            "cci_20": -50.0,
            "adx_14": 18.0,
            "plus_di_14": 12.0,
            "minus_di_14": 28.0,
            "pct_chg": -0.03,
            "close_pos": 0.20,
            "body": 0.50,
            "tr": 2.50,
            "atr_14": 2.50,
            "atr_pct_14": 0.025,
            "realized_vol_20": 0.15,
            "volume_ratio_20": 0.60,
            "volume_ratio_50": 0.80,
            "volume_zscore_20": -1.00,
            "cmf_20": -0.10,
            "mfi_14": 35.0,
            "rs_spy_20": -0.05,
            "rs_spy_50": -0.02,
            "rs_qqq_20": -0.05,
            "rs_qqq_50": -0.02,
            "lr_slope_log_63": -0.0025,
            "lr_slope_log_126": -0.0015,
            "lr_r2_63": 0.20,
            "lr_r2_126": 0.18,
            "ema_50_slope_20": -0.0012,
            "pct_above_ema_50_63": 0.15,
            "max_drawdown_63": -0.28,
        }
    )
    return row


def _cooldown_row(date: str, ticker: str, close: float = 99.0) -> dict:
    row = _strong_row(date, ticker, close)
    row.update(
        {
            "close_vs_ema_20": -0.01,
            "close_vs_sma_50": 0.01,
            "rs_spy_20": -0.02,
            "rs_qqq_20": -0.02,
            "pct_chg": -0.005,
        }
    )
    return row


def _raw_ohlcv_rows(ticker: str, start: str, periods: int, close_start: float, step: float) -> list[dict]:
    dates = pd.date_range(start, periods=periods, freq="B")
    rows = []
    for index, date in enumerate(dates):
        close = close_start + (index * step)
        rows.append(
            {
                "ticker": ticker,
                "Date": date,
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000 + (index * 10_000),
            }
        )
    return rows


def test_score_row_matches_bucket_spec():
    strategy = RallyPatternStrategy()

    scored = strategy.score_row(pd.Series(_strong_row("2024-01-02", "AAA")))

    assert scored["trend_points"] == 20
    assert scored["breakout_points"] == 18
    assert scored["momentum_points"] == 18
    assert scored["flow_points"] == 16
    assert scored["rs_points"] == 16
    assert scored["volatility_points"] == 10
    assert scored["penalty"] == 0
    assert scored["score"] == 98
    assert scored["label"] == "A"
    assert scored["pattern_stage"] == "breakout_or_power_trend"


def test_score_dataframe_uses_neutral_defaults_for_missing_fields():
    strategy = RallyPatternStrategy()
    df = pd.DataFrame([{"ticker": "AAA", "Date": "2024-01-02"}])

    scored = strategy.score_dataframe(df)

    assert scored.loc[0, "pct_from_20d_high"] == -1.0
    assert scored.loc[0, "williams_r_14"] == -100.0
    assert scored.loc[0, "flow_points"] == 4.0
    assert scored.loc[0, "score"] == 4.0
    assert scored.loc[0, "label"] == "D"
    assert scored.loc[0, "pattern_stage"] == "non_signal"


def test_build_feature_dataframe_computes_required_technicals_from_raw_prices():
    strategy = RallyPatternStrategy()
    raw_df = pd.DataFrame(
        _raw_ohlcv_rows("AAA", "2024-01-02", 60, 100.0, 1.0)
        + _raw_ohlcv_rows("SPY", "2024-01-02", 60, 400.0, 0.5)
        + _raw_ohlcv_rows("QQQ", "2024-01-02", 60, 300.0, 0.4)
    )

    features = strategy.build_feature_dataframe(raw_df)
    aaa_last = features[features["ticker"] == "AAA"].sort_values("Date").iloc[-1]
    aaa_only = raw_df[raw_df["ticker"] == "AAA"].sort_values("Date").reset_index(drop=True)
    last_close = aaa_only["close"].iloc[-1]
    sma_10 = aaa_only["close"].rolling(10, min_periods=1).mean().iloc[-1]
    ema_20 = aaa_only["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    avg_vol_20 = aaa_only["volume"].rolling(20, min_periods=1).mean().iloc[-1]
    roll_high_20 = aaa_only["high"].rolling(20, min_periods=1).max().iloc[-1]

    assert abs(aaa_last["close_vs_sma_10"] - ((last_close - sma_10) / sma_10)) < 1e-9
    assert abs(aaa_last["close_vs_ema_20"] - ((last_close - ema_20) / ema_20)) < 1e-9
    assert abs(aaa_last["volume_ratio_20"] - (aaa_only["volume"].iloc[-1] / avg_vol_20)) < 1e-9
    assert abs(aaa_last["pct_from_20d_high"] - ((last_close - roll_high_20) / roll_high_20)) < 1e-9
    assert "price_to_spy" in features.columns
    assert "price_to_qqq" in features.columns
    assert "rs_spy_20" in features.columns
    assert "rs_qqq_50" in features.columns
    assert "lr_slope_log_63" in features.columns
    assert "lr_r2_126" in features.columns
    assert "leader_regime_signal" in features.columns


def test_score_dataframe_auto_builds_features_from_raw_prices():
    strategy = RallyPatternStrategy()
    raw_df = pd.DataFrame(
        _raw_ohlcv_rows("AAA", "2024-01-02", 60, 100.0, 1.0)
        + _raw_ohlcv_rows("SPY", "2024-01-02", 60, 400.0, 0.5)
        + _raw_ohlcv_rows("QQQ", "2024-01-02", 60, 300.0, 0.4)
    )

    scored = strategy.score_dataframe(raw_df)
    aaa_scored = scored[scored["ticker"] == "AAA"].sort_values("Date").iloc[-1]

    assert "trend_points" in scored.columns
    assert "score" in scored.columns
    assert "close_vs_sma_20" in scored.columns
    assert "macd_hist" in scored.columns
    assert "rs_spy_20" in scored.columns
    assert "leader_trend_score" in scored.columns
    assert "leader_regime_signal" in scored.columns
    assert aaa_scored["score"] >= 0


def test_score_dataframe_flags_leader_regime_for_persistent_trend():
    strategy = RallyPatternStrategy()
    scored = strategy.score_dataframe(pd.DataFrame([_strong_row("2024-01-02", "AAA")]))

    assert float(scored.loc[0, "leader_trend_score"]) >= strategy.leader_regime_min_score
    assert bool(scored.loc[0, "leader_regime_signal"])


def test_breakout_requires_leader_regime():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
    ]
    rows[2].update(
        {
            "lr_slope_log_63": 0.0,
            "lr_slope_log_126": 0.0,
            "lr_r2_63": 0.10,
            "lr_r2_126": 0.10,
            "ema_50_slope_20": 0.0,
            "pct_above_ema_50_63": 0.30,
            "max_drawdown_63": -0.30,
            "rs_spy_20": 0.01,
            "rs_qqq_20": 0.01,
        }
    )
    rows[2]["volume_ratio_20"] = 1.30

    scored = strategy.score_dataframe(pd.DataFrame(rows))

    assert not bool(scored.iloc[-1]["leader_regime_signal"])
    assert not bool(scored.iloc[-1]["entry_signal"])


def test_score_dataframe_flags_emerging_leader_regime_without_confirmed_regime():
    strategy = RallyPatternStrategy()
    rows = [
        _emerging_row("2024-01-02", "AAA", 99.8),
        _emerging_row("2024-01-03", "AAA", 100.2),
        _emerging_row("2024-01-04", "AAA", 100.7),
        _emerging_row("2024-01-05", "AAA", 101.2),
    ]
    rows[0]["high"] = 100.4
    rows[1]["high"] = 100.7
    rows[2]["high"] = 101.0
    rows[3]["high"] = 101.4
    rows[0]["low"] = 98.9
    rows[1]["low"] = 99.5
    rows[2]["low"] = 100.0
    rows[3]["low"] = 100.4
    rows[0]["rs_qqq_20"] = 0.014
    rows[1]["rs_qqq_20"] = 0.020
    rows[2]["rs_qqq_20"] = 0.027
    rows[3]["rs_qqq_20"] = 0.034
    rows[0]["rs_spy_20"] = 0.009
    rows[1]["rs_spy_20"] = 0.013
    rows[2]["rs_spy_20"] = 0.018
    rows[3]["rs_spy_20"] = 0.024

    scored = strategy.score_dataframe(pd.DataFrame(rows))

    assert not bool(scored.iloc[-1]["confirmed_leader_regime_signal"])
    assert bool(scored.iloc[-1]["emerging_leader_regime_signal"])
    assert scored.iloc[-1]["leadership_stage"] == "emerging"


def test_generate_entries_triggers_emerging_leader_breakout():
    strategy = RallyPatternStrategy()
    rows = [
        _emerging_row("2024-01-02", "AAA", 99.8),
        _emerging_row("2024-01-03", "AAA", 100.2),
        _emerging_row("2024-01-04", "AAA", 100.7),
        _emerging_row("2024-01-04", "AAA", 103.2),
    ]
    rows[0]["Date"] = pd.Timestamp("2024-01-02")
    rows[1]["Date"] = pd.Timestamp("2024-01-03")
    rows[2]["Date"] = pd.Timestamp("2024-01-04")
    rows[3]["Date"] = pd.Timestamp("2024-01-05")
    rows[0]["high"] = 100.4
    rows[1]["high"] = 100.7
    rows[2]["high"] = 101.0
    rows[3]["high"] = 103.8
    rows[0]["low"] = 98.9
    rows[1]["low"] = 99.5
    rows[2]["low"] = 100.0
    rows[3]["low"] = 101.7
    rows[3]["volume_ratio_20"] = 1.30
    rows[3]["pct_chg"] = 0.018
    rows[3]["close_pos"] = 0.84
    rows[0]["rs_qqq_20"] = 0.014
    rows[1]["rs_qqq_20"] = 0.020
    rows[2]["rs_qqq_20"] = 0.027
    rows[3]["rs_qqq_20"] = 0.037
    rows[0]["rs_spy_20"] = 0.009
    rows[1]["rs_spy_20"] = 0.013
    rows[2]["rs_spy_20"] = 0.018
    rows[3]["rs_spy_20"] = 0.024

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "leadership_stage", "setup_type", "entry_signal"]]

    assert entry_view.iloc[-1]["leadership_stage"] == "emerging"
    assert entry_view.iloc[-1]["setup_type"] == "emerging_leader_breakout"
    assert bool(entry_view.iloc[-1]["entry_signal"])


def test_generate_entries_triggers_emerging_leader_shelf_after_base_tightening():
    strategy = RallyPatternStrategy(min_setup_days=1)
    rows = [
        _emerging_row("2024-01-02", "AAA", 99.8),
        _emerging_row("2024-01-03", "AAA", 100.3),
        _emerging_row("2024-01-04", "AAA", 100.7),
        _emerging_row("2024-01-05", "AAA", 100.9),
        _emerging_row("2024-01-08", "AAA", 101.0),
        _emerging_row("2024-01-09", "AAA", 101.8),
        _emerging_row("2024-01-10", "AAA", 102.2),
    ]
    rows[0]["high"] = 101.2
    rows[1]["high"] = 101.0
    rows[2]["high"] = 101.1
    rows[3]["high"] = 101.15
    rows[4]["high"] = 101.2
    rows[5]["high"] = 102.0
    rows[6]["high"] = 102.4
    rows[0]["low"] = 97.0
    rows[1]["low"] = 99.7
    rows[2]["low"] = 100.2
    rows[3]["low"] = 100.5
    rows[4]["low"] = 100.7
    rows[5]["low"] = 100.9
    rows[6]["low"] = 101.2
    rows[0]["rs_qqq_20"] = 0.012
    rows[1]["rs_qqq_20"] = 0.016
    rows[2]["rs_qqq_20"] = 0.022
    rows[3]["rs_qqq_20"] = 0.028
    rows[4]["rs_qqq_20"] = 0.032
    rows[5]["rs_qqq_20"] = 0.036
    rows[6]["rs_qqq_20"] = 0.040
    rows[0]["rs_spy_20"] = 0.008
    rows[1]["rs_spy_20"] = 0.010
    rows[2]["rs_spy_20"] = 0.014
    rows[3]["rs_spy_20"] = 0.018
    rows[4]["rs_spy_20"] = 0.020
    rows[5]["rs_spy_20"] = 0.024
    rows[6]["rs_spy_20"] = 0.028
    rows[4]["volume_ratio_20"] = 1.00
    rows[4]["pct_chg"] = 0.003
    rows[5]["volume_ratio_20"] = 1.02
    rows[5]["pct_chg"] = 0.006
    rows[6]["volume_ratio_20"] = 1.05
    rows[6]["pct_chg"] = 0.008
    rows[6]["close_pos"] = 0.82

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "leadership_stage", "setup_type", "entry_signal", "setup_state"]]

    assert entry_view.iloc[-1]["leadership_stage"] == "emerging"
    assert entry_view.iloc[-1]["setup_type"] == "emerging_leader_shelf"
    assert entry_view.iloc[-1]["setup_state"] == "entered"
    assert bool(entry_view.iloc[-1]["entry_signal"])


def test_generate_entries_triggers_emerging_leader_ignition_for_explosive_launch():
    strategy = RallyPatternStrategy()
    rows = [
        _emerging_row("2024-01-02", "AAA", 99.5),
        _emerging_row("2024-01-03", "AAA", 100.0),
        _emerging_row("2024-01-04", "AAA", 100.7),
        _emerging_ignition_row("2024-01-05", "AAA", 107.5),
    ]
    rows[0]["high"] = 100.2
    rows[1]["high"] = 100.7
    rows[2]["high"] = 101.1
    rows[3]["high"] = 108.4
    rows[0]["low"] = 98.8
    rows[1]["low"] = 99.3
    rows[2]["low"] = 99.8
    rows[3]["low"] = 102.0
    rows[0]["rs_qqq_20"] = 0.012
    rows[1]["rs_qqq_20"] = 0.020
    rows[2]["rs_qqq_20"] = 0.032
    rows[3]["rs_qqq_20"] = 0.110
    rows[0]["rs_spy_20"] = 0.008
    rows[1]["rs_spy_20"] = 0.012
    rows[2]["rs_spy_20"] = 0.020
    rows[3]["rs_spy_20"] = 0.095

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "leadership_stage", "setup_type", "entry_signal"]]

    assert entry_view.iloc[-1]["leadership_stage"] == "emerging"
    assert entry_view.iloc[-1]["setup_type"] == "emerging_leader_ignition"
    assert bool(entry_view.iloc[-1]["entry_signal"])


def test_zone_reentry_signal_allows_emerging_leader_followthrough():
    strategy = RallyPatternStrategy()
    scored = pd.DataFrame(
        [
            {
                "confirmed_leader_regime_signal": False,
                "emerging_leader_regime_signal": True,
                "score": strategy.add_on_min_score,
                "trend_stack_bullish": 1,
                "close_vs_ema_20": 0.04,
                "close_vs_sma_50": 0.03,
                "rs_spy_20": 0.02,
                "rs_qqq_20": 0.01,
                "zone_width_20": strategy.zone_max_width_20 - 0.01,
                "close_to_prior_20bar_high": -0.01,
            }
        ]
    )

    signal = strategy._zone_reentry_signal(scored)

    assert bool(signal.iloc[0])


def test_generate_entries_and_exits_follow_exact_rules():
    strategy = RallyPatternStrategy()
    df = pd.DataFrame(
        [
            _strong_row("2024-01-02", "AAA", 100.0),
            _strong_row("2024-01-03", "AAA", 101.0),
            _strong_row("2024-01-04", "AAA", 103.0),
            _exit_row("2024-01-02", "BBB"),
            _exit_row("2024-01-03", "BBB"),
        ]
    )
    df.loc[2, "volume_ratio_20"] = 1.30

    scored = strategy.score_dataframe(df)
    entries = strategy.generate_entries(scored)
    exits = strategy.generate_exits(scored)

    assert entries.tolist() == [False, False, True, False, False]
    assert exits.tolist() == [False, False, False, True, True]


def test_generate_entries_tracks_setup_state_and_uses_prior_3bar_pivot():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 102.0),
        _strong_row("2024-01-05", "AAA", 102.4),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 104.0
    rows[2]["high"] = 105.0
    rows[3]["high"] = 103.0
    rows[3]["close"] = 104.5
    rows[3]["volume_ratio_20"] = 1.30

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_state", "setup_signal", "entry_signal", "trigger_level"]]

    assert entry_view["setup_state"].tolist() == ["setup_ready", "setup_ready", "setup_ready", "setup_ready"]
    assert entry_view["entry_signal"].tolist() == [False, False, False, False]
    assert float(entry_view.iloc[-1]["trigger_level"]) == 105.0

    rows[3]["close"] = 105.5
    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_state", "entry_signal", "trigger_level"]]

    assert entry_view["entry_signal"].tolist() == [False, False, False, True]
    assert entry_view.iloc[-1]["setup_state"] == "entered"


def test_generate_entries_does_not_refresh_setup_without_signal_reset():
    strategy = RallyPatternStrategy(trigger_window_days=3)
    rows = [_strong_row(f"2024-01-0{day}", "AAA", 100.0) for day in range(2, 8)]

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    setup_view = scored[["Date", "setup_signal", "setup_state", "setup_age", "entry_signal"]]

    assert setup_view["setup_signal"].tolist() == [True, False, False, False, False, False]
    assert setup_view["setup_state"].tolist() == [
        "setup_ready",
        "setup_ready",
        "setup_ready",
        "setup_ready",
        "setup_expired",
        "no_setup",
    ]
    assert setup_view["entry_signal"].tolist() == [False, False, False, False, False, False]


def test_generate_entries_blocks_overextended_trigger_breakout():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 102.0),
        _strong_row("2024-01-05", "AAA", 108.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 103.0
    rows[3]["high"] = 109.0
    rows[3]["close"] = 104.5
    rows[3]["volume_ratio_20"] = 1.30
    rows[3]["close_vs_ema_20"] = 0.09
    rows[3]["rsi_14"] = 78.0

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_state", "entry_signal", "trigger_level"]]

    assert entry_view["entry_signal"].tolist() == [False, False, False, False]
    assert entry_view.iloc[-1]["setup_state"] == "setup_ready"


def test_generate_entries_blocks_extended_breakout_without_exceptional_close_or_volume():
    strategy = RallyPatternStrategy()
    rows = [
        _cooldown_row("2024-01-02", "AAA", 100.0),
        _cooldown_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 104.0
    rows[2]["close_vs_ema_20"] = 0.058
    rows[2]["volume_ratio_20"] = 1.20
    rows[2]["close_pos"] = 0.79
    rows[2]["pct_chg"] = 0.008

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "setup_state", "entry_signal"]]

    assert entry_view["entry_signal"].tolist() == [False, False, False]
    assert entry_view.iloc[-1]["setup_type"] != "breakout"
    assert entry_view.iloc[-1]["setup_state"] == "no_setup"


def test_generate_entries_allows_extended_breakout_with_exceptional_close_quality():
    strategy = RallyPatternStrategy(min_setup_days=1)
    rows = [
        _cooldown_row("2024-01-02", "AAA", 100.0),
        _cooldown_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
        _strong_row("2024-01-05", "AAA", 104.5),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 104.0
    rows[3]["high"] = 105.0
    rows[2]["close_vs_ema_20"] = 0.058
    rows[2]["volume_ratio_20"] = 1.20
    rows[2]["close_pos"] = 0.92
    rows[2]["pct_chg"] = 0.008
    rows[3]["volume_ratio_20"] = 1.20

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "entry_signal"]]

    assert entry_view["entry_signal"].tolist() == [False, False, False, True]
    assert entry_view.iloc[-1]["setup_type"] == "breakout"


def test_generate_entries_blocks_breakout_with_weak_combined_rs():
    strategy = RallyPatternStrategy()
    rows = [
        _cooldown_row("2024-01-02", "AAA", 100.0),
        _cooldown_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 104.0
    rows[2]["rs_spy_20"] = 0.03
    rows[2]["rs_qqq_20"] = 0.04

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "setup_state", "entry_signal"]]

    assert entry_view["entry_signal"].tolist() == [False, False, False]
    assert entry_view.iloc[-1]["setup_type"] != "breakout"
    assert entry_view.iloc[-1]["setup_state"] == "no_setup"


def test_generate_entries_triggers_continuation_setup_on_tight_flag():
    strategy = RallyPatternStrategy()
    rows = [
        _continuation_row("2024-01-02", "AAA", 100.0),
        _continuation_row("2024-01-03", "AAA", 100.6),
        _continuation_row("2024-01-04", "AAA", 101.0),
        _continuation_row("2024-01-05", "AAA", 101.6),
        _continuation_row("2024-01-08", "AAA", 102.2),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 101.1
    rows[2]["high"] = 101.2
    rows[3]["high"] = 101.8
    rows[4]["high"] = 102.4
    rows[0]["low"] = 99.4
    rows[1]["low"] = 100.0
    rows[2]["low"] = 100.4
    rows[3]["low"] = 101.0
    rows[4]["low"] = 101.5
    rows[0]["rs_qqq_20"] = 0.035
    rows[1]["rs_qqq_20"] = 0.038
    rows[2]["rs_qqq_20"] = 0.041
    rows[3]["rs_qqq_20"] = 0.043
    rows[4]["rs_qqq_20"] = 0.045
    rows[3]["close"] = 101.5
    rows[4]["close"] = 101.9
    rows[4]["volume_ratio_20"] = 0.94
    rows[4]["donchian_pos_20"] = 0.88

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_state", "setup_type", "entry_signal", "trigger_level"]]

    assert entry_view["setup_type"].tolist() == [
        "none",
        "none",
        "continuation_shelf",
        "continuation_shelf",
        "continuation_shelf",
    ]
    assert entry_view["entry_signal"].tolist() == [False, False, False, False, True]
    assert entry_view.iloc[-1]["setup_state"] == "entered"


def test_generate_entries_triggers_pullback_continuation_on_higher_volatility_reset():
    strategy = RallyPatternStrategy()
    rows = [
        _continuation_pullback_row("2024-01-02", "AAA", 100.0),
        _continuation_pullback_row("2024-01-03", "AAA", 100.8),
        _continuation_pullback_row("2024-01-04", "AAA", 101.4),
        _continuation_pullback_row("2024-01-05", "AAA", 101.9),
        _continuation_pullback_row("2024-01-08", "AAA", 102.5),
    ]
    rows[0]["high"] = 101.2
    rows[1]["high"] = 101.5
    rows[2]["high"] = 101.8
    rows[3]["high"] = 102.1
    rows[4]["high"] = 103.0
    rows[0]["low"] = 98.0
    rows[1]["low"] = 98.8
    rows[2]["low"] = 99.4
    rows[3]["low"] = 100.0
    rows[4]["low"] = 101.0
    rows[0]["rs_qqq_20"] = 0.032
    rows[1]["rs_qqq_20"] = 0.036
    rows[2]["rs_qqq_20"] = 0.041
    rows[3]["rs_qqq_20"] = 0.045
    rows[4]["rs_qqq_20"] = 0.047
    rows[4]["volume_ratio_20"] = 0.98

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_state", "setup_type", "entry_signal"]]

    assert entry_view.iloc[-1]["setup_type"] == "continuation_pullback"
    assert entry_view["entry_signal"].tolist() == [False, False, False, False, True]


def test_generate_entries_triggers_leader_reentry_after_recent_superleader():
    strategy = RallyPatternStrategy(enable_leader_reentry=True)
    rows = [
        _power_breakout_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.5),
        _strong_row("2024-01-04", "AAA", 104.2),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 103.0
    rows[2]["high"] = 104.8
    rows[0]["low"] = 98.0
    rows[1]["low"] = 100.0
    rows[2]["low"] = 102.5
    rows[2].update(
        {
            "close_vs_ema_20": 0.04,
            "close_vs_sma_50": 0.05,
            "pct_chg": 0.012,
            "volume_ratio_20": 1.08,
            "close_pos": 0.80,
            "donchian_pos_20": 0.88,
            "tight_range_5": 0.07,
            "close_tightness_3": 0.018,
            "support_cluster_gap": 0.04,
            "rs_spy_20": 0.08,
            "rs_qqq_20": 0.08,
            "pct_from_20d_high": -0.005,
        }
    )

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "entry_signal", "recent_super_leader"]]

    assert bool(entry_view.iloc[-1]["recent_super_leader"])
    assert entry_view.iloc[-1]["setup_type"] == "leader_reentry"
    assert bool(entry_view.iloc[-1]["entry_signal"])


def test_generate_entries_triggers_late_stage_leader_after_recent_superleader():
    strategy = RallyPatternStrategy(enable_late_stage_leaders=True)
    rows = [
        _power_breakout_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 102.0),
        _strong_row("2024-01-04", "AAA", 103.3),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.8
    rows[2]["high"] = 103.8
    rows[0]["low"] = 98.0
    rows[1]["low"] = 100.8
    rows[2]["low"] = 102.4
    rows[2].update(
        {
            "close_vs_ema_20": 0.04,
            "close_vs_sma_50": 0.08,
            "pct_chg": 0.003,
            "volume_ratio_20": 0.92,
            "close_pos": 0.78,
            "donchian_pos_20": 0.88,
            "tight_range_5": 0.06,
            "close_tightness_3": 0.015,
            "support_cluster_gap": 0.035,
            "rs_spy_20": 0.08,
            "rs_qqq_20": 0.09,
            "pct_from_20d_high": -0.008,
        }
    )

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "entry_signal", "recent_super_leader"]]

    assert bool(entry_view.iloc[-1]["recent_super_leader"])
    assert entry_view.iloc[-1]["setup_type"] == "late_stage_leader"
    assert bool(entry_view.iloc[-1]["entry_signal"])


def test_continuation_requires_positive_rs_acceleration():
    strategy = RallyPatternStrategy(
        continuation_min_rs_qqq_change_3=0.002,
        continuation_max_atr_pct_change_3=0.008,
    )
    rows = [
        _continuation_row("2024-01-02", "AAA", 100.0),
        _continuation_row("2024-01-03", "AAA", 100.6),
        _continuation_row("2024-01-04", "AAA", 101.0),
        _continuation_row("2024-01-05", "AAA", 101.6),
        _continuation_row("2024-01-08", "AAA", 102.2),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 101.1
    rows[2]["high"] = 101.2
    rows[3]["high"] = 101.8
    rows[4]["high"] = 102.4
    rows[0]["low"] = 99.4
    rows[1]["low"] = 100.0
    rows[2]["low"] = 100.4
    rows[3]["low"] = 101.0
    rows[4]["low"] = 101.5
    rows[0]["rs_qqq_20"] = 0.035
    rows[1]["rs_qqq_20"] = 0.038
    rows[2]["rs_qqq_20"] = 0.041
    rows[3]["rs_qqq_20"] = 0.043
    rows[4]["rs_qqq_20"] = 0.034
    rows[4]["close"] = 101.9
    rows[4]["volume_ratio_20"] = 0.94
    rows[4]["donchian_pos_20"] = 0.88

    scored = strategy.score_dataframe(pd.DataFrame(rows))

    assert scored["entry_signal"].tolist() == [False, False, False, False, False]


def test_continuation_setup_requires_zone_reclaim_context():
    strategy = RallyPatternStrategy()
    rows = [
        _continuation_row("2024-01-02", "AAA", 100.0),
        _continuation_row("2024-01-03", "AAA", 100.6),
        _continuation_row("2024-01-04", "AAA", 101.0),
        _continuation_row("2024-01-05", "AAA", 102.2),
    ]
    rows[0]["high"] = 110.0
    rows[1]["high"] = 109.0
    rows[2]["high"] = 108.0
    rows[3]["high"] = 102.4
    rows[0]["low"] = 94.0
    rows[1]["low"] = 95.0
    rows[2]["low"] = 96.0
    rows[3]["low"] = 101.5
    rows[3]["close"] = 101.9
    rows[3]["volume_ratio_20"] = 0.94
    rows[3]["donchian_pos_20"] = 0.88

    scored = strategy.score_dataframe(pd.DataFrame(rows))

    assert scored["entry_signal"].tolist() == [False, False, False, False]
    assert scored.iloc[-1]["zone_reentry_signal"] == 0


def test_continuation_shelf_rejects_seller_zone_without_room_to_run():
    strategy = RallyPatternStrategy()
    scored = strategy._prepare_dataframe(pd.DataFrame([_continuation_row("2024-01-04", "AAA", 103.0)]))
    scored["score"] = 82.0
    scored["prior_20bar_high"] = 103.0
    scored["prior_20bar_low"] = 98.0
    scored["zone_width_20"] = 0.06
    scored["close_to_prior_20bar_high"] = -0.01
    scored["prior_60bar_high"] = 105.0
    scored["room_to_60bar_high"] = 0.009
    scored["in_60bar_seller_zone"] = True
    scored["prior_5bar_low"] = 100.0

    assert not bool(strategy._continuation_shelf_setup_signal(scored).iloc[0])


def test_generate_entries_triggers_power_breakout_same_day():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 106.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 107.0

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_signal", "setup_type", "setup_state", "entry_signal", "trigger_level"]]

    assert entry_view["entry_signal"].tolist() == [False, False, True]
    assert entry_view["setup_signal"].tolist() == [True, False, True]
    assert entry_view.iloc[-1]["setup_type"] == "power_breakout"
    assert entry_view.iloc[-1]["setup_state"] == "entered"
    assert float(entry_view.iloc[-1]["trigger_level"]) == 102.0


def test_generate_entries_blocks_power_breakout_below_broader_20d_high():
    strategy = RallyPatternStrategy()
    rows = [
        _cooldown_row("2024-01-02", "AAA", 100.0),
        _cooldown_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 106.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 107.0
    rows[2]["pct_from_20d_high"] = -0.02

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "entry_signal"]]

    assert entry_view["entry_signal"].tolist() == [False, False, False]
    assert entry_view.iloc[-1]["setup_type"] != "power_breakout"


def test_generate_entries_blocks_power_breakout_without_calm_or_explosive_profile():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 106.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 107.0
    rows[2]["close_pos"] = 0.82
    rows[2]["volume_ratio_20"] = 1.35
    rows[2]["pct_chg"] = 0.016
    rows[2]["atr_pct_14"] = 0.06
    rows[2]["tight_range_5"] = 0.15

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "entry_signal"]]

    assert entry_view.iloc[-1]["setup_type"] != "power_breakout"


def test_generate_entries_blocks_power_breakout_with_weak_breakout_clearance_and_body():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 102.1),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["open"] = 101.95
    rows[2]["high"] = 102.4
    rows[2]["low"] = 101.6
    rows[2]["close"] = 102.1
    rows[2]["body"] = 0.15
    rows[2]["tr"] = 0.8
    rows[2]["close_pos"] = 0.63
    rows[2]["pct_chg"] = 0.012
    rows[2]["volume_ratio_20"] = 1.90

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "entry_signal"]]

    assert entry_view.iloc[-1]["setup_type"] != "power_breakout"


def test_generate_entries_blocks_power_breakout_with_large_upper_wick():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 104.1),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["open"] = 103.3
    rows[2]["high"] = 105.1
    rows[2]["low"] = 100.0
    rows[2]["close"] = 104.1
    rows[2]["body"] = 0.8
    rows[2]["tr"] = 5.1
    rows[2]["close_pos"] = 0.81
    rows[2]["pct_chg"] = 0.03
    rows[2]["volume_ratio_20"] = 1.9

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "entry_signal"]]

    assert entry_view.iloc[-1]["setup_type"] != "power_breakout"


def test_generate_entries_triggers_expansion_leader_same_day():
    strategy = RallyPatternStrategy()
    rows = [
        _cooldown_row("2024-01-02", "AAA", 100.0),
        _cooldown_row("2024-01-03", "AAA", 101.0),
        _expansion_leader_row("2024-01-04", "AAA", 112.0),
    ]
    rows[0]["high"] = 107.0
    rows[1]["high"] = 108.0
    rows[2]["high"] = 113.0
    rows[2]["close"] = 109.0

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_signal", "setup_type", "setup_state", "entry_signal", "trigger_level"]]

    assert entry_view["entry_signal"].tolist() == [False, False, True]
    assert entry_view.iloc[-1]["setup_type"] == "expansion_leader"
    assert entry_view.iloc[-1]["setup_state"] == "entered"
    assert float(entry_view.iloc[-1]["trigger_level"]) == 108.0


def test_generate_entries_blocks_expansion_leader_far_above_breakout_level_without_big_volume():
    strategy = RallyPatternStrategy()
    rows = [
        _cooldown_row("2024-01-02", "AAA", 100.0),
        _cooldown_row("2024-01-03", "AAA", 101.0),
        _expansion_leader_row("2024-01-04", "AAA", 112.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 113.0
    rows[2]["close"] = 109.0
    rows[2]["volume_ratio_20"] = 1.55

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    entry_view = scored[["Date", "setup_type", "setup_state", "entry_signal"]]

    assert entry_view["entry_signal"].tolist() == [False, False, False]
    assert entry_view.iloc[-1]["setup_type"] != "expansion_leader"
    assert entry_view.iloc[-1]["setup_state"] == "no_setup"


def test_generate_exits_requires_persistence_for_soft_and_relative_weakness():
    strategy = RallyPatternStrategy()
    df = pd.DataFrame(
        [
            _strong_row("2024-01-02", "AAA"),
            _cooldown_row("2024-01-03", "AAA"),
            _cooldown_row("2024-01-04", "AAA"),
        ]
    )

    scored = strategy.score_dataframe(df)
    scored.loc[1:, "score"] = 30.0
    exits = strategy.generate_exits(scored)

    assert exits.tolist() == [False, False, True]


def test_rank_candidates_uses_score_then_rs_then_volume():
    strategy = RallyPatternStrategy()
    leader_setup = _strong_row("2024-01-02", "AAA", 100.0)
    leader_ready = _strong_row("2024-01-03", "AAA", 101.0)
    tie_setup = _strong_row("2024-01-02", "BBB", 100.0)
    tie_ready = _strong_row("2024-01-03", "BBB", 101.0)
    third_setup = _strong_row("2024-01-02", "CCC", 100.0)
    third_ready = _strong_row("2024-01-03", "CCC", 101.0)
    leader = _strong_row("2024-01-04", "AAA", 103.0)
    tie_breaker = _strong_row("2024-01-04", "BBB", 103.0)
    third = _strong_row("2024-01-04", "CCC", 103.0)

    leader["rs_spy_20"] = 0.08
    tie_breaker["rs_spy_20"] = 0.05
    third["rs_spy_20"] = 0.05
    third["volume_ratio_20"] = 1.30

    ranked = strategy.rank_candidates(
        pd.DataFrame(
            [
                leader_setup,
                leader_ready,
                tie_setup,
                tie_ready,
                third_setup,
                third_ready,
                tie_breaker,
                third,
                leader,
            ]
        )
    )

    assert ranked["ticker"].tolist() == ["AAA", "CCC", "BBB"]
    assert ranked["candidate_rank"].tolist() == [1, 2, 3]


def test_rank_candidates_prefers_power_breakout_over_other_setup_types():
    strategy = RallyPatternStrategy(min_setup_days=1)
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 106.0),
        _strong_row("2024-01-02", "BBB", 100.0),
        _strong_row("2024-01-03", "BBB", 101.0),
        _strong_row("2024-01-04", "BBB", 103.0),
        _continuation_row("2024-01-01", "CCC", 99.6),
        _continuation_row("2024-01-02", "CCC", 100.0),
        _continuation_row("2024-01-03", "CCC", 100.5),
        _continuation_row("2024-01-04", "CCC", 101.8),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 107.0
    rows[3]["high"] = 101.0
    rows[4]["high"] = 102.0
    rows[5]["high"] = 104.0
    rows[5]["volume_ratio_20"] = 1.35
    rows[6]["high"] = 100.3
    rows[7]["high"] = 101.0
    rows[8]["high"] = 101.2
    rows[9]["high"] = 102.0
    rows[6]["low"] = 98.9
    rows[7]["low"] = 99.4
    rows[8]["low"] = 100.0
    rows[9]["low"] = 101.0
    rows[6]["rs_qqq_20"] = 0.035
    rows[7]["rs_qqq_20"] = 0.038
    rows[8]["rs_qqq_20"] = 0.041
    rows[9]["rs_qqq_20"] = 0.045
    rows[9]["volume_ratio_20"] = 0.95

    ranked = strategy.rank_candidates(pd.DataFrame(rows))

    ranked = ranked[ranked["Date"] == pd.Timestamp("2024-01-04")].reset_index(drop=True)

    assert ranked["ticker"].tolist() == ["AAA", "BBB", "CCC"]
    assert ranked["setup_type"].tolist() == ["power_breakout", "breakout", "continuation_shelf"]


def test_backtest_enforces_cooldown_but_allows_score_75_override():
    strategy = RallyPatternStrategy()
    dates = pd.date_range("2024-01-02", periods=10, freq="B")

    rows = [
        _strong_row(str(dates[0].date()), "AAA", 100.0),
        _strong_row(str(dates[1].date()), "AAA", 101.0),
        _strong_row(str(dates[2].date()), "AAA", 103.0),
        _exit_row(str(dates[3].date()), "AAA", 95.0),
        _moderate_entry_row(str(dates[4].date()), "AAA", 96.0),
        _moderate_entry_row(str(dates[5].date()), "AAA", 97.0),
        _strong_row(str(dates[6].date()), "AAA", 104.0),
        _strong_row(str(dates[7].date()), "AAA", 105.0),
        _strong_row(str(dates[8].date()), "AAA", 107.0),
        _exit_row(str(dates[9].date()), "AAA", 102.0),
    ]
    rows[2]["volume_ratio_20"] = 1.30
    rows[8]["volume_ratio_20"] = 1.30

    results = strategy.backtest(pd.DataFrame(rows), max_positions=2, initial_capital=100_000.0)
    trades = results["trades"]
    equity_curve = results["equity_curve"]
    daily_holdings = results["daily_holdings"]

    assert len(trades) == 2
    assert trades["entry_date"].dt.normalize().tolist() == [dates[2], dates[8]]
    assert trades["exit_date"].dt.normalize().tolist() == [dates[3], dates[9]]
    assert not daily_holdings.empty
    assert not equity_curve.empty
    assert equity_curve["num_positions"].max() == 1


def test_backtest_blocks_same_trigger_aggressive_retry_during_cooldown():
    strategy = RallyPatternStrategy()
    dates = pd.date_range("2024-01-02", periods=7, freq="B")

    rows = [
        _strong_row(str(dates[0].date()), "AAA", 100.0),
        _strong_row(str(dates[1].date()), "AAA", 101.0),
        _power_breakout_row(str(dates[2].date()), "AAA", 104.0),
        _exit_row(str(dates[3].date()), "AAA", 96.0),
        _strong_row(str(dates[4].date()), "AAA", 100.0),
        _strong_row(str(dates[5].date()), "AAA", 101.0),
        _power_breakout_row(str(dates[6].date()), "AAA", 104.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 104.8
    rows[3]["high"] = 97.0
    rows[4]["high"] = 101.0
    rows[5]["high"] = 102.0
    rows[6]["high"] = 104.8

    results = strategy.backtest(pd.DataFrame(rows), max_positions=1, initial_capital=100_000.0)
    trades = results["trades"]

    assert len(trades) == 1
    assert trades.iloc[0]["entry_date"].normalize() == dates[2]
    assert trades.iloc[0]["exit_date"].normalize() == dates[3]


def test_backtest_allows_aggressive_retry_when_breakout_is_fresh_above_failed_trigger():
    strategy = RallyPatternStrategy()
    dates = pd.date_range("2024-01-02", periods=8, freq="B")

    rows = [
        _strong_row(str(dates[0].date()), "AAA", 100.0),
        _strong_row(str(dates[1].date()), "AAA", 101.0),
        _power_breakout_row(str(dates[2].date()), "AAA", 104.0),
        _exit_row(str(dates[3].date()), "AAA", 96.0),
        _strong_row(str(dates[4].date()), "AAA", 100.0),
        _strong_row(str(dates[5].date()), "AAA", 104.5),
        _power_breakout_row(str(dates[6].date()), "AAA", 107.0),
        _exit_row(str(dates[7].date()), "AAA", 101.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 104.8
    rows[3]["high"] = 97.0
    rows[4]["high"] = 101.0
    rows[5]["high"] = 105.0
    rows[6]["high"] = 107.8
    rows[7]["high"] = 102.0

    results = strategy.backtest(pd.DataFrame(rows), max_positions=1, initial_capital=100_000.0)
    trades = results["trades"]

    assert len(trades) == 2
    assert trades["entry_date"].dt.normalize().tolist() == [dates[2], dates[6]]
    assert trades["exit_date"].dt.normalize().tolist() == [dates[3], dates[7]]


def test_backtest_trade_start_date_blocks_warmup_entries():
    strategy = RallyPatternStrategy()
    dates = pd.date_range("2022-01-03", periods=8, freq="B")
    rows = [_strong_row(str(date.date()), "AAA", 100.0 + i) for i, date in enumerate(dates)]
    for index in range(1, len(rows)):
        rows[index]["volume_ratio_20"] = 1.30

    results = strategy.backtest(
        pd.DataFrame(rows),
        max_positions=1,
        initial_capital=100_000.0,
        start_date="2022-01-03",
        trade_start_date="2022-02-01",
    )

    assert results["trades"].empty
    assert results["daily_holdings"].empty
    assert not results["equity_curve"].empty
    assert results["equity_curve"]["num_positions"].eq(0).all()


def test_backtest_baseline_splits_cash_evenly_when_max_positions_is_zero():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
        _strong_row("2024-01-02", "BBB", 100.0),
        _strong_row("2024-01-03", "BBB", 101.0),
        _strong_row("2024-01-04", "BBB", 103.0),
    ]
    rows[2]["volume_ratio_20"] = 1.30
    rows[5]["volume_ratio_20"] = 1.30

    results = strategy.backtest(pd.DataFrame(rows), max_positions=0, initial_capital=100_000.0)

    held_on_entry_day = results["daily_holdings"][
        results["daily_holdings"]["Date"] == pd.Timestamp("2024-01-04")
    ]
    assert sorted(held_on_entry_day["ticker"].tolist()) == ["AAA", "BBB"]
    assert results["equity_curve"]["num_positions"].max() == 2
    assert held_on_entry_day["market_value"].tolist() == [50_000.0, 50_000.0]


def test_backtest_baseline_respects_absolute_stock_allocation_cap():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
    ]
    rows[2]["volume_ratio_20"] = 1.30

    results = strategy.backtest(pd.DataFrame(rows), max_positions=0, initial_capital=500_000.0)
    held_on_entry_day = results["daily_holdings"][
        results["daily_holdings"]["Date"] == pd.Timestamp("2024-01-04")
    ]

    assert len(held_on_entry_day) == 1
    assert abs(held_on_entry_day.iloc[0]["market_value"] - 50_000.0) < 1e-6


def test_backtest_hybrid_risk_capped_uses_risk_budget():
    strategy = RallyPatternStrategy(allocation_mode="hybrid_risk_capped")
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
        _strong_row("2024-01-02", "BBB", 100.0),
        _strong_row("2024-01-03", "BBB", 101.0),
        _strong_row("2024-01-04", "BBB", 103.0),
    ]
    rows[2]["volume_ratio_20"] = 1.30
    rows[5]["volume_ratio_20"] = 1.30

    results = strategy.backtest(pd.DataFrame(rows), max_positions=0, initial_capital=100_000.0)

    held_on_entry_day = results["daily_holdings"][
        results["daily_holdings"]["Date"] == pd.Timestamp("2024-01-04")
    ]
    scored_on_entry_day = results["scored_data"][
        results["scored_data"]["Date"] == pd.Timestamp("2024-01-04")
    ][["ticker", "close", "entry_risk_per_share", "setup_type"]]
    expected_values = {
        row["ticker"]: min(
            float(row["close"]) * (2_000.0 / float(row["entry_risk_per_share"])),
            100_000.0 * strategy._setup_target_weight(str(row["setup_type"])),
        )
        for _, row in scored_on_entry_day.iterrows()
    }
    assert sorted(held_on_entry_day["ticker"].tolist()) == ["AAA", "BBB"]
    assert results["equity_curve"]["num_positions"].max() == 2
    for _, row in held_on_entry_day.iterrows():
        assert abs(row["market_value"] - expected_values[row["ticker"]]) < 1e-6


def test_backtest_add_on_respects_absolute_stock_allocation_cap():
    strategy = RallyPatternStrategy(max_add_ons_per_ticker=1)
    rows = [
        _power_breakout_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 111.0),
        _power_breakout_row("2024-01-04", "AAA", 113.0),
        _exit_row("2024-01-05", "AAA", 110.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 112.0
    rows[2]["high"] = 114.0
    rows[3]["high"] = 111.0
    rows[0]["low"] = 98.0
    rows[1]["low"] = 109.0
    rows[2]["low"] = 112.0
    rows[3]["low"] = 109.5
    rows[2]["volume_ratio_20"] = 1.90
    rows[2]["pct_chg"] = 0.025
    rows[2]["close_vs_ema_20"] = 0.055
    rows[2]["close_vs_sma_50"] = 0.05

    results = strategy.backtest(pd.DataFrame(rows), max_positions=1, initial_capital=500_000.0)
    holdings = results["daily_holdings"]
    day_two_shares = holdings.loc[holdings["Date"] == pd.Timestamp("2024-01-03"), "shares"].iloc[0]
    day_three_shares = holdings.loc[holdings["Date"] == pd.Timestamp("2024-01-04"), "shares"].iloc[0]

    assert day_three_shares == day_two_shares


def test_backtest_equal_weight_cap_limits_single_entry_allocation():
    strategy = RallyPatternStrategy(allocation_mode="equal_weight_cap")
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
    ]
    rows[2]["volume_ratio_20"] = 1.30

    results = strategy.backtest(pd.DataFrame(rows), max_positions=0, initial_capital=100_000.0)
    held_on_entry_day = results["daily_holdings"][
        results["daily_holdings"]["Date"] == pd.Timestamp("2024-01-04")
    ]

    assert len(held_on_entry_day) == 1
    assert abs(held_on_entry_day.iloc[0]["market_value"] - 35_000.0) < 1e-6


def test_aggressive_starter_sizing_halves_initial_power_breakout_allocation():
    strategy = RallyPatternStrategy(enable_aggressive_starter_sizing=True)
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 106.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 107.0

    results = strategy.backtest(pd.DataFrame(rows), max_positions=0, initial_capital=100_000.0)
    held_on_entry_day = results["daily_holdings"][
        results["daily_holdings"]["Date"] == pd.Timestamp("2024-01-04")
    ]

    assert len(held_on_entry_day) == 1
    assert abs(held_on_entry_day.iloc[0]["market_value"] - 25_000.0) < 1e-6


def test_emerging_leader_ignition_uses_starter_sizing_by_default():
    strategy = RallyPatternStrategy()
    rows = [
        _emerging_row("2024-01-02", "AAA", 99.5),
        _emerging_row("2024-01-03", "AAA", 100.0),
        _emerging_row("2024-01-04", "AAA", 100.7),
        _emerging_ignition_row("2024-01-05", "AAA", 107.5),
    ]
    rows[0]["high"] = 100.2
    rows[1]["high"] = 100.7
    rows[2]["high"] = 101.1
    rows[3]["high"] = 108.4
    rows[0]["low"] = 98.8
    rows[1]["low"] = 99.3
    rows[2]["low"] = 99.8
    rows[3]["low"] = 102.0

    results = strategy.backtest(pd.DataFrame(rows), max_positions=0, initial_capital=100_000.0)
    held_on_entry_day = results["daily_holdings"][
        results["daily_holdings"]["Date"] == pd.Timestamp("2024-01-05")
    ]

    assert len(held_on_entry_day) == 1
    assert abs(held_on_entry_day.iloc[0]["market_value"] - 25_000.0) < 1e-6


def test_backtest_setup_tiered_cap_allocates_more_to_power_breakout_than_continuation():
    strategy = RallyPatternStrategy(allocation_mode="setup_tiered_cap", min_setup_days=1)
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 106.0),
        _continuation_row("2024-01-01", "BBB", 99.6),
        _continuation_row("2024-01-02", "BBB", 100.0),
        _continuation_row("2024-01-03", "BBB", 100.5),
        _continuation_row("2024-01-04", "BBB", 101.8),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 102.0
    rows[2]["high"] = 107.0
    rows[3]["high"] = 100.3
    rows[4]["high"] = 101.0
    rows[5]["high"] = 101.2
    rows[6]["high"] = 102.0
    rows[3]["low"] = 98.9
    rows[4]["low"] = 99.4
    rows[5]["low"] = 100.0
    rows[6]["low"] = 101.0
    rows[6]["rs_qqq_20"] = 0.045
    rows[6]["volume_ratio_20"] = 0.95

    results = strategy.backtest(pd.DataFrame(rows), max_positions=0, initial_capital=100_000.0)
    held_on_entry_day = results["daily_holdings"][
        results["daily_holdings"]["Date"] == pd.Timestamp("2024-01-04")
    ].sort_values("market_value", ascending=False)

    assert held_on_entry_day["ticker"].tolist() == ["AAA", "BBB"]
    assert held_on_entry_day.iloc[0]["market_value"] > held_on_entry_day.iloc[1]["market_value"]


def test_rank_candidates_penalizes_artificially_tight_stops():
    strategy = RallyPatternStrategy(allocation_mode="hybrid_risk_capped")
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
        _strong_row("2024-01-02", "BBB", 100.0),
        _strong_row("2024-01-03", "BBB", 101.0),
        _strong_row("2024-01-04", "BBB", 103.0),
    ]
    rows[2]["volume_ratio_20"] = 1.30
    rows[5]["volume_ratio_20"] = 1.30
    rows[2]["close_vs_ema_20"] = 0.01
    rows[2]["close_vs_sma_50"] = 0.01
    rows[5]["close_vs_ema_20"] = 0.05
    rows[5]["close_vs_sma_50"] = 0.08
    rows[0]["low"] = 99.0
    rows[1]["low"] = 100.0
    rows[2]["low"] = 101.0
    rows[3]["low"] = 95.0
    rows[4]["low"] = 96.0
    rows[5]["low"] = 97.0
    rows[2]["atr_14"] = 1.0
    rows[5]["atr_14"] = 1.0
    rows[2]["rs_spy_20"] = rows[5]["rs_spy_20"] = 0.05
    rows[2]["rs_qqq_20"] = rows[5]["rs_qqq_20"] = 0.05

    scored = strategy.score_dataframe(pd.DataFrame(rows))
    trigger_date = pd.Timestamp("2024-01-04")
    scored.loc[scored["Date"] == trigger_date, "score"] = 95.0
    ranked = strategy.rank_candidates(scored)
    ranked = ranked[ranked["Date"] == trigger_date].reset_index(drop=True)

    assert ranked["ticker"].tolist() == ["BBB", "AAA"]
    assert not bool(ranked.loc[0, "entry_tight_stop_penalty"])
    assert bool(ranked.loc[1, "entry_tight_stop_penalty"])


def test_backtest_uses_trailing_stop_reason():
    strategy = RallyPatternStrategy(trailing_stop_min_days_to_arm=1)
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _power_breakout_row("2024-01-04", "AAA", 108.0),
        _strong_row("2024-01-05", "AAA", 95.0),
    ]
    rows[2]["volume_ratio_20"] = 2.10
    rows[0]["low"] = 90.0
    rows[1]["low"] = 96.0
    rows[2]["low"] = 100.0
    rows[3]["low"] = 94.0
    rows[3].update(
        {
            "close_vs_sma_50": 0.02,
            "close_vs_ema_20": 0.01,
            "rs_spy_20": 0.03,
            "rs_qqq_20": 0.03,
            "atr_14": 3.0,
        }
    )

    results = strategy.backtest(pd.DataFrame(rows), max_positions=2, initial_capital=100_000.0)

    assert len(results["trades"]) == 1
    assert results["trades"].loc[0, "exit_reason"] == "atr_trailing_stop"


def test_backtest_adds_to_winning_position_on_zone_reentry():
    strategy = RallyPatternStrategy(max_allocation_per_stock=100_000.0)
    rows = [
        _power_breakout_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 111.0),
        _power_breakout_row("2024-01-04", "AAA", 113.0),
        _exit_row("2024-01-05", "AAA", 110.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 112.0
    rows[2]["high"] = 114.0
    rows[3]["high"] = 111.0
    rows[0]["low"] = 98.0
    rows[1]["low"] = 109.0
    rows[2]["low"] = 112.0
    rows[3]["low"] = 109.5
    rows[0]["pct_from_20d_high"] = 0.0
    rows[1]["pct_from_20d_high"] = -0.005
    rows[2]["pct_from_20d_high"] = 0.0
    rows[2]["volume_ratio_20"] = 1.90
    rows[2]["pct_chg"] = 0.025
    rows[2]["close_vs_ema_20"] = 0.055
    rows[2]["close_vs_sma_50"] = 0.05

    results = strategy.backtest(pd.DataFrame(rows), max_positions=2, initial_capital=100_000.0)
    holdings = results["daily_holdings"]
    day_two_shares = holdings.loc[holdings["Date"] == pd.Timestamp("2024-01-03"), "shares"].iloc[0]
    day_three_shares = holdings.loc[holdings["Date"] == pd.Timestamp("2024-01-04"), "shares"].iloc[0]

    assert day_three_shares > day_two_shares
    assert len(results["trades"]) == 1


def test_aggressive_starter_sizing_can_top_up_power_breakout_after_confirmation():
    strategy = RallyPatternStrategy(
        enable_aggressive_starter_sizing=True,
        max_allocation_per_stock=50_000.0,
        max_add_ons_per_ticker=1,
    )
    rows = [
        _power_breakout_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 111.0),
        _power_breakout_row("2024-01-04", "AAA", 113.0),
        _exit_row("2024-01-05", "AAA", 110.0),
    ]
    rows[0]["high"] = 101.0
    rows[1]["high"] = 112.0
    rows[2]["high"] = 114.0
    rows[3]["high"] = 111.0
    rows[0]["low"] = 98.0
    rows[1]["low"] = 109.0
    rows[2]["low"] = 112.0
    rows[3]["low"] = 109.5
    rows[0]["pct_from_20d_high"] = 0.0
    rows[1]["pct_from_20d_high"] = -0.005
    rows[2]["pct_from_20d_high"] = 0.0
    rows[2]["volume_ratio_20"] = 1.90
    rows[2]["pct_chg"] = 0.025
    rows[2]["close_vs_ema_20"] = 0.055
    rows[2]["close_vs_sma_50"] = 0.05

    results = strategy.backtest(pd.DataFrame(rows), max_positions=2, initial_capital=100_000.0)
    holdings = results["daily_holdings"]
    entry_day_value = holdings.loc[holdings["Date"] == pd.Timestamp("2024-01-02"), "market_value"].iloc[0]
    day_two_value = holdings.loc[holdings["Date"] == pd.Timestamp("2024-01-03"), "market_value"].iloc[0]

    assert abs(entry_day_value - 25_000.0) < 1e-6
    assert abs(day_two_value - 50_000.0) < 1e-6


def test_trend_hold_skips_relative_weakness_for_strong_open_winner():
    strategy = RallyPatternStrategy()
    row = pd.Series(_strong_row("2024-01-24", "AAA", 112.0))
    row["relative_weak_2d"] = True
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["soft_score_fail_2d"] = False
    row["atr_14"] = 2.0
    row["roll_low_10"] = 105.0
    row["score"] = 82.0
    row["close_vs_ema_20"] = 0.05
    row["close_vs_sma_50"] = 0.08
    row["rs_spy_20"] = 0.04
    row["rs_qqq_20"] = 0.03

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-02"),
        entry_price=100.0,
        shares=10.0,
        entry_score=95.0,
        setup_type="power_breakout",
        best_score=98.0,
        highest_close=119.0,
        has_new_high=True,
        score_improved=True,
        days_held=18,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) is None


def test_continuation_exit_is_less_forgiving_on_single_ema20_break():
    strategy = RallyPatternStrategy()
    row = pd.Series(_continuation_row("2024-01-24", "AAA", 104.0))
    row["close_below_ema20"] = True
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 2.0
    row["score"] = 76.0

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-02"),
        entry_price=100.0,
        shares=10.0,
        entry_score=82.0,
        setup_type="continuation_shelf",
        best_score=84.0,
        highest_close=108.0,
        has_new_high=True,
        score_improved=True,
        days_held=6,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "structure_ema20_fail"


def test_zone_support_fail_exits_before_other_structure_checks():
    strategy = RallyPatternStrategy()
    row = pd.Series(_strong_row("2024-01-24", "AAA", 101.0))
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 95.0
    row["atr_14"] = 2.0
    row["score"] = 80.0

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=103.0,
        shares=10.0,
        entry_score=92.0,
        setup_type="breakout",
        best_score=92.0,
        highest_close=103.0,
        has_new_high=False,
        score_improved=False,
        days_held=2,
        add_on_count=0,
        zone_support=102.0,
    )

    assert strategy._exit_reason(row, position) == "zone_support_fail"


def test_zone_support_fail_allows_brief_reclaim_grace_for_strong_aggressive_setup():
    strategy = RallyPatternStrategy()
    row = pd.Series(_power_breakout_row("2024-01-24", "AAA", 103.8))
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 99.0
    row["atr_14"] = 2.0
    row["score"] = 91.0
    row["close_pos"] = 0.72
    row["rs_spy_20"] = 0.05
    row["rs_qqq_20"] = 0.05
    row["prior_20bar_low"] = 101.0
    row["close_vs_ema_20"] = 0.015
    row["close_vs_sma_50"] = 0.02

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=105.0,
        shares=10.0,
        entry_score=95.0,
        setup_type="power_breakout",
        best_score=95.0,
        highest_close=105.0,
        has_new_high=False,
        score_improved=False,
        days_held=1,
        add_on_count=0,
        zone_support=104.0,
    )

    assert strategy._exit_reason(row, position) is None


def test_breakout_exit_cuts_failed_followthrough_early():
    strategy = RallyPatternStrategy()
    row = pd.Series(_strong_row("2024-01-24", "AAA", 101.0))
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 95.0
    row["atr_14"] = 2.0
    row["score"] = 68.0
    row["close_vs_ema_20"] = 0.015

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=103.0,
        shares=10.0,
        entry_score=92.0,
        setup_type="breakout",
        best_score=92.0,
        highest_close=103.0,
        has_new_high=False,
        score_improved=False,
        days_held=2,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "breakout_failed_followthrough"


def test_expansion_leader_exit_cuts_failed_followthrough_early():
    strategy = RallyPatternStrategy()
    row = pd.Series(_expansion_leader_row("2024-01-24", "AAA", 111.0))
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["score"] = 78.0
    row["close_vs_ema_20"] = 0.05

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=112.0,
        shares=10.0,
        entry_score=98.0,
        setup_type="expansion_leader",
        best_score=98.0,
        highest_close=112.0,
        has_new_high=False,
        score_improved=False,
        days_held=2,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "expansion_failed_followthrough"


def test_expansion_leader_aggressive_early_failure_respects_weak_close():
    strategy = RallyPatternStrategy(enable_aggressive_early_failure=True)
    row = pd.Series(_expansion_leader_row("2024-01-24", "AAA", 112.0))
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["score"] = 90.0
    row["close_vs_ema_20"] = 0.08
    row["close_pos"] = 0.45

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=113.0,
        shares=10.0,
        entry_score=98.0,
        setup_type="expansion_leader",
        best_score=98.0,
        highest_close=113.0,
        has_new_high=False,
        score_improved=False,
        days_held=2,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "expansion_failed_followthrough"


def test_power_breakout_exit_cuts_failed_followthrough_early_when_enabled():
    strategy = RallyPatternStrategy(enable_aggressive_early_failure=True)
    row = pd.Series(_power_breakout_row("2024-01-24", "AAA", 108.0))
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["score"] = 79.0
    row["close_vs_ema_20"] = 0.02
    row["close_pos"] = 0.50

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=109.0,
        shares=10.0,
        entry_score=95.0,
        setup_type="power_breakout",
        best_score=95.0,
        highest_close=109.0,
        has_new_high=False,
        score_improved=False,
        days_held=2,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "power_breakout_failed_followthrough"


def test_power_breakout_does_not_use_aggressive_early_failure_by_default():
    strategy = RallyPatternStrategy()
    row = pd.Series(_power_breakout_row("2024-01-24", "AAA", 108.0))
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["score"] = 79.0
    row["close_vs_ema_20"] = 0.02
    row["close_pos"] = 0.50

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=109.0,
        shares=10.0,
        entry_score=95.0,
        setup_type="power_breakout",
        best_score=95.0,
        highest_close=109.0,
        has_new_high=False,
        score_improved=False,
        days_held=2,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) is None


def test_bb_micro_support_fail_exits_power_breakout_when_enabled():
    strategy = RallyPatternStrategy(enable_bb_micro_failure=True)
    row = pd.Series(_power_breakout_row("2024-01-24", "AAA", 108.0))
    row["score"] = 82.0
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["close_vs_ema_20"] = 0.01
    row["close_vs_sma_50"] = 0.03
    row["rs_spy_20"] = 0.02
    row["rs_qqq_20"] = 0.02
    row["bb_micro_support_fail"] = True

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=109.0,
        shares=10.0,
        entry_score=95.0,
        setup_type="power_breakout",
        best_score=95.0,
        highest_close=109.0,
        has_new_high=False,
        score_improved=False,
        days_held=3,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "bb_micro_support_fail"


def test_bb_micro_support_fail_exits_expansion_leader_when_enabled():
    strategy = RallyPatternStrategy(enable_bb_micro_failure=True)
    row = pd.Series(_expansion_leader_row("2024-01-24", "AAA", 111.0))
    row["score"] = 88.0
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["close_vs_ema_20"] = 0.02
    row["close_vs_sma_50"] = 0.04
    row["rs_spy_20"] = 0.02
    row["rs_qqq_20"] = 0.02
    row["bb_micro_support_fail"] = True

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=112.0,
        shares=10.0,
        entry_score=98.0,
        setup_type="expansion_leader",
        best_score=98.0,
        highest_close=112.0,
        has_new_high=False,
        score_improved=False,
        days_held=4,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "bb_micro_support_fail"


def test_bb_micro_support_fail_is_disabled_by_default():
    strategy = RallyPatternStrategy()
    row = pd.Series(_power_breakout_row("2024-01-24", "AAA", 108.0))
    row["score"] = 82.0
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["close_vs_ema_20"] = 0.01
    row["close_vs_sma_50"] = 0.03
    row["rs_spy_20"] = 0.02
    row["rs_qqq_20"] = 0.02
    row["bb_micro_support_fail"] = True

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=109.0,
        shares=10.0,
        entry_score=95.0,
        setup_type="power_breakout",
        best_score=95.0,
        highest_close=109.0,
        has_new_high=False,
        score_improved=False,
        days_held=3,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) is None


def test_medium_confirm_failure_exits_power_breakout_when_enabled():
    strategy = RallyPatternStrategy(enable_medium_confirm_failure=True)
    row = pd.Series(_power_breakout_row("2024-01-24", "AAA", 108.0))
    row["score"] = 82.0
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["close_vs_ema_20"] = 0.01
    row["close_vs_sma_50"] = 0.03
    row["rs_spy_20"] = 0.02
    row["rs_qqq_20"] = 0.02
    row["medium_confirm_failure"] = True

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=109.0,
        shares=10.0,
        entry_score=95.0,
        setup_type="power_breakout",
        best_score=95.0,
        highest_close=109.0,
        has_new_high=False,
        score_improved=False,
        days_held=4,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "medium_confirm_failure"


def test_medium_confirm_failure_exits_expansion_leader_when_enabled():
    strategy = RallyPatternStrategy(enable_medium_confirm_failure=True)
    row = pd.Series(_expansion_leader_row("2024-01-24", "AAA", 111.0))
    row["score"] = 88.0
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = True
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["close_vs_ema_20"] = 0.02
    row["close_vs_sma_50"] = 0.04
    row["rs_spy_20"] = -0.01
    row["rs_qqq_20"] = -0.01
    row["medium_confirm_failure"] = True

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=112.0,
        shares=10.0,
        entry_score=98.0,
        setup_type="expansion_leader",
        best_score=98.0,
        highest_close=112.0,
        has_new_high=False,
        score_improved=False,
        days_held=4,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) == "medium_confirm_failure"


def test_medium_confirm_failure_is_disabled_by_default():
    strategy = RallyPatternStrategy()
    row = pd.Series(_power_breakout_row("2024-01-24", "AAA", 108.0))
    row["score"] = 82.0
    row["close_below_ema20"] = False
    row["close_below_ema20_2d"] = False
    row["close_below_sma50"] = False
    row["relative_weak"] = False
    row["relative_weak_2d"] = False
    row["soft_score_fail_2d"] = False
    row["roll_low_10"] = 100.0
    row["atr_14"] = 3.0
    row["close_vs_ema_20"] = 0.01
    row["close_vs_sma_50"] = 0.03
    row["rs_spy_20"] = 0.02
    row["rs_qqq_20"] = 0.02
    row["medium_confirm_failure"] = True

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-22"),
        entry_price=109.0,
        shares=10.0,
        entry_score=95.0,
        setup_type="power_breakout",
        best_score=95.0,
        highest_close=109.0,
        has_new_high=False,
        score_improved=False,
        days_held=4,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) is None


def test_trend_hold_skips_shallow_ema20_fail_while_above_sma50():
    strategy = RallyPatternStrategy()
    row = pd.Series(_strong_row("2024-01-24", "AAA", 108.0))
    row["relative_weak_2d"] = False
    row["close_below_ema20_2d"] = True
    row["close_below_sma50"] = False
    row["soft_score_fail_2d"] = False
    row["atr_14"] = 2.0
    row["roll_low_10"] = 103.0
    row["score"] = 78.0
    row["close_vs_ema_20"] = 0.01
    row["close_vs_sma_50"] = 0.04
    row["rs_spy_20"] = 0.05
    row["rs_qqq_20"] = 0.04

    position = _BacktestPosition(
        ticker="AAA",
        entry_date=pd.Timestamp("2024-01-02"),
        entry_price=96.0,
        shares=10.0,
        entry_score=92.0,
        setup_type="power_breakout",
        best_score=98.0,
        highest_close=114.0,
        has_new_high=True,
        score_improved=True,
        days_held=22,
        add_on_count=0,
    )

    assert strategy._exit_reason(row, position) is None


def test_rank_candidates_excludes_benchmark_tickers():
    strategy = RallyPatternStrategy()
    rows = [
        _strong_row("2024-01-02", "AAA", 100.0),
        _strong_row("2024-01-03", "AAA", 101.0),
        _strong_row("2024-01-04", "AAA", 103.0),
        _strong_row("2024-01-02", "SPY", 400.0),
        _strong_row("2024-01-03", "SPY", 401.0),
        _strong_row("2024-01-04", "SPY", 403.0),
        _strong_row("2024-01-02", "QQQ", 300.0),
        _strong_row("2024-01-03", "QQQ", 301.0),
        _strong_row("2024-01-04", "QQQ", 303.0),
    ]
    rows[2]["volume_ratio_20"] = 1.30
    rows[5]["volume_ratio_20"] = 1.30
    rows[8]["volume_ratio_20"] = 1.30

    ranked = strategy.rank_candidates(pd.DataFrame(rows))

    assert ranked["ticker"].tolist() == ["AAA"]
