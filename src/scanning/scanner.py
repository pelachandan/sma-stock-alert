"""
Long-Term Position Trading Scanner
===================================
Complete scanner for long-term position strategies (60-120 day holds).
Target: 8-20 trades/year total across all strategies.

STRATEGIES:
1. EMA_Crossover_Position
2. %B_MeanReversion_Position
3. High52_Position
4. BigBase_Breakout_Position
5. TrendContinuation_Position
6. RelativeStrength_Ranker_Position
"""

import pandas as pd
import numpy as np
import os
import traceback
from datetime import datetime
from src.data.market import get_historical_data
from src.data.indicators import compute_rsi, compute_bollinger_bands, compute_percent_b
from src.analysis.sectors import get_ticker_sector
from src.analysis.regime import get_regime_label, get_regime_config, is_short_regime_ok
from src.analysis.market_regime import get_position_regime, get_regime_params
from src.scanning.rs_bought_tracker import RSBoughtTracker
from src.strategies.registry import StrategyRegistry
from src.config.settings import (
    # Global settings
    POSITION_INITIAL_EQUITY,
    POSITION_RISK_PER_TRADE_PCT,
    POSITION_MAX_PER_STRATEGY,
    POSITION_MAX_TOTAL,
    MIN_LIQUIDITY_USD,
    MIN_PRICE,
    MAX_PRICE,
    REGIME_INDEX,
    REGIME_BULL_MA,
    REGIME_BEAR_MA,
    STRATEGY_PRIORITY,
    TECH_SECTORS,

    # Universal filters
    UNIVERSAL_RS_MIN,
    UNIVERSAL_ADX_MIN,
    UNIVERSAL_VOLUME_MULT,
    UNIVERSAL_ALL_MAS_RISING,
    UNIVERSAL_QQQ_BULL_MA,
    UNIVERSAL_QQQ_MA_RISING_DAYS,

    # Strategy 1: EMA_Crossover_Position
    EMA_CROSS_POS_VOLUME_MULT,
    EMA_CROSS_POS_STOP_ATR_MULT,
    EMA_CROSS_POS_MAX_DAYS,

    # Strategy 2: %B_MeanReversion_Position
    PERCENT_B_POS_OVERSOLD,
    PERCENT_B_POS_RSI_OVERSOLD,
    PERCENT_B_POS_STOP_ATR_MULT,
    PERCENT_B_POS_MAX_DAYS,

    # Strategy 4: High52_Position
    HIGH52_POS_RS_MIN,
    HIGH52_POS_VOLUME_MULT,
    HIGH52_POS_ADX_MIN,
    HIGH52_POS_STOP_ATR_MULT,
    HIGH52_POS_MAX_DAYS,

    # Strategy 5: BigBase_Breakout_Position
    BIGBASE_MIN_WEEKS,
    BIGBASE_MAX_RANGE_PCT,
    BIGBASE_RS_MIN,
    BIGBASE_VOLUME_MULT,
    BIGBASE_STOP_ATR_MULT,
    BIGBASE_MAX_DAYS,

    # Strategy 6: TrendContinuation_Position
    TREND_CONT_MA_LOOKBACK,
    TREND_CONT_MA_RISING_DAYS,
    TREND_CONT_RS_THRESHOLD,
    TREND_CONT_RSI_MIN,
    TREND_CONT_PULLBACK_EMA,
    TREND_CONT_PULLBACK_ATR,
    TREND_CONT_STOP_ATR_MULT,
    TREND_CONT_MAX_DAYS,

    # Strategy 7: RelativeStrength_Ranker_Position
    RS_RANKER_SECTORS,
    RS_RANKER_TOP_N,
    RS_RANKER_RS_THRESHOLD,
    RS_RANKER_STOP_ATR_MULT,
    RS_RANKER_MAX_DAYS,

    # Short Strategies (regime-based)
    SHORT_ENABLED,
    SHORT_REJECTION_MA,
    SHORT_REJECTION_TOLERANCE,
    SHORT_CFG_BULL,
    SHORT_CFG_SIDEWAYS,
    SHORT_CFG_BEAR,
    LEADER_SHORT_CFG_BULL,
    LEADER_SHORT_ALLOWED_REGIMES,

)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr


def calculate_relative_strength(stock_df, index_df, days=126):
    """
    Calculate 6-month (126 trading days) relative strength vs index
    Returns: RS value (e.g., 0.15 = stock outperformed by 15%)
    """
    if len(stock_df) < days or len(index_df) < days:
        return None

    stock_return = (stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-days]) - 1.0
    index_return = (index_df["Close"].iloc[-1] / index_df["Close"].iloc[-days]) - 1.0

    return stock_return - index_return


def check_regime_bullish(index_df, ma_period=200):
    """Check if index is in bullish regime (close > MA)"""
    if len(index_df) < ma_period:
        return False

    close = index_df["Close"].iloc[-1]
    ma = index_df["Close"].rolling(ma_period).mean().iloc[-1]

    return close > ma


def check_regime_bearish(index_df, ma_period=200):
    """Check if index is in bearish regime (close < MA falling)"""
    if len(index_df) < ma_period + 20:
        return False

    close = index_df["Close"].iloc[-1]
    ma_current = index_df["Close"].rolling(ma_period).mean().iloc[-1]
    ma_20d_ago = index_df["Close"].rolling(ma_period).mean().iloc[-21]

    return close < ma_current and ma_current < ma_20d_ago


def check_ma_rising(df, period, lookback_days):
    """Check if MA is rising over lookback days"""
    if len(df) < period + lookback_days:
        return False

    ma_current = df["Close"].rolling(period).mean().iloc[-1]
    ma_past = df["Close"].rolling(period).mean().iloc[-lookback_days-1]

    return ma_current > ma_past


def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index) for trend strength"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Calculate True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Calculate smoothed TR and DMs
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # Calculate DX and ADX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(period).mean()

    return adx


def check_all_mas_rising(df, lookback_days=20):
    """Check if MA50, MA100, and MA200 are ALL rising over lookback period"""
    if len(df) < 200 + lookback_days:
        return False

    ma50 = df["Close"].rolling(50).mean()
    ma100 = df["Close"].rolling(100).mean()
    ma200 = df["Close"].rolling(200).mean()

    ma50_rising = ma50.iloc[-1] > ma50.iloc[-lookback_days]
    ma100_rising = ma100.iloc[-1] > ma100.iloc[-lookback_days]
    ma200_rising = ma200.iloc[-1] > ma200.iloc[-lookback_days]

    return ma50_rising and ma100_rising and ma200_rising


def get_sector_for_ticker(ticker):
    """
    Get sector for a ticker. Returns sector name or None if not found.

    In production, this would query from a data provider.
    For now, uses hardcoded mappings for common S&P 500 stocks.
    """
    # Sector mapping for major S&P 500 stocks
    SECTOR_MAP = {
        # Technology
        "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AVGO": "Technology",
        "ORCL": "Technology", "CRM": "Technology", "ADBE": "Technology", "AMD": "Technology",
        "INTC": "Technology", "QCOM": "Technology", "TXN": "Technology", "AMAT": "Technology",
        "LRCX": "Technology", "KLAC": "Technology", "MCHP": "Technology", "MU": "Technology",
        "ANET": "Technology", "PANW": "Technology", "CRWD": "Technology", "SNPS": "Technology",
        "CDNS": "Technology", "ON": "Technology", "PLTR": "Technology", "HPE": "Technology",

        # Communication Services
        "META": "Communication Services", "GOOGL": "Communication Services", "GOOG": "Communication Services",
        "NFLX": "Communication Services", "DIS": "Communication Services", "CMCSA": "Communication Services",
        "T": "Communication Services", "VZ": "Communication Services", "TMUS": "Communication Services",

        # Consumer Discretionary
        "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
        "NKE": "Consumer Discretionary", "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
        "TGT": "Consumer Discretionary", "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
        "BKNG": "Consumer Discretionary", "MAR": "Consumer Discretionary", "RCL": "Consumer Discretionary",
        "EXPE": "Consumer Discretionary", "YUM": "Consumer Discretionary", "CMG": "Consumer Discretionary",

        # Financials
        "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS": "Financials",
        "MS": "Financials", "C": "Financials", "BLK": "Financials", "SPGI": "Financials",
        "CME": "Financials", "ICE": "Financials", "AXP": "Financials", "SCHW": "Financials",
        "COIN": "Financials",

        # Industrials
        "BA": "Industrials", "CAT": "Industrials", "DE": "Industrials", "GE": "Industrials",
        "RTX": "Industrials", "LMT": "Industrials", "UPS": "Industrials", "FDX": "Industrials",
        "UNP": "Industrials", "NSC": "Industrials", "CSX": "Industrials",

        # Energy (BLACKLIST)
        "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
        "EOG": "Energy", "MPC": "Energy", "PSX": "Energy", "VLO": "Energy",

        # Consumer Staples (BLACKLIST)
        "WMT": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
        "PEP": "Consumer Staples", "COST": "Consumer Staples", "WBA": "Consumer Staples",
        "CL": "Consumer Staples", "KMB": "Consumer Staples", "GIS": "Consumer Staples",

        # Utilities (BLACKLIST)
        "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
        "AEP": "Utilities", "EXC": "Utilities", "XEL": "Utilities",

        # Real Estate (BLACKLIST)
        "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
        "PSA": "Real Estate", "SPG": "Real Estate", "O": "Real Estate",

        # Healthcare (BLACKLIST - mostly defensive)
        "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare", "ABBV": "Healthcare",
        "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
        "PFE": "Healthcare", "CVS": "Healthcare", "MCK": "Healthcare", "CI": "Healthcare",
        "HUM": "Healthcare", "ELV": "Healthcare",

        # Materials (BLACKLIST - gold, commodities)
        "NEM": "Materials", "FCX": "Materials", "NUE": "Materials", "DOW": "Materials",
        "LIN": "Materials", "APD": "Materials", "ECL": "Materials", "SHW": "Materials",

        # Tobacco/Defensive (BLACKLIST)
        "PM": "Consumer Staples", "MO": "Consumer Staples", "BTI": "Consumer Staples",

        # Utilities (power/energy)
        "CEG": "Utilities", "VST": "Utilities", "AES": "Utilities",

        # Semiconductors (Technology - high priority)
        "ASML": "Information Technology", "TSM": "Information Technology", "STX": "Information Technology", "WDC": "Information Technology",
        "SMCI": "Information Technology", "ARM": "Information Technology",

        # Tech Hardware
        "IBM": "Information Technology", "DELL": "Information Technology", "HPQ": "Information Technology", "NTAP": "Information Technology",

        # Aerospace/Defense (Industrials)
        "NOC": "Industrials", "GD": "Industrials", "HII": "Industrials",

        # Pharma/Biotech (Healthcare - defensive)
        "GILD": "Healthcare", "AMGN": "Healthcare", "BMY": "Healthcare", "REGN": "Healthcare",
        "VRTX": "Healthcare", "BIIB": "Healthcare", "MRNA": "Healthcare",

        # Software (Technology)
        "NOW": "Information Technology", "INTU": "Information Technology", "WDAY": "Information Technology", "TEAM": "Information Technology",
        "ZM": "Information Technology", "ZS": "Information Technology", "DDOG": "Information Technology", "SNOW": "Information Technology",
        "OKTA": "Information Technology", "SHOP": "Information Technology",
    }

    return SECTOR_MAP.get(ticker, None)


def detect_liquidity_zone(df, cfg):
    """
    Detect liquidity/consolidation zone over the lookback period.

    Returns:
        dict with:
            - zone_high: float
            - zone_low: float
            - zone_detected: bool (True if valid consolidation zone found)
            - consolidation_bars: int (number of bars in consolidation)
    """
    lookback = cfg.get("ZONE_LOOKBACK", 20)
    compression_mult = cfg.get("ZONE_COMPRESSION_ATR_MULT", 3.0)
    min_bars = cfg.get("ZONE_MIN_BARS", 8)
    consol_threshold = cfg.get("ZONE_CONSOLIDATION_THRESHOLD", 0.8)

    if len(df) < lookback + 20:  # Need ATR data too
        return {"zone_high": None, "zone_low": None, "zone_detected": False, "consolidation_bars": 0}

    # Get recent data - EXCLUDE today's bar (use previous 20 bars only)
    # This is critical! If we include today, close < zone_low is often impossible
    recent_df = df.iloc[-lookback-1:-1]  # Previous 20 bars, NOT including current bar
    zone_high = recent_df["High"].max()
    zone_low = recent_df["Low"].min()
    zone_range = zone_high - zone_low

    # Calculate ATR20
    atr20 = calculate_atr(df, 20)
    if atr20 is None or len(atr20) == 0:
        return {"zone_high": zone_high, "zone_low": zone_low, "zone_detected": False, "consolidation_bars": 0}

    current_atr = atr20.iloc[-1]

    # DEBUG MODE: Drastically simplified zone detection
    # Just check if range <= 5 * ATR20, no other filters
    zone_detected = zone_range <= (compression_mult * current_atr)
    consolidation_bars = 0  # Not used in debug mode

    return {
        "zone_high": zone_high,
        "zone_low": zone_low,
        "zone_range": zone_range,
        "atr20": current_atr,
        "zone_detected": zone_detected,
        "consolidation_bars": consolidation_bars,
    }


# =============================================================================
# REGISTRY INTEGRATION
# =============================================================================
# Strategies handled by hardcoded blocks in run_scan_as_of().
# Any strategy registered in StrategyRegistry but NOT in this set
# will be dispatched automatically via strategy.scan(ticker, df, as_of_date).
_LEGACY_STRATEGY_NAMES = frozenset({
    "EMA_Crossover_Position",
    "%B_MeanReversion_Position",
    "High52_Position",
    "BigBase_Breakout_Position",
    "TrendContinuation_Position",
    "RelativeStrength_Ranker_Position",
    "ShortWeakRS_Retrace_Position",
    "LeaderPullback_Short_Position",
    "ConsumerDisc_Ranker_Position",
})


def _backtest_debug_enabled() -> bool:
    return os.getenv("STOCK_ALERT_BACKTEST_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _report_registry_strategy_exception(context: str, name: str, exc: Exception) -> None:
    if not _backtest_debug_enabled():
        return
    print(f"❌ [registry-strategy] {context} failed for {name}: {exc}")
    traceback.print_exc()


def _get_active_registry_strategies():
    """Return list of (name, strategy_instance) for all registry strategies NOT in legacy set."""
    from src.config.settings import POSITION_MAX_PER_STRATEGY
    result = []
    for name in StrategyRegistry.list_available():
        if name in _LEGACY_STRATEGY_NAMES:
            continue
        if POSITION_MAX_PER_STRATEGY.get(name, 0) > 0:
            try:
                result.append((name, StrategyRegistry.create(name)))
            except Exception as exc:
                _report_registry_strategy_exception("create", name, exc)
                pass
    return result


# =============================================================================
# MAIN SCANNER FUNCTION
# =============================================================================

def run_scan_as_of(as_of_date, tickers, rs_bought_tracker=None):
    """
    Walk-forward scanner for long-term position strategies.
    Returns signals with priority ordering for deduplication.
    """
    as_of_date = pd.to_datetime(as_of_date)

    # -------------------------------------------------
    # Load index data for regime filters
    # -------------------------------------------------
    qqq_df = get_historical_data(REGIME_INDEX)
    if not qqq_df.empty and isinstance(qqq_df.index, pd.DatetimeIndex):
        qqq_df = qqq_df[qqq_df.index <= as_of_date]
    else:
        qqq_df = pd.DataFrame()

    # Check regime (STRONGER: QQQ > 100-MA AND MA100 rising)
    qqq_bull_basic = check_regime_bullish(qqq_df, UNIVERSAL_QQQ_BULL_MA) if not qqq_df.empty else False
    qqq_ma_rising = check_ma_rising(qqq_df, UNIVERSAL_QQQ_BULL_MA, UNIVERSAL_QQQ_MA_RISING_DAYS) if not qqq_df.empty else False
    is_bull_regime = qqq_bull_basic and qqq_ma_rising
    is_bear_regime = check_regime_bearish(qqq_df, REGIME_BEAR_MA) if not qqq_df.empty else False
    
    # Detect position regime for ADX threshold adjustment
    try:
        position_regime = get_position_regime(as_of_date=as_of_date, index_symbol="QQQ")
        regime_params = get_regime_params(position_regime)
        adx_threshold = regime_params.get('adx_threshold', UNIVERSAL_ADX_MIN)
    except Exception:
        # Fallback to default if regime detection fails
        adx_threshold = UNIVERSAL_ADX_MIN

    signals = []
    
    # Initialize or use provided RS Ranker bought tracker
    # If tracker passed from backtester, use it for persistent tracking across scans
    # If called directly (live), create fresh tracker
    if rs_bought_tracker is None:
        rs_tracker = RSBoughtTracker()
    else:
        rs_tracker = rs_bought_tracker
    for ticker in tickers:
        df = get_historical_data(ticker)
        if df.empty:
            continue

        # Ensure index is datetime (safety check for string indices)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                df = df[df.index.notna()]
            except:
                continue

        # Cut future data
        df = df[df.index <= as_of_date]

        # Need sufficient history
        if len(df) < 252:  # 1 year minimum
            continue

        # Basic data
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        open_col = df["Open"] if "Open" in df.columns else close
        last_close = close.iloc[-1]

        # Skip if price too low/high
        if last_close < MIN_PRICE or last_close > MAX_PRICE:
            continue

        # Liquidity check
        avg_vol_20d = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else 0
        dollar_volume = avg_vol_20d * last_close
        if dollar_volume < MIN_LIQUIDITY_USD:
            continue

        # Calculate common indicators
        ema20 = close.ewm(span=20).mean()
        ema21 = close.ewm(span=21).mean()
        ema34 = close.ewm(span=34).mean()
        ema50 = close.ewm(span=50).mean()
        ema100 = close.ewm(span=100).mean()
        ma50 = close.rolling(50).mean()
        ma100 = close.rolling(100).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()

        rsi14 = compute_rsi(close, 14)
        atr14 = calculate_atr(df, 14)
        atr20 = calculate_atr(df, 20)
        adx14 = calculate_adx(df, 14)

        # Relative strength vs index
        rs_6mo = calculate_relative_strength(df, qqq_df, 126) if not qqq_df.empty else None

        # Universal filters (pre-calculate for all strategies)
        all_mas_rising = check_all_mas_rising(df, UNIVERSAL_QQQ_MA_RISING_DAYS) if UNIVERSAL_ALL_MAS_RISING else True
        strong_adx = adx14.iloc[-1] >= adx_threshold if not pd.isna(adx14.iloc[-1]) else False

        # =====================================================================
        # STRATEGY 1: EMA_CROSSOVER_POSITION
        # =====================================================================
        # Entry: Strong trend, EMA20 crosses above EMA50 + new 50-day high
        # Regime: Bull (QQQ > 200-MA)
        # =====================================================================
        if is_bull_regime and len(df) >= 100:
            try:
                # Check for EMA20 crossing EMA50 in last 3 days
                ema20_crossed_ema50 = False
                for i in range(1, 4):
                    if i < len(ema20) and ema20.iloc[-i] <= ema50.iloc[-i] and ema20.iloc[-i+1] > ema50.iloc[-i+1]:
                        ema20_crossed_ema50 = True
                        break

                if ema20_crossed_ema50:
                    # MULTI-MONTH TREND FILTERS (Position Trading)
                    # Stacked MAs: Price > 50 > 100 > 200
                    stacked_mas = (last_close > ma50.iloc[-1] and
                                   ma50.iloc[-1] > ma100.iloc[-1] and
                                   ma100.iloc[-1] > ma200.iloc[-1])

                    # 50-day MA rising over 20 days
                    ma50_rising = check_ma_rising(df, 50, 20)

                    # Strong RS requirement (vs QQQ)
                    strong_rs = rs_6mo is not None and rs_6mo >= 0.20  # +20% vs QQQ

                    # New 50-day high
                    high_50d = high.rolling(50).max().iloc[-1]
                    is_new_high = last_close >= high_50d * 0.995  # Within 0.5%

                    # Volume confirmation
                    vol_ratio = volume.iloc[-1] / max(avg_vol_20d, 1)
                    volume_confirmed = vol_ratio >= EMA_CROSS_POS_VOLUME_MULT

                    if all([stacked_mas, ma50_rising, strong_rs, is_new_high, volume_confirmed]):
                        # Calculate stop and quality score
                        current_atr = atr14.iloc[-1]
                        stop_price = last_close - (EMA_CROSS_POS_STOP_ATR_MULT * current_atr)

                        # Quality score
                        trend_strength = (ma50.iloc[-1] - ma100.iloc[-1]) / ma100.iloc[-1] * 100
                        score = min(trend_strength * 5, 50)  # Max 50
                        score += min(vol_ratio / EMA_CROSS_POS_VOLUME_MULT * 25, 25)  # Max 25
                        score += 25 if rs_6mo and rs_6mo > 0 else 0  # Bonus for positive RS

                        signals.append({
                            "Ticker": ticker,
                            "Strategy": "EMA_Crossover_Position",
                            "Priority": STRATEGY_PRIORITY["EMA_Crossover_Position"],
                            "Price": round(last_close, 2),
                            "StopPrice": round(stop_price, 2),
                            "ATR14": round(current_atr, 2),
                            "Score": round(score, 2),
                            "AsOfDate": as_of_date,
                            "MaxDays": EMA_CROSS_POS_MAX_DAYS,
                        })
            except Exception:
                pass

        # =====================================================================
        # STRATEGY 2: %B_MEANREVERSION_POSITION
        # =====================================================================
        # Entry: %B < 0.12, RSI14 < 38, then close above lower BB
        # =====================================================================
        if is_bull_regime and len(df) >= 150:
            try:
                # Calculate Bollinger Bands
                middle_band, upper_band, lower_band, bandwidth = compute_bollinger_bands(close, period=20, std_dev=2)
                percent_b = compute_percent_b(close, upper_band, lower_band)

                if not percent_b.isna().iloc[-1]:
                    percent_b_value = percent_b.iloc[-1]

                    # Long-term uptrend
                    close_above_ma150 = last_close > ma150.iloc[-1]
                    ma150_rising = check_ma_rising(df, 150, 20)

                    # Oversold conditions
                    percent_b_oversold = percent_b_value < PERCENT_B_POS_OVERSOLD
                    rsi_oversold = rsi14.iloc[-1] < PERCENT_B_POS_RSI_OVERSOLD

                    # Trigger: Close back above lower BB and prior high
                    close_above_lower_bb = last_close > lower_band.iloc[-1]
                    if len(high) >= 2:
                        close_above_prior_high = last_close > high.iloc[-2]
                    else:
                        close_above_prior_high = False

                    if all([close_above_ma150, ma150_rising, percent_b_oversold, rsi_oversold,
                           close_above_lower_bb, close_above_prior_high]):
                        # Stop
                        stop_price = last_close - (PERCENT_B_POS_STOP_ATR_MULT * atr14.iloc[-1])

                        # Quality score
                        score = (PERCENT_B_POS_OVERSOLD - percent_b_value) * 500  # Max 60
                        score += (PERCENT_B_POS_RSI_OVERSOLD - rsi14.iloc[-1]) * 1.5  # Max 40

                        signals.append({
                            "Ticker": ticker,
                            "Strategy": "%B_MeanReversion_Position",
                            "Priority": STRATEGY_PRIORITY["%B_MeanReversion_Position"],
                            "Price": round(last_close, 2),
                            "StopPrice": round(stop_price, 2),
                            "ATR14": round(atr14.iloc[-1], 2),
                            "PercentB": round(percent_b_value, 2),
                            "RSI14": round(rsi14.iloc[-1], 2),
                            "Score": round(score, 2),
                            "AsOfDate": as_of_date,
                            "MaxDays": PERCENT_B_POS_MAX_DAYS,
                        })
            except Exception:
                pass

        # =====================================================================
        # STRATEGY 4: HIGH52_POSITION (ULTRA-SELECTIVE)
        # =====================================================================
        # Entry: 30% RS (leaders only), new 52-week high, 2.5x volume explosion,
        #        ADX 30+, stacked MAs
        # Goal: Catch ONLY high-conviction breakouts, not exhaustion tops
        # =====================================================================
        if is_bull_regime and len(df) >= 252 and rs_6mo is not None:
            try:
                # MULTI-MONTH TREND FILTERS
                # Stacked MAs: Price > 50 > 100 > 200
                stacked_mas = (last_close > ma50.iloc[-1] and
                               ma50.iloc[-1] > ma100.iloc[-1] and
                               ma100.iloc[-1] > ma200.iloc[-1])

                # New 52-week high
                high_52w = high.rolling(252).max().iloc[-1]
                is_new_52w_high = last_close >= high_52w * 0.998  # Within 0.2%

                # RS requirement - LEADERS ONLY (30%+ outperformance)
                strong_rs = rs_6mo >= HIGH52_POS_RS_MIN

                # Volume EXPLOSION (single-day conviction, not 5-day avg)
                avg_vol_50d = volume.rolling(50).mean().iloc[-1] if len(volume) >= 50 else avg_vol_20d
                vol_ratio = volume.iloc[-1] / max(avg_vol_50d, 1)
                volume_surge = vol_ratio >= HIGH52_POS_VOLUME_MULT  # 2.5x single-day

                # ADX confirmation (momentum strength)
                has_momentum = adx14.iloc[-1] >= HIGH52_POS_ADX_MIN if not pd.isna(adx14.iloc[-1]) else False

                # ULTRA-SELECTIVE: All filters must pass
                if all([stacked_mas, is_new_52w_high, strong_rs, volume_surge, has_momentum]):
                    # Stop
                    stop_price = last_close - (HIGH52_POS_STOP_ATR_MULT * atr20.iloc[-1])

                    # Quality score
                    score = min(rs_6mo / 0.30 * 50, 70)  # Max 70 (adjusted for 30% threshold)
                    score += min((vol_ratio / HIGH52_POS_VOLUME_MULT) * 30, 30)

                    signals.append({
                        "Ticker": ticker,
                        "Strategy": "High52_Position",
                        "Priority": STRATEGY_PRIORITY["High52_Position"],
                        "Price": round(last_close, 2),
                        "StopPrice": round(stop_price, 2),
                        "ATR20": round(atr20.iloc[-1], 2),
                        "RS_6mo": round(rs_6mo * 100, 2),
                        "VolumeRatio": round(vol_ratio, 2),
                        "Score": round(score, 2),
                        "AsOfDate": as_of_date,
                        "MaxDays": HIGH52_POS_MAX_DAYS,
                    })
            except Exception:
                pass

        # =====================================================================
        # STRATEGY 5: BIGBASE_BREAKOUT_POSITION (ACTIVE - RARE HOME RUNS)
        # =====================================================================
        # Entry: 14+ week consolidation (≤22% range), 6-mo high breakout, RS 15%+, 1.5x 5-day vol
        # Note: NO ADX requirement (consolidations have low ADX by definition)
        # =====================================================================
        if is_bull_regime and len(df) >= 140:  # 14+ weeks * 5 days + buffer
            try:
                # Check 14-week (70-day) base
                lookback_days = BIGBASE_MIN_WEEKS * 5
                if len(df) >= lookback_days:
                    base_high = high.iloc[-lookback_days:].max()
                    base_low = low.iloc[-lookback_days:].min()
                    base_range_pct = (base_high - base_low) / base_low

                    # Tight base (≤22% range - controlled consolidation)
                    is_tight_base = base_range_pct <= BIGBASE_MAX_RANGE_PCT

                    # MULTI-MONTH TREND FILTERS
                    # Base must be above 200-day MA (long-term uptrend)
                    above_200ma = last_close > ma200.iloc[-1]

                    # RS requirement - strong performers (15%+ outperformance)
                    strong_rs = rs_6mo is not None and rs_6mo >= BIGBASE_RS_MIN

                    # New 6-month high breakout
                    high_6mo = high.rolling(126).max().iloc[-1]
                    is_breakout = last_close >= high_6mo * 0.998

                    # Volume confirmation: 5-day average (sustained, not spike)
                    avg_vol_50d = volume.rolling(50).mean().iloc[-1] if len(volume) >= 50 else avg_vol_20d
                    vol_5d_avg = volume.iloc[-5:].mean() if len(volume) >= 5 else volume.iloc[-1]
                    vol_ratio = vol_5d_avg / max(avg_vol_50d, 1)
                    volume_surge = vol_ratio >= BIGBASE_VOLUME_MULT  # 1.5x 5-day avg (sustained interest)

                    # RELAXED: Removed all_mas_rising and ADX filters
                    # ADX is LOW during consolidation, rises AFTER breakout (catches it too late)
                    if all([is_tight_base, above_200ma, strong_rs,
                           is_breakout, volume_surge]):
                        # Stop: ATR-based from entry (aligned with backtester)
                        stop_price = last_close - (BIGBASE_STOP_ATR_MULT * atr20.iloc[-1])

                        # Quality score (HIGH - this is rare!)
                        score = 80  # Base score
                        score += (BIGBASE_MAX_RANGE_PCT - base_range_pct) / BIGBASE_MAX_RANGE_PCT * 10
                        score += min((vol_ratio / BIGBASE_VOLUME_MULT) * 10, 10)

                        signals.append({
                            "Ticker": ticker,
                            "Strategy": "BigBase_Breakout_Position",
                            "Priority": STRATEGY_PRIORITY["BigBase_Breakout_Position"],
                            "Price": round(last_close, 2),
                            "StopPrice": round(stop_price, 2),
                            "ATR14": round(atr14.iloc[-1], 2),
                            "BaseRangePct": round(base_range_pct * 100, 2),
                            "VolumeRatio": round(vol_ratio, 2),
                            "Score": round(score, 2),
                            "AsOfDate": as_of_date,
                            "MaxDays": BIGBASE_MAX_DAYS,
                        })
            except Exception:
                pass

        # =====================================================================
        # STRATEGY 6: TRENDCONTINUATION_POSITION (NEW)
        # =====================================================================
        # Entry: Strong trend, 150-MA rising, pullback to 21-EMA, then resume
        # =====================================================================
        if is_bull_regime and len(df) >= 150 and rs_6mo is not None:
            try:
                # MULTI-MONTH TREND FILTERS
                # Stacked MAs: Price > 50 > 100 > 150 > 200
                stacked_mas = (last_close > ma50.iloc[-1] and
                               ma50.iloc[-1] > ma100.iloc[-1] and
                               ma100.iloc[-1] > ma150.iloc[-1] and
                               ma150.iloc[-1] > ma200.iloc[-1])

                # 150-MA rising over 20 days
                ma150_rising = check_ma_rising(df, 150, TREND_CONT_MA_RISING_DAYS)

                # Very strong RS (>+25% vs QQQ - already strong)
                strong_rs = rs_6mo >= TREND_CONT_RS_THRESHOLD

                # Pullback to 21-EMA
                ema21_value = ema21.iloc[-1]
                pullback_distance = abs(last_close - ema21_value) / ema21_value
                near_ema21 = pullback_distance <= (TREND_CONT_PULLBACK_ATR * atr14.iloc[-1] / last_close)

                # RSI not too weak
                rsi_ok = rsi14.iloc[-1] >= TREND_CONT_RSI_MIN

                # Trigger: Close > prior high AND > 21-EMA
                close_above_ema21 = last_close > ema21_value
                if len(high) >= 2:
                    close_above_prior_high = last_close > high.iloc[-2]
                else:
                    close_above_prior_high = False

                if all([stacked_mas, ma150_rising, strong_rs,
                       near_ema21, rsi_ok, close_above_ema21, close_above_prior_high]):
                    # Stop: Swing low or 3x ATR
                    swing_low = low.iloc[-10:].min() if len(low) >= 10 else last_close
                    stop_atr = last_close - (TREND_CONT_STOP_ATR_MULT * atr14.iloc[-1])
                    stop_price = max(swing_low, stop_atr)  # Most conservative

                    # Quality score
                    score = min((rs_6mo / TREND_CONT_RS_THRESHOLD) * 50, 70)  # Max 70
                    score += min((rsi14.iloc[-1] - TREND_CONT_RSI_MIN) / 20 * 30, 30)

                    signals.append({
                        "Ticker": ticker,
                        "Strategy": "TrendContinuation_Position",
                        "Priority": STRATEGY_PRIORITY["TrendContinuation_Position"],
                        "Price": round(last_close, 2),
                        "StopPrice": round(stop_price, 2),
                        "ATR14": round(atr14.iloc[-1], 2),
                        "RS_6mo": round(rs_6mo * 100, 2),
                        "RSI14": round(rsi14.iloc[-1], 2),
                        "Score": round(score, 2),
                        "AsOfDate": as_of_date,
                        "MaxDays": TREND_CONT_MAX_DAYS,
                    })
            except Exception:
                pass

        # =====================================================================
        # STRATEGY 7: RELATIVESTRENGTH_RANKER_POSITION (ACTIVE - BEST PERFORMER)
        # =====================================================================
        # Entry: Tech stocks, RS > +30%, new 3-mo high or pullback, ADX 30+, all MAs rising
        # =====================================================================
        if rs_6mo is not None and POSITION_MAX_PER_STRATEGY.get("RelativeStrength_Ranker_Position", 0) > 0:
            try:
                # Check if already bought - skip duplicate BUY signals
                if rs_tracker.is_bought(ticker):
                    pass  # Skip, already in active position
                elif rs_tracker.has_recent_stop(ticker, trading_days_lookback=5, as_of_date=as_of_date):
                    pass  # Skip, stopped out within last 5 trading days
                else:
                    # Check if ticker is in tech sectors
                    ticker_sector = get_ticker_sector(ticker)
                    is_tech = ticker_sector in RS_RANKER_SECTORS

                    if is_tech:
                        # VOLATILITY FILTER (Skip overly volatile stocks prone to whipsaw)
                        daily_returns = close.pct_change()
                        volatility_20d = daily_returns.rolling(20).std().iloc[-1] if len(daily_returns) >= 20 else 0
                        if volatility_20d > 0.04:  # More than 4% daily volatility
                            if ticker == "WDC" and as_of_date >= pd.Timestamp('2024-04-20') and as_of_date <= pd.Timestamp('2024-05-05'):
                                print(f"  [FAIL] Volatility: {volatility_20d:.2%}")
                            continue  # Too volatile, skip

                        # MULTI-MONTH TREND FILTERS
                        # Stacked MAs: Price > 50 > 100 > 200
                        stacked_mas = (last_close > ma50.iloc[-1] and
                                       ma50.iloc[-1] > ma100.iloc[-1] and
                                       ma100.iloc[-1] > ma200.iloc[-1])

                        # UNIVERSAL FILTERS (STRONGER)
                        strong_rs = rs_6mo >= UNIVERSAL_RS_MIN  # 30% minimum

                        # Trigger options:
                        # Option A: New 3-month high
                        high_3mo = high.rolling(63).max().iloc[-1] if len(high) >= 63 else 0
                        is_3mo_high = last_close >= high_3mo * 0.995

                        # Option B: Pullback to 21-EMA then close above
                        near_ema21 = abs(last_close - ema21.iloc[-1]) / ema21.iloc[-1] < 0.02  # Within 2%
                        if len(high) >= 2:
                            close_above_prior = last_close > high.iloc[-2]
                        else:
                            close_above_prior = False
                        pullback_breakout = near_ema21 and close_above_prior

                        if all([stacked_mas, strong_rs,
                               (is_3mo_high or pullback_breakout), strong_adx]):
                            # Stop
                            stop_price = last_close - (RS_RANKER_STOP_ATR_MULT * atr20.iloc[-1])

                            # Quality score (high for top RS)
                            score = min((rs_6mo / RS_RANKER_RS_THRESHOLD) * 100, 100)

                            signals.append({
                                "Ticker": ticker,
                                "Strategy": "RelativeStrength_Ranker_Position",
                                "Priority": STRATEGY_PRIORITY["RelativeStrength_Ranker_Position"],
                                "Price": round(last_close, 2),
                                "StopPrice": round(stop_price, 2),
                                "ATR20": round(atr20.iloc[-1], 2),
                                "RS_6mo": round(rs_6mo * 100, 2),
                                "Score": round(score, 2),
                                "AsOfDate": as_of_date,
                                "MaxDays": RS_RANKER_MAX_DAYS,
                            })
                            
                            # Record this ticker as recommended
                            rs_tracker.add_bought(
                                ticker,
                                entry_date=str(as_of_date.date()),
                                entry_price=round(last_close, 2)
                            )
            except Exception:
                pass

        # =====================================================================
        # =====================================================================
        # STRATEGY 8: SHORT_WEAKRS_RETRACE_POSITION (REGIME-BASED)
        # =====================================================================
        # Adaptive short strategy with different parameters for each regime:
        # - BULL: Extremely selective (RS ≤ -15%, RSI 70-80, 30d max hold)
        # - SIDEWAYS: Range-trade mean reversion (RS ≤ -10%, RSI 65-75, 20d max)
        # - BEAR: Primary offensive (RS ≤ -5%, RSI 55-70, 45d max hold)
        # =====================================================================
        if rs_6mo is not None and SHORT_ENABLED and POSITION_MAX_PER_STRATEGY.get("ShortWeakRS_Retrace_Position", 0) > 0:
            try:
                # REGIME CLASSIFICATION
                regime = get_regime_label(as_of_date)
                cfg = get_regime_config(regime)
                short_regime_ok = is_short_regime_ok(regime, allow_bull_shorts=True)

                if not short_regime_ok:
                    pass  # Skip this ticker (regime doesn't allow shorts)
                else:
                    # ENTRY FILTERS (using regime-specific config)

                    # Filter 1: Weak relative strength
                    weak_rs = rs_6mo <= cfg["RS_MAX"]

                    # Filter 2: Downtrend - price below declining MA
                    if len(df) >= cfg["MA_PERIOD"]:
                        ma_short = close.rolling(cfg["MA_PERIOD"]).mean()
                        below_ma = last_close < ma_short.iloc[-1]

                        # Check if MA is declining (if required)
                        if cfg["MA_DECLINING_DAYS"] > 0 and len(ma_short) >= cfg["MA_DECLINING_DAYS"]:
                            ma_declining = ma_short.iloc[-1] <= ma_short.iloc[-cfg["MA_DECLINING_DAYS"]]
                        else:
                            ma_declining = True  # No decline requirement
                    else:
                        below_ma = False
                        ma_declining = False

                    # Filter 3: Rally to 50-MA with rejection
                    if len(df) >= 50:
                        ma50_short = close.rolling(50).mean()
                        ma50_val = ma50_short.iloc[-1]

                        # High touched 50-MA (within tolerance band)
                        tol = SHORT_REJECTION_TOLERANCE
                        high_touched_ma50 = (
                            high.iloc[-1] >= ma50_val * (1 - tol) and
                            high.iloc[-1] <= ma50_val * (1 + tol)
                        )

                        # Close below 50-MA (rejection)
                        close_below_ma50 = last_close < ma50_val

                        # Optional upper wick (strong rejection)
                        upper_wick_pct = (high.iloc[-1] - last_close) / last_close if last_close > 0 else 0.0
                        if cfg["REQUIRE_WICK"]:
                            strong_rejection = upper_wick_pct >= cfg["WICK_MIN"]
                        else:
                            strong_rejection = True  # Not required
                    else:
                        high_touched_ma50 = False
                        close_below_ma50 = False
                        strong_rejection = False

                    # Filter 4: Lower high (structure confirmation)
                    if cfg["REQUIRE_LOWER_HIGH"] and len(high) >= 10:
                        # Find peaks in last 10 bars
                        recent_peaks = []
                        for i in range(-10, -2):
                            if i >= -len(high) + 1:
                                if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i+1]:
                                    recent_peaks.append(high.iloc[i])

                        # Check most recent bar separately
                        if len(high) >= 2:
                            if high.iloc[-1] > high.iloc[-2]:
                                recent_peaks.append(high.iloc[-1])

                        if len(recent_peaks) >= 2:
                            lower_high = recent_peaks[-1] < recent_peaks[-2]
                        else:
                            lower_high = True  # Not enough peaks
                    else:
                        lower_high = True  # Not required

                    # Filter 5: RSI (regime-specific thresholds)
                    current_rsi = rsi14.iloc[-1]
                    rsi_in_range = cfg["RSI_MIN"] <= current_rsi <= cfg["RSI_MAX"]

                    # Filter 6: Volatility filter
                    atr_pct = (atr14.iloc[-1] / last_close) if last_close > 0 else 999
                    if cfg["MAX_ATR_PCT"] is not None:
                        volatility_ok = atr_pct <= cfg["MAX_ATR_PCT"]
                    else:
                        volatility_ok = True

                    # Filter 7: Volume filter
                    if len(volume) >= 50:
                        volume_ok = volume.iloc[-1] >= volume.iloc[-50:].mean() * cfg["MIN_VOL_MULT"]
                    else:
                        volume_ok = True

                    # ALL conditions must be TRUE for short entry
                    if all([weak_rs, below_ma, ma_declining,
                            high_touched_ma50, close_below_ma50, strong_rejection,
                            lower_high, rsi_in_range,
                            volatility_ok, volume_ok]):

                        # Calculate stop loss (regime-specific)
                        recent_high = high.iloc[-5:].max()
                        ma50_stop = ma50_val if 'ma50_val' in locals() else recent_high
                        stop_above_resistance = max(recent_high, ma50_stop)

                        # Add ATR buffer
                        stop_price = stop_above_resistance + (
                            cfg["STOP_ATR_MULT"] * atr14.iloc[-1]
                        ) + (
                            cfg["STOP_BUFFER_ATR"] * atr14.iloc[-1]
                        )

                        # Target for partial (regime-specific R-multiple)
                        risk_per_share = stop_price - last_close
                        target_price = last_close - (cfg["PARTIAL_R"] * risk_per_share)

                        # Score: Higher for weaker RS
                        score = abs(rs_6mo) * 100

                        signals.append({
                            "Ticker": ticker,
                            "Strategy": "ShortWeakRS_Retrace_Position",
                            "Entry": last_close,
                            "StopLoss": stop_price,
                            "Target": target_price,
                            "Score": score,
                            "RS": rs_6mo,
                            "MaxDays": cfg["MAX_DAYS"],
                            "Direction": "SHORT",
                            "Priority": STRATEGY_PRIORITY.get("ShortWeakRS_Retrace_Position", 100),
                            "Regime": regime,  # NEW: Track regime for this entry
                        })
            except Exception:
                pass

        # =====================================================================
        # STRATEGY 9: LEADERPULLBACK_SHORT_POSITION (BULL/SIDEWAYS REGIMES)
        # =====================================================================
        # Leader Pullback Short - Liquidity Zone Breakdown (Perplexity v2.0)
        #
        # Entry Logic:
        # 1. Universe: Large cap ($20B+), liquid ($100M+ volume), sector whitelist
        # 2. Leader: RS percentile >= 80 (top 20%), was extended in last 30 bars
        # 3. Liquidity zone: 20-bar consolidation with compressed range
        # 4. Signal: Close breaks below zone_low with impulsive weakness + volume
        # 5. Stop: Above zone_high (logical level)
        #
        # Exit: 50% at +2R, early stop at 20d if R<=0, hard stop at 40d
        # Portfolio: Max 3 positions, 0.35% risk per trade
        # =====================================================================
        if rs_6mo is not None and SHORT_ENABLED and POSITION_MAX_PER_STRATEGY.get("LeaderPullback_Short_Position", 0) > 0:
            try:
                # Active in configured regimes (bull/sideways)
                regime = get_regime_label(as_of_date)
                cfg = LEADER_SHORT_CFG_BULL

                # DEBUG: Track filtering stages
                debug_counters = getattr(run_scan_as_of, '_debug_counters', None)
                if debug_counters is None and cfg.get("DEBUG_MODE", False):
                    debug_counters = {
                        'total_checked': 0,
                        'failed_price': 0,
                        'failed_dollar_vol': 0,
                        'failed_sector': 0,
                        'failed_zone': 0,
                        'failed_entry': 0,
                        'passed_all': 0,
                    }
                    run_scan_as_of._debug_counters = debug_counters

                if regime in LEADER_SHORT_ALLOWED_REGIMES and cfg["ENABLED"]:
                    if cfg.get("DEBUG_MODE", False) and debug_counters:
                        debug_counters['total_checked'] += 1

                    # Require minimum bars
                    if len(df) < 100:
                        pass  # Skip
                    else:
                        # ==============================================================
                        # UNIVERSE FILTERS
                        # ==============================================================

                        # 1. Price filter
                        min_price = cfg.get("MIN_PRICE", 30)
                        if last_close < min_price:
                            if cfg.get("DEBUG_MODE", False) and debug_counters:
                                debug_counters['failed_price'] += 1
                            continue

                        # 2. Dollar volume filter (liquidity)
                        dollar_vol_20d = avg_vol_20d * last_close
                        min_dollar_vol = cfg.get("MIN_DOLLAR_VOLUME", 100_000_000)
                        if dollar_vol_20d < min_dollar_vol:
                            if cfg.get("DEBUG_MODE", False) and debug_counters:
                                debug_counters['failed_dollar_vol'] += 1
                            continue

                        # 3. Sector filter (whitelist/blacklist)
                        sector = get_sector_for_ticker(ticker)
                        sector_whitelist = cfg.get("SECTOR_WHITELIST", [])
                        sector_blacklist = cfg.get("SECTOR_BLACKLIST", [])

                        # If sector unknown, allow (benefit of doubt)
                        if sector is not None:
                            # Check blacklist first
                            if sector in sector_blacklist:
                                if cfg.get("DEBUG_MODE", False) and debug_counters:
                                    debug_counters['failed_sector'] += 1
                                continue  # Skip blacklisted sectors

                            # Check whitelist (if defined)
                            if sector_whitelist and sector not in sector_whitelist:
                                if cfg.get("DEBUG_MODE", False) and debug_counters:
                                    debug_counters['failed_sector'] += 1
                                continue  # Skip if not in whitelist

                        # ==============================================================
                        # LEADER CONTEXT
                        # ==============================================================

                        if not cfg.get("DEBUG_MODE", False):
                            # Full context filters (only when not in debug mode)

                            # 1. RS percentile >= 80 (top 20%)
                            context_strong_rs = rs_6mo >= 0.10

                            # 2. Historical extension check
                            context_extended = False
                            lookback_ext = cfg.get("EXTENSION_LOOKBACK", 30)
                            if len(close) >= cfg["EXTENSION_MA50"] + lookback_ext:
                                ma50_series = close.rolling(cfg["EXTENSION_MA50"]).mean()
                                recent_closes = close.iloc[-lookback_ext:]
                                recent_ma50 = ma50_series.iloc[-lookback_ext:]
                                ext_ratios_ma50 = recent_closes / recent_ma50
                                was_extended_ma50 = (ext_ratios_ma50 >= cfg.get("EXTENSION_HISTORICAL_MIN_MA50", 1.08)).any()
                            else:
                                was_extended_ma50 = False

                            if len(close) >= cfg["EXTENSION_MA100"] + lookback_ext:
                                ma100_series = close.rolling(cfg["EXTENSION_MA100"]).mean()
                                recent_closes = close.iloc[-lookback_ext:]
                                recent_ma100 = ma100_series.iloc[-lookback_ext:]
                                ext_ratios_ma100 = recent_closes / recent_ma100
                                was_extended_ma100 = (ext_ratios_ma100 >= cfg.get("EXTENSION_HISTORICAL_MIN_MA100", 1.12)).any()
                            else:
                                was_extended_ma100 = False

                            context_extended = was_extended_ma50 or was_extended_ma100

                            # 3. Currently above MA50 or MA100
                            if len(close) >= cfg["EXTENSION_MA50"]:
                                ma50_series = close.rolling(cfg["EXTENSION_MA50"]).mean()
                                ma50_val = ma50_series.iloc[-1]
                                above_ma50 = last_close >= ma50_val * cfg.get("EXTENSION_CURRENT_MIN_MA50", 1.00)
                            else:
                                above_ma50 = False

                            if len(close) >= cfg["EXTENSION_MA100"]:
                                ma100_series = close.rolling(cfg["EXTENSION_MA100"]).mean()
                                ma100_val = ma100_series.iloc[-1]
                                above_ma100 = last_close >= ma100_val * cfg.get("EXTENSION_CURRENT_MIN_MA100", 1.00)
                            else:
                                above_ma100 = False

                            context_above_ma = above_ma50 or above_ma100

                            # 4. Was overbought recently (RSI >= 65 in last 20 bars)
                            rsi14 = compute_rsi(close, cfg["RSI_PERIOD"])
                            if len(rsi14) >= cfg.get("RSI_LOOKBACK", 20):
                                max_rsi = rsi14.iloc[-cfg.get("RSI_LOOKBACK", 20):].max()
                                context_was_overbought = max_rsi >= cfg.get("RSI_CLIMAX", 65)
                            else:
                                context_was_overbought = False

                            # All context must pass
                            context_ok = all([
                                context_strong_rs,
                                context_extended,
                                context_above_ma,
                                context_was_overbought,
                            ])

                            if not context_ok:
                                continue

                        # ==============================================================
                        # LIQUIDITY ZONE DETECTION
                        # ==============================================================

                        zone = detect_liquidity_zone(df, cfg)

                        if not zone["zone_detected"]:
                            if cfg.get("DEBUG_MODE", False) and debug_counters:
                                debug_counters['failed_zone'] += 1
                            continue  # No valid consolidation zone found

                        zone_high = zone["zone_high"]
                        zone_low = zone["zone_low"]
                        atr20 = zone["atr20"]

                        # ==============================================================
                        # ENTRY SIGNALS
                        # ==============================================================

                        # Get today's OHLC
                        today_high = df["High"].iloc[-1]
                        today_low = df["Low"].iloc[-1]
                        today_volume = df["Volume"].iloc[-1]

                        # Calculate average volume
                        vol_20 = df["Volume"].iloc[-20:].mean() if len(df) >= 20 else df["Volume"].mean()

                        if cfg.get("DEBUG_MODE", False):
                            # DEBUG MODE: Ultra-simplified entry (no RSI, no variants)
                            zone_break = last_close < zone_low
                            enter_short = zone_break
                            signal_type = "zone_break_debug"

                            # DEBUG: Log details for specific date and tickers
                            debug_date = pd.to_datetime("2024-06-27")
                            debug_tickers = ["MU", "NVDA", "MSFT", "AAPL", "TSLA", "META", "GOOGL", "AMZN", "NFLX", "AMD"]
                            if as_of_date.date() == debug_date.date() and ticker in debug_tickers:
                                print(f"[DEBUG] {as_of_date.date()} {ticker}: close={last_close:.2f}, zone_low={zone_low:.2f}, zone_high={zone_high:.2f}, close<zone_low={zone_break}, enter={enter_short}")

                        else:
                            # FULL ENTRY LOGIC (OR-based variants)
                            today_open = df["Open"].iloc[-1] if "Open" in df.columns else last_close
                            vol_50 = df["Volume"].iloc[-50:].mean() if len(df) >= 50 else df["Volume"].mean()

                            # Core conditions (always required)
                            core_zone_break = last_close < zone_low
                            core_volume = today_volume >= (vol_50 * cfg.get("CORE_VOLUME_MULT", 1.2))

                            # Not a hammer
                            core_not_hammer = True
                            if cfg.get("REJECT_HAMMER", True):
                                day_range = today_high - today_low
                                if day_range > 0:
                                    close_position = (last_close - today_low) / day_range
                                    if close_position > cfg.get("HAMMER_THRESHOLD", 0.70):
                                        core_not_hammer = False

                            core_ok = core_zone_break and core_volume and core_not_hammer
                            if not core_ok:
                                continue

                            # Entry variants (need core + ANY ONE)
                            # Variant A: Core only (simplest)
                            variant_A = cfg.get("VARIANT_A_ENABLED", True)

                            # Variant B: Core + impulsive weakness (gap down OR wide range)
                            variant_B = False
                            if cfg.get("VARIANT_B_ENABLED", True):
                                gap_down = today_open < zone_low and today_open < df["Close"].iloc[-2]
                                true_range = today_high - today_low
                                wide_range = true_range >= (cfg.get("WIDE_RANGE_ATR_MULT", 1.2) * atr20)
                                close_below_open = last_close < today_open
                                impulsive_weakness = gap_down or (wide_range and close_below_open)
                                variant_B = impulsive_weakness

                            # Variant C: Core + big volume spike only
                            # (RSI already checked in context, no need to duplicate)
                            variant_C = False
                            if cfg.get("VARIANT_C_ENABLED", True):
                                # Big volume spike
                                big_volume = today_volume >= (vol_50 * cfg.get("BIG_VOLUME_MULT", 1.5))
                                variant_C = big_volume

                            # Entry if ANY variant passes
                            enter_short = variant_A or variant_B or variant_C

                            # Determine which variant triggered
                            if variant_C:
                                signal_type = "zone_big_volume"
                            elif variant_B:
                                gap_down = today_open < zone_low and today_open < df["Close"].iloc[-2]
                                if gap_down:
                                    signal_type = "zone_gap_down"
                                else:
                                    signal_type = "zone_wide_range"
                            else:
                                signal_type = "zone_break_simple"

                        if not enter_short:
                            if cfg.get("DEBUG_MODE", False) and debug_counters:
                                debug_counters['failed_entry'] += 1
                            continue

                        # ==============================================================
                        # STOP PLACEMENT (Logical level above zone)
                        # ==============================================================

                        # Start with zone_high
                        stop_price = zone_high * (1 + cfg.get("STOP_BUFFER_PCT", 0.01))

                        # Also consider MA20 if configured
                        if cfg.get("STOP_ALSO_MA20", True) and len(close) >= 20:
                            ma20 = close.rolling(20).mean().iloc[-1]
                            stop_price = max(stop_price, ma20 * 1.01)

                        # Also consider recent swing high if configured
                        if cfg.get("STOP_ALSO_SWING_HIGH", True) and len(df) >= 10:
                            recent_swing_high = df["High"].iloc[-10:].max()
                            stop_price = max(stop_price, recent_swing_high * 1.005)

                        # Calculate stop distance
                        stop_distance = stop_price - last_close  # Positive value for short
                        if stop_distance <= 0:
                            continue  # Invalid stop (price already above stop)

                        # Calculate ATR-based risk (for position sizing)
                        atr14 = calculate_atr(df, 14)
                        if atr14 is None or len(atr14) == 0:
                            continue
                        current_atr = atr14.iloc[-1]

                        # Position size based on stop distance
                        risk_per_trade = cfg.get("RISK_PER_TRADE_PCT", 0.35) / 100
                        risk_dollars = POSITION_INITIAL_EQUITY * risk_per_trade
                        shares = int(risk_dollars / stop_distance)

                        if shares <= 0:
                            continue

                        # ==============================================================
                        # GENERATE SIGNAL
                        # ==============================================================

                        entry_price = last_close
                        score = rs_6mo * 100  # Use RS as score

                        # Calculate target for SHORT position
                        # For shorts: target is BELOW entry (profit on downside)
                        # Use 2:1 reward:risk ratio
                        risk_per_share = stop_price - entry_price  # Positive value (stop is above entry)
                        target_price = entry_price - (2.0 * risk_per_share)  # Target below entry

                        # signal_type already determined above in variant logic

                        if cfg.get("DEBUG_MODE", False) and debug_counters:
                            debug_counters['passed_all'] += 1

                        signals.append({
                            "Ticker": ticker,
                            "Date": as_of_date,
                            "Strategy": "LeaderPullback_Short_Position",
                            "PositionType": "Full",
                            "Entry": entry_price,
                            "StopLoss": stop_price,
                            "Target": target_price,  # REQUIRED by pre_buy_check
                            "Shares": shares,
                            "Score": score,
                            "RS": rs_6mo,
                            "MaxDays": cfg["MAX_DAYS"],
                            "Direction": "SHORT",
                            "Priority": STRATEGY_PRIORITY.get("LeaderPullback_Short_Position", 101),
                            "Regime": regime,
                            "StrategyType": signal_type,  # Track which signal triggered
                            "ZoneHigh": zone_high,
                            "ZoneLow": zone_low,
                            "ConsolidationBars": zone["consolidation_bars"],
                        })
            except Exception as e:
                print(f"❌ ERROR in LeaderPullback_Short for {ticker}: {e}")
                import traceback
                traceback.print_exc()

    # -------------------------------------------------
    # Post-processing: Sort by priority then score
    # -------------------------------------------------
    if signals:
        # Sort by priority (lower = higher priority) then by score (higher = better)
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values(
            by=["Priority", "Score"],
            ascending=[True, False]
        )
        signals = signals_df.to_dict("records")

    # DEBUG: Print filtering summary if debug mode was active
    debug_counters = getattr(run_scan_as_of, '_debug_counters', None)
    if debug_counters and debug_counters['total_checked'] > 0:
        print(f"\n🔍 DEBUG SUMMARY for {as_of_date.date()}:")
        print(f"   Total stocks checked: {debug_counters['total_checked']}")
        print(f"   Failed price filter (<$30): {debug_counters['failed_price']}")
        print(f"   Failed dollar volume (<$100M): {debug_counters['failed_dollar_vol']}")
        print(f"   Failed sector filter: {debug_counters['failed_sector']}")
        print(f"   Failed zone detection (range > 5×ATR): {debug_counters['failed_zone']}")
        print(f"   Failed entry conditions: {debug_counters['failed_entry']}")
        print(f"   ✅ PASSED ALL: {debug_counters['passed_all']}")
        # Reset counters for next day
        run_scan_as_of._debug_counters = None

    # =========================================================================
    # GAP REVERSAL POSITION
    # =========================================================================
    # Runs as a separate pass so it can use its own 60-bar minimum
    # (main loop requires 252 bars, which would filter out some valid setups).
    if POSITION_MAX_PER_STRATEGY.get("GapReversal_Position", 0) > 0:
        try:
            from src.strategies.gap_reversal import GapReversalPosition
            gap_strategy = GapReversalPosition()
            for ticker in tickers:
                try:
                    df = get_historical_data(ticker)
                    if df.empty:
                        continue
                    if not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                            df = df[df.index.notna()]
                        except Exception:
                            continue
                    df = df[df.index <= as_of_date]
                    if len(df) < 65:  # EMA21 + RSI10 + 30 warmup + buffer
                        continue
                    signal = gap_strategy.scan(ticker, df, as_of_date)
                    if signal is not None:
                        signals.append(signal)
                except Exception:
                    continue
        except Exception:
            pass

    # =========================================================================
    # GAP CONTINUATION POSITION
    # =========================================================================
    if POSITION_MAX_PER_STRATEGY.get("GapContinuation_Position", 0) > 0:
        try:
            from src.strategies.gap_continuation import GapContinuationPosition

            gap_continuation_strategy = GapContinuationPosition()
            for ticker in tickers:
                try:
                    df = get_historical_data(ticker)
                    if df.empty:
                        continue
                    if not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                            df = df[df.index.notna()]
                        except Exception:
                            continue
                    df = df[df.index <= as_of_date]
                    if len(df) < 65:
                        continue
                    signal = gap_continuation_strategy.scan(ticker, df, as_of_date)
                    if signal is not None:
                        signals.append(signal)
                except Exception:
                    continue
        except Exception:
            pass

    # =========================================================================
    # REGISTRY-DRIVEN STRATEGIES
    # =========================================================================
    for _, strategy in _get_active_registry_strategies():
        try:
            strategy_signals = strategy.run(tickers, as_of_date=as_of_date)
            if strategy_signals:
                signals.extend(strategy_signals)
        except Exception as exc:
            _report_registry_strategy_exception("run", getattr(strategy, "name", type(strategy).__name__), exc)
            continue

    # =========================================================================
    # Phase 2-3: SECTOR-BASED STRATEGIES
    # =========================================================================
    
    # Load sector ETF data for the surviving sector strategy
    xly_df = get_historical_data("XLY")  # Consumer Discretionary

    # Ensure all sector ETF indices are datetime
    for etf_df in [xly_df]:
        if not etf_df.empty and not isinstance(etf_df.index, pd.DatetimeIndex):
            try:
                etf_df.index = pd.to_datetime(etf_df.index, format='%Y-%m-%d', errors='coerce')
                etf_df = etf_df[etf_df.index.notna()]
            except:
                pass

    if not xly_df.empty:
        xly_df = xly_df[xly_df.index <= as_of_date]

    # Consumer Discretionary Ranker
    if POSITION_MAX_PER_STRATEGY.get("ConsumerDisc_Ranker_Position", 0) > 0 and not xly_df.empty:
        try:
            from src.strategies.consumer_disc_ranker import scan_consumer_disc
            cd_signals = scan_consumer_disc(tickers, as_of_date, xly_df, adx_threshold)
            signals.extend(cd_signals)
        except Exception as e:
            pass

    return signals
