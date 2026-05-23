"""
Central Trading Configuration
=============================
Single source of truth for all trading parameters.
Consolidates all strategy, risk, and regime settings.
"""

# =============================================================================
# GLOBAL POSITION TRADING SETTINGS
# =============================================================================

# Universal Filters (Applied to ALL active strategies)
UNIVERSAL_RS_MIN = 0.30               # Minimum +30% RS vs QQQ for ALL strategies
UNIVERSAL_ADX_MIN = 30                # Minimum ADX(14) >= 30 for strong trends
UNIVERSAL_VOLUME_MULT = 2.5           # Minimum 2.5x volume surge for ALL strategies
UNIVERSAL_ALL_MAS_RISING = True       # MA50, MA100, MA200 all must be rising
UNIVERSAL_QQQ_BULL_MA = 100           # QQQ > 100-MA (stronger than 200-MA)
UNIVERSAL_QQQ_MA_RISING_DAYS = 20     # QQQ MA100 must be rising over 20 days

# Risk Management
POSITION_INITIAL_EQUITY = 100000   # Starting equity for position sizing ($100k default)
POSITION_RISK_PER_TRADE_PCT = 2.0  # 2.0% of equity per trade
POSITION_MAX_TOTAL = 20            # Max 20 total positions

# Per-Strategy Position Limits
POSITION_MAX_PER_STRATEGY = {
    "GapReversal_Position": 10,               # ACTIVE: 46.8% WR, 0.58R, +$457k net (backtested)
    "GapContinuation_Position": 10,           # ACTIVE: bullish earnings / gap-and-go continuation
    "RelativeStrength_Ranker_Position": 10,   # ACTIVE: 48.8% WR, 2.16R, +$620k net (backtested)
    "RallyPattern_Position": 10,              # ACTIVE: daily rally-pattern leader scan
    "ConsumerDisc_Ranker_Position": 0,        # DISABLED: unvalidated
    "High52_Position": 0,                      # DISABLED
    "BigBase_Breakout_Position": 0,           # DISABLED
    "EMA_Crossover_Position": 0,              # DISABLED
    "EMA_StackAlignment_Position": 0,         # DISABLED
    "TrendContinuation_Position": 0,          # DISABLED
    "%B_MeanReversion_Position": 0,           # DISABLED
    "ShortWeakRS_Retrace_Position": 0,        # DISABLED
    "LeaderPullback_Short_Position": 0,       # DISABLED
}

POSITION_MAX_PER_STRATEGY_DEFAULT = 5

# Time Horizons
POSITION_MAX_DAYS_SHORT = 90       # Mean reversion styles (60-90 days)
POSITION_MAX_DAYS_LONG = 120       # Momentum/breakout styles (60-120 days)

# Partial Profits
POSITION_PARTIAL_ENABLED = True
POSITION_PARTIAL_SIZE = 0.3              # 30% (runner = 70%)
POSITION_PARTIAL_R_TRIGGER_LOW = 2.0     # Most strategies
POSITION_PARTIAL_R_TRIGGER_MID = 2.5     # High52, BigBase
POSITION_PARTIAL_R_TRIGGER_HIGH = 3.0    # RS_Ranker

# =============================================================================
# PYRAMIDING (ADD TO WINNERS)
# =============================================================================

POSITION_PYRAMID_ENABLED = True
POSITION_PYRAMID_R_TRIGGER = 1.5      # Add after +1.5R profit
POSITION_PYRAMID_SIZE = 0.5           # 50% of original position size
POSITION_PYRAMID_MAX_ADDS = 3         # Maximum 3 add-ons per position
POSITION_PYRAMID_PULLBACK_EMA = 21    # Must pull back to 21-day EMA
POSITION_PYRAMID_PULLBACK_ATR = 1.0   # Within 1 ATR of EMA21

# =============================================================================
# STRATEGY PRIORITY (DEDUPLICATION)
# =============================================================================

STRATEGY_PRIORITY = {
    "BigBase_Breakout_Position": 1,
    "RelativeStrength_Ranker_Position": 2,
    "RallyPattern_Position": 3,
    "TrendContinuation_Position": 4,
    "EMA_Crossover_Position": 5,
    "EMA_StackAlignment_Position": 6,
    "High52_Position": 7,
    "%B_MeanReversion_Position": 8,
}

# =============================================================================
# STRATEGY-SPECIFIC PARAMETERS
# =============================================================================

# 1. EMA_CROSSOVER_POSITION
EMA_CROSS_POS_VOLUME_MULT = 1.5
EMA_CROSS_POS_STOP_ATR_MULT = 3.5
EMA_CROSS_POS_PARTIAL_R = 2.0
EMA_CROSS_POS_PARTIAL_SIZE = 0.3
EMA_CROSS_POS_TRAIL_MA = 100
EMA_CROSS_POS_TRAIL_DAYS = 5
EMA_CROSS_POS_MAX_DAYS = 120

# 2. %B_MEANREVERSION_POSITION
PERCENT_B_POS_OVERSOLD = 0.12
PERCENT_B_POS_RSI_OVERSOLD = 38
PERCENT_B_POS_STOP_ATR_MULT = 3.5
PERCENT_B_POS_PARTIAL_R = 2.0
PERCENT_B_POS_PARTIAL_SIZE = 0.3
PERCENT_B_POS_TRAIL_MA = 50
PERCENT_B_POS_TRAIL_DAYS = 5
PERCENT_B_POS_MAX_DAYS = 90

# 4. HIGH52_POSITION
HIGH52_POS_RS_MIN = 0.30
HIGH52_POS_VOLUME_MULT = 2.5
HIGH52_POS_ADX_MIN = 30
HIGH52_POS_STOP_ATR_MULT = 4.5
HIGH52_POS_PARTIAL_R = 2.5
HIGH52_POS_PARTIAL_SIZE = 0.3
HIGH52_POS_TRAIL_MA = 100
HIGH52_POS_TRAIL_DAYS = 8
HIGH52_POS_MAX_DAYS = 150

# 5. BIGBASE_BREAKOUT_POSITION
BIGBASE_MIN_WEEKS = 14
BIGBASE_MAX_RANGE_PCT = 0.22
BIGBASE_RS_MIN = 0.15
BIGBASE_VOLUME_MULT = 1.5
BIGBASE_STOP_ATR_MULT = 4.5
BIGBASE_PARTIAL_R = 4.0
BIGBASE_PARTIAL_SIZE = 0.3
BIGBASE_TRAIL_MA = 200
BIGBASE_TRAIL_DAYS = 10
BIGBASE_MAX_DAYS = 180

# 6. TRENDCONTINUATION_POSITION
TREND_CONT_MA_LOOKBACK = 150
TREND_CONT_MA_RISING_DAYS = 20
TREND_CONT_RS_THRESHOLD = 0.25
TREND_CONT_RSI_MIN = 45
TREND_CONT_PULLBACK_EMA = 21
TREND_CONT_PULLBACK_ATR = 1.0
TREND_CONT_STOP_ATR_MULT = 3.5
TREND_CONT_PARTIAL_R = 2.0
TREND_CONT_PARTIAL_SIZE = 0.3
TREND_CONT_TRAIL_MA = 50
TREND_CONT_TRAIL_DAYS = 5
TREND_CONT_MAX_DAYS = 90

# 7. RELATIVESTRENGTH_RANKER_POSITION
RS_RANKER_SECTORS = ["Information Technology", "Communication Services"]
RS_RANKER_TOP_N = 10
RS_RANKER_RS_THRESHOLD = 0.30
RS_RANKER_STOP_ATR_MULT = 4.5
RS_RANKER_PARTIAL_R = 3.0
RS_RANKER_PARTIAL_SIZE = 0.3
RS_RANKER_TRAIL_MA = 100
RS_RANKER_TRAIL_DAYS = 10
RS_RANKER_MAX_DAYS = 150

# 8. RALLYPATTERN_POSITION — keep repo defaults empty so tuning can live only in
# local/GCS config overrides. The live wrapper falls back to built-in strategy
# defaults when this dict is empty.
RALLY_PATTERN_CONFIG = {}

# 9. GAPREVERSAL_POSITION — Breakaway Gap Reversal Strategy
# Long: gap up + smoothed RSI(10 on EMA21) < 10 + weekly trend UP
# Short: gap down + smoothed RSI(10 on EMA21) > 90 + weekly trend DOWN
# Stop: gap fill (prior close). Exit: trailing EMA21.
GAP_REVERSAL_MIN_GAP_PCT = 0.005       # Minimum 0.5% gap (baseline check)
GAP_REVERSAL_MIN_GAP_ATR_MULT = 1.0    # Gap must also be ≥ 1×ATR20 (noise filter)
GAP_REVERSAL_MIN_VOL_MULT = 1.5        # Gap day volume must be ≥ 1.5× 20-day avg
GAP_REVERSAL_RSI_OVERSOLD = 10         # smoothed RSI threshold for long
GAP_REVERSAL_RSI_OVERBOUGHT = 90       # smoothed RSI threshold for short
GAP_REVERSAL_EMA_PERIOD = 21           # EMA period used for price smoothing
GAP_REVERSAL_RSI_PERIOD = 10           # RSI period computed on EMA series
GAP_REVERSAL_TRAIL_MA = 21             # Trailing exit MA period
GAP_REVERSAL_MAX_DAYS = 120            # Maximum hold period
GAP_REVERSAL_TARGET_R_MULTIPLE = 2     # Initial target = entry + N×risk (2R default)
GAP_REVERSAL_DIRECTION = "both"        # "long", "short", or "both"
GAP_REVERSAL_WEEKLY_TF_FILTER = True   # Require weekly trend alignment
GAP_REVERSAL_PRIORITY = 1              # Signal priority (lower = higher priority)
GAP_REVERSAL_MAX_GAP_AGE_DAYS = 5      # Reject gap signals if the gap bar is older than this many
                                        # calendar days vs as_of_date (guards against stale CSV data)
# Prior move filter: stock must have declined/rallied before the gap
# For longs: prior close <= recent_high * (1 - DECLINE_PCT) — stock declined at least X%
# For shorts: prior close >= recent_low * (1 + RALLY_PCT) — stock rallied at least X%
GAP_REVERSAL_PRIOR_DECLINE_LOOKBACK = 20   # bars to look back for prior high/low
GAP_REVERSAL_PRIOR_DECLINE_PCT = 0.10      # stock must be down ≥10% from lookback high (long)
GAP_REVERSAL_PRIOR_RALLY_PCT = 0.10        # stock must be up ≥10% from lookback low (short)
GAP_REVERSAL_SHORT_PRIOR_RALLY_PCT = 0.20  # shorts need ≥20% prior rally — stricter quality gate
GAP_REVERSAL_SHORT_REGIME_FILTER = True    # enable regime filter for shorts
GAP_REVERSAL_SHORT_REQUIRE_RISK_OFF = False  # False = block only RISK_ON (allow NEUTRAL+RISK_OFF)
                                              # True  = block all except RISK_OFF (bear market only)
GAP_REVERSAL_LONG_MACRO_FILTER = True      # skip long reversal gaps in high-risk macro weeks

# 10. GAPCONTINUATION_POSITION — Bullish earnings gap continuation
# Long: qualified gap setup + post-gap hold + confirmed breakout + practical room/risk
# Stop: structural support with a minimum practical breathing distance. Exit: support fail / trailing EMA21.
GAP_CONTINUATION_MIN_GAP_PCT = 0.02
GAP_CONTINUATION_MIN_GAP_ATR_MULT = 1.0
GAP_CONTINUATION_MIN_VOL_MULT = 1.5
GAP_CONTINUATION_EMA_PERIOD = 21
GAP_CONTINUATION_RSI_PERIOD = 10
GAP_CONTINUATION_RSI_MIN = 55
GAP_CONTINUATION_RSI_MAX = 80
GAP_CONTINUATION_MIN_CLOSE_POS = 0.70
GAP_CONTINUATION_TRAIL_MA = 21
GAP_CONTINUATION_MAX_DAYS = 120
GAP_CONTINUATION_TARGET_R_MULTIPLE = 2
GAP_CONTINUATION_PRIORITY = 1
GAP_CONTINUATION_MAX_GAP_AGE_DAYS = 5
GAP_CONTINUATION_MIN_RS_20 = 0.0
GAP_CONTINUATION_WEEKLY_TF_FILTER = True
GAP_CONTINUATION_LONG_MACRO_FILTER = False
GAP_CONTINUATION_MAX_SHELF_DAYS = 5
GAP_CONTINUATION_MAX_SHELF_RANGE_PCT = 0.10
GAP_CONTINUATION_MIN_SHELF_CLOSE_POS = 0.50

# =============================================================================
# INDEX REGIME FILTERS
# =============================================================================

REGIME_INDEX = "QQQ"
REGIME_BULL_MA = 200
REGIME_BULL_MA_RISING_DAYS = 0
REGIME_BEAR_MA = 200
REGIME_BEAR_MA_FALLING_DAYS = 0
REGIME_INDEX_ALT = "SPY"

# =============================================================================
# LIQUIDITY & UNIVERSE FILTERS
# =============================================================================

MIN_LIQUIDITY_USD = 30_000_000
MIN_PRICE = 10.0
MAX_PRICE = 999999.0

TECH_SECTORS = ["Information Technology", "Communication Services"]

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================

BACKTEST_START_DATE = "2022-01-01"
BACKTEST_SCAN_FREQUENCY = "B"  # "B" = daily, "W-MON" = weekly Monday

# --- Compounding ---
# True  = position sizing uses current equity (grows with profits/losses)
# False = fixed initial capital (original behavior)
BACKTEST_COMPOUNDING = True

# --- Brokerage (Robinhood / Fidelity zero-commission) ---
# No per-trade commission, but regulatory fees apply on the SELL side:
#   SEC fee:   $0.0000278 per $1 of sell proceeds
#   FINRA TAF: $0.000119 per share sold, capped at $5.95/trade
BACKTEST_BROKERAGE_ENABLED = True
BACKTEST_SEC_FEE_RATE = 0.0000278       # per $ of sell proceeds
BACKTEST_FINRA_TAF_RATE = 0.000119      # per share sold
BACKTEST_FINRA_TAF_MAX = 5.95           # max FINRA TAF per trade

# --- Taxation (US, high earner bracket) ---
BACKTEST_TAX_ENABLED = True
BACKTEST_TAX_SHORT_TERM_RATE = 0.37     # ≤ 365 days held
BACKTEST_TAX_LONG_TERM_RATE = 0.20     # > 365 days held

# =============================================================================
# EMAIL & NOTIFICATION SETTINGS
# =============================================================================

MIN_NORM_SCORE = 7.0           # Minimum normalized score to send email
MAX_TRADES_EMAIL = 5           # Maximum trades in email alert

# =============================================================================
# LEGACY SETTINGS (KEPT FOR BACKWARD COMPATIBILITY)
# =============================================================================

CAPITAL_PER_TRADE = 20_000
RISK_REWARD_RATIO = 2
MAX_HOLDING_DAYS = 120
PARTIAL_EXIT_ENABLED = True
PARTIAL_EXIT_SIZE = 0.4
MAX_TRADES_PER_SCAN = 10
MAX_OPEN_POSITIONS = 25
REQUIRE_CONFIRMATION_BAR = False
MIN_HOLDING_DAYS = 0
CATASTROPHIC_LOSS_THRESHOLD = 999
MAX_ENTRY_GAP_PCT = 999
ADX_THRESHOLD = 30
RSI_MIN = 30
RSI_MAX = 70
VOLUME_MULTIPLIER = 2.5
PRICE_ABOVE_EMA20_MIN = 0.95
PRICE_ABOVE_EMA20_MAX = 1.10

# =============================================================================
# SHORT STRATEGY SETTINGS (from config/trading_config.py)
# =============================================================================

SHORT_ENABLED = False
LEADER_SHORT_ALLOWED_REGIMES = ("bull", "sideways")
SHORT_MAX_POSITIONS = 5
SHORT_MAX_EQUITY_PCT = 0.30
SHORT_RISK_PER_TRADE_PCT = 1.5
SHORT_REGIME_INDEX = "QQQ"
SHORT_REGIME_MA_PERIOD = 200
SHORT_REGIME_SLOPE_LOOKBACK = 20
SHORT_REGIME_ADX_PERIOD = 14
SHORT_REGIME_ADX_SMOOTH = 14
SHORT_REGIME_SIDEWAYS_ADX_MAX = 25

SHORT_REJECTION_MA = 50
SHORT_REJECTION_TOLERANCE = 0.02

SHORT_CFG_BULL = {}
SHORT_CFG_SIDEWAYS = {}
SHORT_CFG_BEAR = {}

# Leader Pullback Short Config (for bull/sideways markets)
LEADER_SHORT_CFG_BULL = {
    "ENABLED": True,
    "DEBUG_MODE": False,
    "PARTIAL_R": 2.0,
    "PARTIAL_SIZE": 0.5,
    "TRAIL_EMA": None,
    "TRAIL_DAYS": None,
    "EARLY_EXIT_DAYS": 20,
    "EARLY_EXIT_R_THRESHOLD": 0.0,
    "MAX_DAYS": 40,
    "MAX_POSITIONS": 10,
    "RISK_PER_TRADE_PCT": 0.35,
}

# =============================================================================
# REGIME CLASSIFICATION
# =============================================================================

REGIME_INDEX = "QQQ"
REGIME_BULL_MA = 200
REGIME_BEAR_MA = 200

# =============================================================================
# ADDITIONAL POSITION TRADING CONFIGS
# =============================================================================

RS_RANKER_SECTORS = ["Information Technology", "Communication Services", "Technology"]
RS_RANKER_TOP_N = 10
RS_RANKER_RS_THRESHOLD = 0.30
RS_RANKER_STOP_ATR_MULT = 2.0
RS_RANKER_MAX_DAYS = 120
RS_RANKER_PARTIAL_R = 2.5
RS_RANKER_PARTIAL_SIZE = 0.3

HIGH52_POS_RS_MIN = 0.30
HIGH52_POS_VOLUME_MULT = 1.5
HIGH52_POS_ADX_MIN = 25
HIGH52_POS_STOP_ATR_MULT = 2.0
HIGH52_POS_MAX_DAYS = 120
HIGH52_POS_PARTIAL_R = 2.5
HIGH52_POS_PARTIAL_SIZE = 0.3

BIGBASE_MIN_WEEKS = 8
BIGBASE_MAX_RANGE_PCT = 0.05
BIGBASE_RS_MIN = 0.30
BIGBASE_VOLUME_MULT = 1.5
BIGBASE_STOP_ATR_MULT = 2.0
BIGBASE_MAX_DAYS = 120
BIGBASE_PARTIAL_R = 3.0
BIGBASE_PARTIAL_SIZE = 0.3

PERCENT_B_POS_OVERSOLD = -0.5
PERCENT_B_POS_RSI_OVERSOLD = 40
PERCENT_B_POS_STOP_ATR_MULT = 3.0
PERCENT_B_POS_MAX_DAYS = 90

EMA_CROSS_POS_VOLUME_MULT = 1.5
EMA_CROSS_POS_STOP_ATR_MULT = 3.5
EMA_CROSS_POS_MAX_DAYS = 120

TREND_CONT_MA_LOOKBACK = 50
TREND_CONT_MA_RISING_DAYS = 20
TREND_CONT_RS_THRESHOLD = 0.20
TREND_CONT_RSI_MIN = 40
TREND_CONT_PULLBACK_EMA = 21
TREND_CONT_PULLBACK_ATR = 1.0
TREND_CONT_STOP_ATR_MULT = 2.5
TREND_CONT_MAX_DAYS = 120

MIN_PRICE = 5.0
MAX_PRICE = 999.0

TECH_SECTORS = ["Information Technology", "Communication Services"]


# =============================================================================
# GCS SETTINGS OVERRIDE
# Load config/settings.json from GCS and override any values defined above.
# This keeps sensitive strategy parameters out of the public repo.
# Local dev without GCS credentials runs on the defaults above.
# =============================================================================

def _apply_gcs_overrides():
    import json
    import tempfile
    import os
    from pathlib import Path

    def _apply(data: dict, source: str):
        g = globals()
        applied = [k for k in data if k in g]
        for key in applied:
            g[key] = data[key]
        if applied:
            print(f"⚙️  [settings] Loaded {len(applied)} override(s) from {source}")

    # 1. Local config/settings.json — used for local dev and backtesting
    local_file = Path(__file__).parent.parent.parent / "config" / "settings.json"
    if local_file.exists():
        try:
            with open(local_file) as f:
                _apply(json.load(f), f"local {local_file}")
            return  # local file takes full precedence; skip GCS
        except Exception as exc:
            print(f"⚠️  [settings] Could not load local settings: {exc}")

    # 2. GCS config/settings.json — used in GitHub Actions (production)
    try:
        from src.storage.gcs import download_file
        tmp = tempfile.mktemp(suffix=".json")
        if download_file("config/settings.json", tmp):
            with open(tmp) as f:
                _apply(json.load(f), "GCS config/settings.json")
            os.unlink(tmp)
    except Exception as exc:
        print(f"⚠️  [settings] Could not load GCS overrides: {exc}")


_apply_gcs_overrides()
