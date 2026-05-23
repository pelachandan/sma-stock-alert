import pandas as pd
import numpy as np
from src.data.market import get_historical_data
from src.data.indicators import compute_rsi, compute_ema_incremental
from src.config.settings import (
    ADX_THRESHOLD,
    RSI_MIN,
    RSI_MAX,
    VOLUME_MULTIPLIER,
    MIN_LIQUIDITY_USD,
    PRICE_ABOVE_EMA20_MIN,
    PRICE_ABOVE_EMA20_MAX,
    RISK_REWARD_RATIO
)

# -------------------------------------------------
# Strategy-Specific Stop Loss & Target Helpers
# -------------------------------------------------
def get_stop_loss(strategy: str, entry: float, atr: float) -> float:
    """
    Calculate stop loss based on strategy-specific ATR multipliers.
    Wider stops for strategies that need more room.
    """
    stops = {
        "EMA Crossover": 2.5,        # Widest - needs room after crossover
        "52-Week High": 2.0,         # Momentum breakout
        "Mean Reversion": 2.0,       # Mean reversion needs room
        "Consolidation Breakout": 2.0,
        "%B Mean Reversion": 2.0,    # Mean reversion
        "BB+RSI Combo": 2.0,         # Mean reversion
        "BB Squeeze": 2.0,           # Volatility breakout
    }
    mult = stops.get(strategy, 2.0)
    return entry - mult * atr


def get_target(strategy: str, entry: float, stop: float) -> float:
    """
    Calculate target based on strategy-specific risk/reward ratios.
    Mean reversion: 1.5R (quick bounces)
    Momentum: 2.0R (trend continuation)
    """
    targets = {
        "EMA Crossover": 2.0,        # Momentum trend
        "52-Week High": 2.0,         # Momentum breakout
        "Mean Reversion": 1.5,       # Quick bounce
        "Consolidation Breakout": 2.0,
        "%B Mean Reversion": 1.5,    # Quick bounce
        "BB+RSI Combo": 1.5,         # Quick bounce
        "BB Squeeze": 2.0,           # Momentum breakout
    }
    rr = targets.get(strategy, 2.0)
    return entry + rr * (entry - stop)


# -------------------------------------------------
# ADX
# -------------------------------------------------
def compute_adx(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = low.diff() * -1

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.rolling(period).mean()


# -------------------------------------------------
# ATR
# -------------------------------------------------
def calculate_atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr.iloc[-1] if not atr.empty else 0


# -------------------------------------------------
# Strategy Performance Metrics (Historical - for Van Tharp Expectancy)
# -------------------------------------------------
STRATEGY_METRICS = {
    # Format: (Win Rate, Avg Win R-Multiple, Avg Loss R-Multiple)
    # 🔧 UPDATED WITH IMPROVED ASSUMPTIONS (based on tighter filters + better exits)
    # These will be overwritten with actual backtest results

    # High Win Rate Strategies (tight filters, mean reversion)
    "BB+RSI Combo": (0.80, 1.50, -1.00),        # Triple confirmation, highest priority
    "Mean Reversion": (0.78, 1.50, -1.00),      # RSI(2) proven winner
    "%B Mean Reversion": (0.78, 1.50, -1.00),   # BB mean reversion variant

    # Moderate Win Rate Strategies (momentum/breakout)
    "52-Week High": (0.50, 2.00, -1.00),        # Tightened filters should improve WR
    "EMA Crossover": (0.45, 2.00, -1.00),       # Fixed detection + wider stops
    "Consolidation Breakout": (0.45, 2.00, -1.00),
    "BB Squeeze": (0.45, 2.00, -1.00),

    "Relative Strength": (0.30, 2.0, -1.0),     # Default values
    "RallyPattern_Position": (0.33, 2.0, -1.0),
    "EMA_StackAlignment_Position": (0.45, 2.0, -1.0),
}

# NOTE: These metrics come from actual backtest 2022-2026
# - Mean Reversion WORKS (75% WR as expected!)
# - Momentum strategies underperforming (30%, 22% WR)
# - Cascading BROKEN (14.6% vs expected 65%) - needs urgent fix

# NOTE: Negative R-multiples represent losses (e.g., -1.0R means losing 1× initial risk)
#       Positive R-multiples represent wins (e.g., 2.0R means gaining 2× initial risk)

# -------------------------------------------------
# Van Tharp Expectancy Scoring Algorithm
# -------------------------------------------------
def normalize_score(score, strategy):
    """
    VAN THARP EXPECTANCY SCORING SYSTEM

    Van Tharp's Expectancy Formula:
    Expectancy = (WinRate × AvgWin) - ((1 - WinRate) × AvgLoss)

    This accounts for the asymmetry between wins and losses:
    - Not all wins are equal (some 0.5R, some 3R)
    - Not all losses are equal (some -0.5R, some -2R)
    - Expectancy represents the average R you can expect per trade

    Algorithm:
    1. Normalize raw score to 0-1 (quality within strategy)
    2. Calculate Van Tharp Expectancy for strategy
    3. FinalScore = Quality × Expectancy
    4. Multiply by 10 for readability (0-13 scale)

    Example:
    - Strategy: 60% WR, +2R avg win, -1R avg loss
    - Expectancy = (0.60 × 2.0) - (0.40 × 1.0) = 1.2 - 0.4 = 0.8R per trade
    - Signal quality: 0.8 (80% of max)
    - FinalScore = 0.8 × 0.8 × 10 = 6.4
    """

    # Step 1: Normalize raw score to 0-1 (quality within strategy)
    ranges = {
        "EMA Crossover": (50, 100),           # Base: 75 pts + Crossover bonus: 25 pts
        "52-Week High": (6, 12),              # Simple scoring
        "Consolidation Breakout": (4, 10),    # Range + volume based
        "BB Squeeze": (50, 100),              # Squeeze + breakout + volume
        "Mean Reversion": (40, 100),          # RSI(2) based: Max 100 pts
        "%B Mean Reversion": (40, 100),       # %B based: Max 100 pts
        "BB+RSI Combo": (50, 100),            # Double confirmation: Max 100 pts
        "Relative Strength": (5, 15),
        "RallyPattern_Position": (45, 100),
        "EMA_StackAlignment_Position": (50, 100),
    }

    low, high = ranges.get(strategy, (0, 20))
    quality = (score - low) / (high - low)
    quality = max(0, min(1, quality))  # Clamp to [0, 1]

    # Step 2: Calculate Van Tharp Expectancy for this strategy
    win_rate, avg_win_r, avg_loss_r = STRATEGY_METRICS.get(strategy, (0.30, 1.5, -1.0))

    # Van Tharp's Expectancy = (WinRate × AvgWin) - ((1 - WinRate) × |AvgLoss|)
    # Note: avg_loss_r is already negative, so we use abs() for clarity
    expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * abs(avg_loss_r))

    # Step 3: Calculate final score (quality × expectancy)
    # Scale by 10 for readability (0-13 range instead of 0-1.3)
    final_score = quality * expectancy * 10

    return round(final_score, 2)


# -------------------------------------------------
# Pre-Buy Check with Market Regime Filter
# -------------------------------------------------
def pre_buy_check(combined_signals, rr_ratio=None, benchmark="SPY", as_of_date=None):
    """
    Deduplicates signals, applies liquidity + trend filters,
    computes ATR-based stops, normalizes scores,
    and blocks breakout trades in bearish market regime.

    Args:
        combined_signals: List of signal dictionaries
        rr_ratio: Risk/reward ratio for target calculation
        benchmark: Benchmark ticker for regime (not used, kept for compatibility)
        as_of_date: Optional date for backtesting. If None, uses latest data (live mode).
                    If provided, filters data to only use information up to this date.
    """

    # Use config value if rr_ratio not provided
    if rr_ratio is None:
        rr_ratio = RISK_REWARD_RATIO

    # Market regime must be supplied by scanner (walk-forward safe)
    is_bullish = True
    mode = "BACKTEST" if as_of_date else "LIVE"
    print(f"📊 Mode: {mode} | Market regime ({benchmark}): {'BULLISH' if is_bullish else 'BEARISH'}")

    # -------------------------------
    # Deduplicate by strategy priority
    # -------------------------------
    # Higher number = higher priority (if same ticker has multiple signals, use highest priority)
    # Updated for position trading strategies
    priority = {
        # Position Trading Strategies (NEW)
        "BigBase_Breakout_Position": 7,           # Highest - rarest, biggest moves
        "RelativeStrength_Ranker_Position": 6,    # Proven workhorse
        "RallyPattern_Position": 6,               # Daily rally-pattern leaders
        "EMA_StackAlignment_Position": 5,         # Bullish EMA stack alignment
        "High52_Position": 5,                     # Momentum breakout
        "EMA_Crossover_Position": 4,
        "TrendContinuation_Position": 3,
        "%B_MeanReversion_Position": 2,
        "GapReversal_Position": 8,                # Highest - rare extreme RSI + gap setup
        "ShortWeakRS_Retrace_Position": 0,        # Shorts LOSE to longs in deduplication (priority 0)

        # Legacy Short-Term Strategies (OLD - for backward compatibility)
        "BB+RSI Combo": 7,           # Triple confirmation
        "Mean Reversion": 6,         # RSI(2) proven winner
        "%B Mean Reversion": 5,      # BB mean reversion variant
        "52-Week High": 4,           # Momentum breakout
        "EMA Crossover": 3,          # Trend following
        "Consolidation Breakout": 2,
        "BB Squeeze": 1,             # Lowest
        "Relative Strength": 1,
    }

    best_signal = {}
    for s in combined_signals:
        t = s["Ticker"]
        strategy = s["Strategy"]
        # Use .get() with default to avoid KeyError if strategy not in priority dict
        if t not in best_signal or priority.get(strategy, 0) > priority.get(best_signal[t]["Strategy"], 0):
            best_signal[t] = s

    signals = list(best_signal.values())
    trades = []

    for s in signals:
        ticker = s["Ticker"]
        strategy = s["Strategy"]

        # -------------------------------
        # Skip breakout trades in bearish regime
        # -------------------------------
        market_regime = s.get("MarketRegime", "BULLISH")

        if market_regime == "BEARISH" and strategy in [
            "52-Week High",
            "Consolidation Breakout"
        ]:
            print(f"   ❌ {ticker} [{strategy}]: filtered — bearish regime")
            continue

        df = get_historical_data(ticker)
        if df.empty:
            print(f"   ❌ {ticker} [{strategy}]: filtered — no historical data")
            continue

        # Ensure index is datetime (safety check for string indices)
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
                df = df[df.index.notna()]
            except:
                continue

        # 🔒 CRITICAL: Filter to as_of_date for backtesting (prevents look-ahead bias)
        if as_of_date is not None:
            df = df[df.index <= as_of_date]

        if len(df) < 60:
            print(f"   ❌ {ticker} [{strategy}]: filtered — insufficient history ({len(df)} bars)")
            continue

        df = df.tail(60)
        close = df["Close"].iloc[-1]

        # -------------------------------
        # Liquidity filter (from config)
        # -------------------------------
        avg_dollar_vol = (df["Close"] * df["Volume"]).rolling(20).mean().iloc[-1]
        if avg_dollar_vol < MIN_LIQUIDITY_USD:
            print(f"   ❌ {ticker} [{strategy}]: filtered — low liquidity (${avg_dollar_vol/1e6:.1f}M < ${MIN_LIQUIDITY_USD/1e6:.0f}M)")
            continue

        # -------------------------------
        # ATR-based risk
        # -------------------------------
        atr = calculate_atr(df)
        if atr == 0:
            atr = close * 0.02

        # Use scanner's price if available (more consistent than re-fetching)
        entry = s.get("Price") or s.get("Entry") or close

        # 🔧 Use stop/target from scanner — supports both field naming conventions:
        #   - Position trading strategies: StopLoss / Target
        #   - Long position strategies:    StopPrice (no Target — calculate 2R)
        #   - Legacy short-term:           fall back to ATR-based helpers
        stop_val  = s.get("StopLoss") or s.get("StopPrice")
        target_val = s.get("Target")

        if stop_val is not None and not pd.isna(stop_val) and stop_val > 0:
            stop = stop_val
            if target_val is not None and not pd.isna(target_val):
                target = target_val
            else:
                risk = s.get("RiskPerShare")
                if risk is None or pd.isna(risk) or risk <= 0:
                    risk = max(abs(entry - stop), entry * 0.01)
                target = entry + 2.0 * risk
        else:
            stop = get_stop_loss(strategy, entry, atr)
            target = get_target(strategy, entry, stop)

        # -------------------------------
        # EMA strategy extra filters (NOW DONE IN SCANNER - kept for other strategies)
        # -------------------------------
        if strategy == "EMA Crossover":
            if not s.get("ADX14") or s.get("ADX14") < ADX_THRESHOLD:
                print(f"   ❌ {ticker} [{strategy}]: filtered — ADX too low")
                continue

        # Calculate final score using Van Tharp Expectancy
        final_score = normalize_score(s.get("Score", 0), strategy)

        # Get strategy metrics for display
        win_rate, avg_win_r, avg_loss_r = STRATEGY_METRICS.get(strategy, (0.30, 1.5, -1.0))

        # Calculate Van Tharp Expectancy
        expectancy = (win_rate * avg_win_r) - ((1 - win_rate) * abs(avg_loss_r))

        # Validate stop and target are valid numbers (must be positive prices, not NaN)
        if pd.isna(stop) or pd.isna(target) or stop <= 0 or target <= 0 or entry <= 0:
            print(f"   ❌ {ticker} [{strategy}]: filtered — invalid stop/target (entry={entry}, stop={stop}, target={target})")
            continue

        # For SHORT: stop must be above entry (if price rises, we lose)
        # For LONG: stop must be below entry (if price falls, we lose)
        direction = s.get("Direction", "LONG")
        if direction == "SHORT" and stop <= entry:
            print(f"   ❌ {ticker} [{strategy}]: filtered — SHORT stop ${stop:.2f} not above entry ${entry:.2f}")
            continue
        elif direction == "LONG" and stop >= entry:
            print(f"   ❌ {ticker} [{strategy}]: filtered — LONG stop ${stop:.2f} not below entry ${entry:.2f}")
            continue

        print(f"   ✅ {ticker} [{strategy}]: passed pre_buy_check (entry=${entry:.2f}, stop=${stop:.2f}, target=${target:.2f})")

        trades.append({
            "Ticker": ticker,
            "Strategy": strategy,
            "Entry": round(entry, 2),
            "StopLoss": round(stop, 2),
            "Target": round(target, 2),
            "Score": s.get("Score", 0),
            "FinalScore": final_score,
            "Expectancy": round(expectancy, 2),
            "CrossoverType": s.get("CrossoverType", "Unknown"),
            "CrossoverBonus": s.get("CrossoverBonus", 0),
            "Direction": s.get("Direction", "LONG"),
            "Priority": s.get("Priority"),           # Preserve Priority for email position-trading detection
            "MaxDays": s.get("MaxDays"),
            "GapPct": s.get("GapPct"),               # GapReversal: gap size % (for trade log)
            "SmoothedRSI": s.get("SmoothedRSI"),     # GapReversal: smoothed RSI at entry (for trade log)
            "GapFillLevel": s.get("GapFillLevel"),   # GapReversal: gap fill stop level
            "GapHigh": s.get("GapHigh"),
            "GapLow": s.get("GapLow"),
            "GapMid": s.get("GapMid"),
            "GapSupport": s.get("GapSupport"),
            "GapResistance": s.get("GapResistance"),
            "ZoneSupport": s.get("ZoneSupport"),
            "ZoneResistance": s.get("ZoneResistance"),
            "RiskPerShare": round(float(s.get("RiskPerShare")), 2) if s.get("RiskPerShare") is not None else round(max(abs(entry - stop), entry * 0.01), 2),
            "RoomToResistancePct": s.get("RoomToResistancePct"),
            "RoomToSupportPct": s.get("RoomToSupportPct"),
            "SetupType": s.get("SetupType"),
            "SignalType": s.get("SignalType"),
            "LeadershipStage": s.get("LeadershipStage"),
            "PositionSizeMultiplier": s.get("PositionSizeMultiplier"),
            "EntryScore": s.get("EntryScore", s.get("Score", 0)),
        })

    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        # Sort by FinalScore (incorporates both quality and profitability)
        df_trades = df_trades.sort_values(by="FinalScore", ascending=False)

    return df_trades
