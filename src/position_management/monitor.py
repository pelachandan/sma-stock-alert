"""
Position Monitor for Live Trading
===================================
Monitors open positions daily and checks for exit conditions.
Matches the backtester's hybrid trail exit logic.

Exit Conditions Checked:
1. Stop Loss (hard stop)
2. Partial Profit (30% at +2.5R / +3.0R)
3. Hybrid Trail System:
   - Days 1-60: EMA21 trail (5 consecutive closes)
   - Days 61+: MA100 trail (8 consecutive closes)
4. Time Stops (150/180 days)
5. Pyramid Opportunities (+1.5R + EMA21 pullback)
"""

import pandas as pd
from datetime import datetime, timedelta
from src.data.market import get_historical_data
from src.data.indicators import compute_rsi
from src.config.settings import (
    RS_RANKER_STOP_ATR_MULT,
    HIGH52_POS_STOP_ATR_MULT,
    BIGBASE_STOP_ATR_MULT,
    RS_RANKER_PARTIAL_R,
    HIGH52_POS_PARTIAL_R,
    BIGBASE_PARTIAL_R,
    RS_RANKER_MAX_DAYS,
    HIGH52_POS_MAX_DAYS,
    BIGBASE_MAX_DAYS,
    GAP_REVERSAL_MAX_DAYS,
    GAP_REVERSAL_TRAIL_MA,
    GAP_REVERSAL_TARGET_R_MULTIPLE,
    GAP_CONTINUATION_MAX_DAYS,
    POSITION_PARTIAL_SIZE,
    POSITION_PYRAMID_R_TRIGGER,
    POSITION_PYRAMID_MAX_ADDS,
    POSITION_PYRAMID_PULLBACK_ATR,
)


def calculate_atr(df, period=20):
    """Calculate ATR(20) for position sizing."""
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"] - df["Close"].shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.iloc[-1] if not atr.empty else 0


def monitor_positions(position_tracker):
    """
    Monitor all open positions and check for exit/action signals.

    Args:
        position_tracker: PositionTracker instance with open positions

    Returns:
        dict: {
            'exits': [list of exit signals],
            'partials': [list of partial profit signals],
            'pyramids': [list of pyramid opportunities],
            'warnings': [list of warning signals]
        }
    """
    positions = position_tracker.get_all_positions()

    if not positions:
        return {'exits': [], 'partials': [], 'pyramids': [], 'warnings': []}

    exits = []
    partials = []
    pyramids = []
    warnings = []

    today = pd.Timestamp.today()

    for ticker, pos in positions.items():
        try:
            # Get current data
            df = get_historical_data(ticker)
            if df.empty or len(df) < 100:
                warnings.append({
                    'ticker': ticker,
                    'type': 'DATA_ERROR',
                    'message': f'Unable to fetch data for {ticker}'
                })
                continue

            # Current price and indicators
            current_close = df['Close'].iloc[-1]
            current_high = df['High'].iloc[-1]
            current_low = df['Low'].iloc[-1]

            # Calculate indicators
            df['EMA21'] = df['Close'].ewm(span=21).mean()
            df['MA100'] = df['Close'].rolling(100).mean()
            df['MA200'] = df['Close'].rolling(200).mean()

            ema21 = df['EMA21'].iloc[-1] if len(df) >= 21 else None
            ma100 = df['MA100'].iloc[-1] if len(df) >= 100 else None
            ma200 = df['MA200'].iloc[-1] if len(df) >= 200 else None

            atr = calculate_atr(df, 20)

            # Position details
            entry_price = pos['entry_price']
            entry_date = pd.to_datetime(pos['entry_date'])
            strategy = pos.get('strategy', 'Unknown')
            stop_loss = pos.get('stop_loss', 0)
            direction = pos.get('direction', 'LONG')
            days_held = (today - entry_date).days

            # Track consecutive closes below trail
            closes_below_trail = pos.get('closes_below_trail', 0)
            partial_exited = pos.get('partial_exited', False)
            pyramids_added = pos.get('pyramids_added', 0)

            if strategy == "Streak_Position":
                latest_bar_date = pd.Timestamp(df.index[-1]).normalize()
                if latest_bar_date >= entry_date.normalize():
                    if direction == "SHORT":
                        streak_r = (entry_price - current_close) / max(entry_price * 0.01, 0.01)
                    else:
                        streak_r = (current_close - entry_price) / max(entry_price * 0.01, 0.01)
                    exits.append({
                        'ticker': ticker,
                        'type': 'NEXT_SESSION_CLOSE',
                        'reason': 'Streak strategy exits at the next trading session close',
                        'action': f'EXIT ALL at ${current_close:.2f}',
                        'current_r': streak_r,
                        'days_held': days_held,
                        'urgency': 'HIGH',
                        'entry_price': entry_price,
                        'current_price': current_close
                    })
                continue

            # Calculate current R-multiple (direction-aware)
            # Apply 1% floor to match position sizing logic and prevent extreme R from tiny stops
            min_risk = entry_price * 0.01 if entry_price > 0 else 0.01
            if direction == "SHORT":
                # For SHORT: stop > entry, risk = stop - entry
                raw_risk = (stop_loss - entry_price) if stop_loss > 0 else entry_price * 0.02
                risk_amount = max(raw_risk, min_risk)
                current_r = (entry_price - current_close) / risk_amount
            else:
                # For LONG: entry > stop, risk = entry - stop
                raw_risk = (entry_price - stop_loss) if stop_loss > 0 else entry_price * 0.02
                risk_amount = max(raw_risk, min_risk)
                current_r = (current_close - entry_price) / risk_amount

            # Get strategy-specific parameters
            if strategy == "RelativeStrength_Ranker_Position":
                partial_r_trigger = RS_RANKER_PARTIAL_R
                max_days = RS_RANKER_MAX_DAYS
            elif strategy == "High52_Position":
                partial_r_trigger = HIGH52_POS_PARTIAL_R
                max_days = HIGH52_POS_MAX_DAYS
            elif strategy == "BigBase_Breakout_Position":
                partial_r_trigger = BIGBASE_PARTIAL_R
                max_days = BIGBASE_MAX_DAYS
            elif strategy == "GapReversal_Position":
                partial_r_trigger = 999  # no partial exits for gap reversal
                max_days = GAP_REVERSAL_MAX_DAYS
            elif strategy == "GapContinuation_Position":
                partial_r_trigger = 999  # continuation runs full-position exits
                max_days = GAP_CONTINUATION_MAX_DAYS
            elif strategy == "RallyPattern_Position":
                partial_r_trigger = 999  # rally strategy currently runs full-position exits
                max_days = int(pos.get('max_days', 120))
            elif strategy in {"EMA_Crossover_Position", "EMA_StackAlignment_Position"}:
                partial_r_trigger = 2.0
                max_days = int(pos.get('max_days', 120))
            else:
                partial_r_trigger = 2.5
                max_days = 150

            # =====================================================
            # 1. CHECK STOP LOSS (HARD EXIT)
            # =====================================================
            # Direction-aware: LONG uses Low (gap-fill going down), SHORT uses High (gap-fill rally)
            stop_hit = (
                (direction == "LONG" and stop_loss > 0 and current_low <= stop_loss) or
                (direction == "SHORT" and stop_loss > 0 and current_high >= stop_loss)
            )
            if stop_hit:
                exits.append({
                    'ticker': ticker,
                    'type': 'STOP_LOSS',
                    'reason': f'Stop loss hit at ${stop_loss:.2f}',
                    'action': f'EXIT ALL at market (current: ${current_close:.2f})',
                    'current_r': -1.0,
                    'days_held': days_held,
                    'urgency': 'IMMEDIATE',
                    'entry_price': entry_price,
                    'current_price': current_close
                })
                continue

            # =====================================================
            # 2. CHECK PARTIAL PROFIT
            # =====================================================
            if not partial_exited and current_r >= partial_r_trigger:
                partials.append({
                    'ticker': ticker,
                    'type': 'PARTIAL_PROFIT',
                    'reason': f'Hit +{partial_r_trigger}R profit target',
                    'action': f'EXIT {int(POSITION_PARTIAL_SIZE*100)}% at ${current_close:.2f}, keep 70% runner',
                    'current_r': current_r,
                    'days_held': days_held,
                    'urgency': 'HIGH',
                    'entry_price': entry_price,
                    'current_price': current_close
                })
                # Mark for next scan
                pos['partial_exited'] = True
                position_tracker._save_positions()

            # =====================================================
            # 3. CHECK HYBRID TRAIL EXITS
            # =====================================================
            trail_triggered = False

            if strategy in ["High52_Position", "RelativeStrength_Ranker_Position"]:
                if days_held <= 60:
                    # First 60 days: EMA21 trail (5 consecutive closes)
                    if ema21 and pd.notna(ema21):
                        if current_close < ema21:
                            closes_below_trail += 1
                            if closes_below_trail >= 5:
                                exits.append({
                                    'ticker': ticker,
                                    'type': 'EMA21_TRAIL_EARLY',
                                    'reason': f'5 closes below EMA21 (${ema21:.2f})',
                                    'action': f'EXIT RUNNER at ${current_close:.2f}',
                                    'current_r': current_r,
                                    'days_held': days_held,
                                    'urgency': 'HIGH',
                                    'entry_price': entry_price,
                                    'current_price': current_close
                                })
                                trail_triggered = True
                        else:
                            closes_below_trail = 0
                else:
                    # After 60 days: MA100 trail (8 consecutive closes)
                    if ma100 and pd.notna(ma100):
                        if current_close < ma100:
                            closes_below_trail += 1
                            if closes_below_trail >= 8:
                                exits.append({
                                    'ticker': ticker,
                                    'type': 'MA100_TRAIL_LATE',
                                    'reason': f'8 closes below MA100 (${ma100:.2f})',
                                    'action': f'EXIT RUNNER at ${current_close:.2f}',
                                    'current_r': current_r,
                                    'days_held': days_held,
                                    'urgency': 'MEDIUM',
                                    'entry_price': entry_price,
                                    'current_price': current_close
                                })
                                trail_triggered = True
                        else:
                            closes_below_trail = 0

            elif strategy == "BigBase_Breakout_Position":
                # BigBase: MA200 trail (10 consecutive closes)
                if ma200 and pd.notna(ma200):
                    if current_close < ma200:
                        closes_below_trail += 1
                        if closes_below_trail >= 10:
                            exits.append({
                                'ticker': ticker,
                                'type': 'MA200_TRAIL',
                                'reason': f'10 closes below MA200 (${ma200:.2f})',
                                'action': f'EXIT RUNNER at ${current_close:.2f}',
                                'current_r': current_r,
                                'days_held': days_held,
                                'urgency': 'MEDIUM',
                                'entry_price': entry_price,
                                'current_price': current_close
                            })
                            trail_triggered = True
                    else:
                        closes_below_trail = 0

            elif strategy == "GapReversal_Position":
                try:
                    from src.strategies.gap_reversal import GapReversalPosition

                    reversal_strategy = GapReversalPosition()
                    position_with_ticker = dict(pos)
                    position_with_ticker["ticker"] = ticker
                    exit_cond = reversal_strategy.get_exit_conditions(position_with_ticker, df, today)
                    if exit_cond is not None:
                        exit_price = float(exit_cond.get("exit_price", current_close))
                        exits.append({
                            'ticker': ticker,
                            'type': str(exit_cond["reason"]).upper(),
                            'reason': f'Gap reversal exit: {exit_cond["reason"]}',
                            'action': f'EXIT ALL at ${exit_price:.2f}',
                            'current_r': current_r,
                            'days_held': days_held,
                            'urgency': 'HIGH',
                            'entry_price': entry_price,
                            'current_price': current_close
                        })
                        trail_triggered = True
                except Exception:
                    pass

            elif strategy == "RallyPattern_Position":
                try:
                    from src.strategies.rally_pattern import RallyPatternPosition

                    rally_strategy = RallyPatternPosition()
                    position_with_ticker = dict(pos)
                    position_with_ticker["ticker"] = ticker
                    exit_cond = rally_strategy.get_exit_conditions(position_with_ticker, df, today)
                    if exit_cond is not None:
                        exits.append({
                            'ticker': ticker,
                            'type': str(exit_cond["reason"]).upper(),
                            'reason': f'Rally pattern exit: {exit_cond["reason"]}',
                            'action': f'EXIT ALL at ${current_close:.2f}',
                            'current_r': current_r,
                            'days_held': days_held,
                            'urgency': 'HIGH',
                            'entry_price': entry_price,
                            'current_price': current_close
                        })
                        trail_triggered = True
                except Exception:
                    pass
            elif strategy in {"EMA_Crossover_Position", "EMA_StackAlignment_Position"}:
                if ma100 and pd.notna(ma100):
                    if current_close < ma100:
                        closes_below_trail += 1
                        if closes_below_trail >= 5:
                            exits.append({
                                'ticker': ticker,
                                'type': 'MA100_TRAIL',
                                'reason': f'5 closes below MA100 (${ma100:.2f})',
                                'action': f'EXIT RUNNER at ${current_close:.2f}',
                                'current_r': current_r,
                                'days_held': days_held,
                                'urgency': 'HIGH',
                                'entry_price': entry_price,
                                'current_price': current_close
                            })
                            trail_triggered = True
                    else:
                        closes_below_trail = 0
            elif strategy == "GapContinuation_Position":
                try:
                    from src.strategies.gap_continuation import GapContinuationPosition

                    continuation_strategy = GapContinuationPosition()
                    position_with_ticker = dict(pos)
                    position_with_ticker["ticker"] = ticker
                    exit_cond = continuation_strategy.get_exit_conditions(position_with_ticker, df, today)
                    if exit_cond is not None:
                        exits.append({
                            'ticker': ticker,
                            'type': str(exit_cond["reason"]).upper(),
                            'reason': f'Gap continuation exit: {exit_cond["reason"]}',
                            'action': f'EXIT ALL at ${float(exit_cond.get("exit_price", current_close)):.2f}',
                            'current_r': current_r,
                            'days_held': days_held,
                            'urgency': 'HIGH',
                            'entry_price': entry_price,
                            'current_price': current_close
                        })
                        trail_triggered = True
                except Exception:
                    pass

            # Update trail counter
            if not trail_triggered:
                pos['closes_below_trail'] = closes_below_trail
                position_tracker._save_positions()

            # =====================================================
            # 4. CHECK TIME STOP
            # =====================================================
            # GapReversal: always hard-cap at MaxDays (no pyramid exception —
            # the PLTR trade ran 1134 days because this check was skipped).
            # Other strategies: skip time stop if position was pyramided (proven winner).
            pyramid_adds = pos.get('pyramid_adds', 0)
            pyramid_count = len(pyramid_adds) if isinstance(pyramid_adds, list) else pyramid_adds
            has_pyramids = pyramid_count > 0

            time_stop_due = (strategy in {"GapReversal_Position", "GapContinuation_Position"} and days_held >= max_days) or \
                            (not has_pyramids and days_held >= max_days)

            if time_stop_due:
                exits.append({
                    'ticker': ticker,
                    'type': f'TIME_STOP_{max_days}d',
                    'reason': f'Held for {days_held} days (max: {max_days})',
                    'action': f'EXIT ALL at ${current_close:.2f}',
                    'current_r': current_r,
                    'days_held': days_held,
                    'urgency': 'MEDIUM',
                    'entry_price': entry_price,
                    'current_price': current_close
                })
                continue

            # =====================================================
            # 5. CHECK PYRAMID OPPORTUNITY
            # =====================================================
            if current_r >= POSITION_PYRAMID_R_TRIGGER and pyramids_added < POSITION_PYRAMID_MAX_ADDS:
                # Check if price pulled back to EMA21
                if ema21 and pd.notna(ema21):
                    distance_to_ema21 = abs(current_close - ema21)
                    if distance_to_ema21 <= (POSITION_PYRAMID_PULLBACK_ATR * atr):
                        pyramids.append({
                            'ticker': ticker,
                            'type': 'PYRAMID',
                            'reason': f'At +{current_r:.2f}R, pulled back to EMA21',
                            'action': f'ADD 50% position at ${current_close:.2f} (pyramid #{pyramids_added + 1})',
                            'current_r': current_r,
                            'days_held': days_held,
                            'urgency': 'LOW',
                            'entry_price': entry_price,
                            'current_price': current_close,
                            'pyramid_num': pyramids_added + 1
                        })

            # =====================================================
            # 6. WARNING SIGNALS (NOT EXITS, JUST FYI)
            # =====================================================
            # GapReversal: daily status — show EMA21 trail level and 2R progress
            if strategy == "GapReversal_Position" and ema21 and pd.notna(ema21):
                initial_target = pos.get('target', 0) or 0
                r_target = GAP_REVERSAL_TARGET_R_MULTIPLE

                if initial_target > 0 and current_r >= r_target:
                    # At or beyond initial R target — trail now protects profit
                    warnings.append({
                        'ticker': ticker,
                        'type': 'GAP_TARGET_REACHED',
                        'message': (
                            f'{ticker} ✅ {r_target}R target reached '
                            f'(+{current_r:.1f}R, ${current_close:.2f}) | '
                            f'EMA{GAP_REVERSAL_TRAIL_MA} trail: ${ema21:.2f} — '
                            f'hold until close < ${ema21:.2f}'
                        ),
                    })
                else:
                    # Still working toward initial target — show trail level for awareness
                    warnings.append({
                        'ticker': ticker,
                        'type': 'GAP_TRAIL_STATUS',
                        'message': (
                            f'{ticker} GapReversal +{current_r:.1f}R (${current_close:.2f}) | '
                            f'EMA{GAP_REVERSAL_TRAIL_MA} trail: ${ema21:.2f} | '
                            f'{r_target}R target: ${initial_target:.2f}'
                        ),
                    })

            # Approaching EMA21/MA100 (other strategies)
            elif strategy != "GapReversal_Position":
                if days_held <= 60 and ema21 and pd.notna(ema21):
                    distance_pct = ((current_close - ema21) / ema21) * 100
                    if 0 < distance_pct < 2:  # Within 2% above EMA21
                        warnings.append({
                            'ticker': ticker,
                            'type': 'APPROACHING_EMA21',
                            'message': f'{ticker} approaching EMA21 (${ema21:.2f}, current: ${current_close:.2f})',
                            'closes_below': closes_below_trail
                        })
                elif days_held > 60 and ma100 and pd.notna(ma100):
                    distance_pct = ((current_close - ma100) / ma100) * 100
                    if 0 < distance_pct < 3:  # Within 3% above MA100
                        warnings.append({
                            'ticker': ticker,
                            'type': 'APPROACHING_MA100',
                            'message': f'{ticker} approaching MA100 (${ma100:.2f}, current: ${current_close:.2f})',
                            'closes_below': closes_below_trail
                        })

        except Exception as e:
            warnings.append({
                'ticker': ticker,
                'type': 'MONITORING_ERROR',
                'message': f'Error monitoring {ticker}: {str(e)}'
            })

    return {
        'exits': exits,
        'partials': partials,
        'pyramids': pyramids,
        'warnings': warnings
    }
