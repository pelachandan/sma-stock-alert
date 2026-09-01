"""
Live Position Trading Scanner
==============================
Uses the same position trading strategies as the backtester.
Scans for active long strategies including:
- RelativeStrength_Ranker_Position
- RallyPattern_Position
- High52_Position
- BigBase_Breakout_Position
"""

import argparse
import pandas as pd
from datetime import datetime
from src.scanning.scanner import run_scan_as_of
from src.scanning.validator import pre_buy_check
from src.scanning.rs_bought_tracker import RSBoughtTracker, StrategyStateTracker
from src.notifications.email import send_email_alert
from src.position_management.tracker import PositionTracker, filter_trades_by_position
from src.position_management.monitor import monitor_positions
from src.data.market import get_historical_data
from src.data.universe import get_sp500_tickers
from src.analysis.market_regime import get_position_regime, get_regime_params, PositionRegime
from src.config.settings import (
    POSITION_MAX_TOTAL,
    POSITION_MAX_PER_STRATEGY,
    POSITION_RISK_PER_TRADE_PCT,
    POSITION_INITIAL_EQUITY,
    REGIME_INDEX,
)

# Position tracker for live trading (persistent file)
position_tracker = PositionTracker(mode="live", file="data/open_positions.json")

# RS Ranker bought tracker for live trading (persistent file)
rs_bought_tracker = RSBoughtTracker(file_path="data/rs_ranker_bought.json")
strategy_trackers: dict[str, StrategyStateTracker] = {
    "RelativeStrength_Ranker_Position": rs_bought_tracker,
}


def get_strategy_tracker(strategy_name: str) -> StrategyStateTracker:
    """Return the persistent per-strategy tracker used for live state/history."""
    tracker = strategy_trackers.get(strategy_name)
    if tracker is None:
        tracker = StrategyStateTracker(strategy_name=strategy_name)
        strategy_trackers[strategy_name] = tracker
    return tracker


def check_market_regime():
    """
    Check market regime using the SAME logic as the backtester.
    Returns (regime, regime_params, allow_new_entries).

    3 states (matching backtester exactly):
      RISK_ON  : QQQ > MA200 AND MA200 rising  → full aggression
      NEUTRAL  : everything else (transitioning)→ cautious, entries allowed
      RISK_OFF : QQQ < MA200 AND MA200 declining→ defensive, NO new entries
    """
    today = pd.Timestamp.today()
    try:
        regime = get_position_regime(as_of_date=today, index_symbol=REGIME_INDEX)
    except Exception:
        print("⚠️ Unable to determine market regime, defaulting to NEUTRAL.")
        regime = PositionRegime.NEUTRAL

    params = get_regime_params(regime)

    # Display (mirrors backtester console output)
    df = get_historical_data(REGIME_INDEX)
    if not df.empty and len(df) >= 200:
        close = df["Close"].iloc[-1]
        ma200 = df["Close"].rolling(200).mean().iloc[-1]
        ma200_20d = df["Close"].rolling(200).mean().iloc[-21] if len(df) >= 221 else ma200
        ma_rising = ma200 > ma200_20d
        regime_label = {
            PositionRegime.RISK_ON:  "🟢 RISK_ON  (QQQ > MA200, MA rising  — full entries)",
            PositionRegime.NEUTRAL:  "🟡 NEUTRAL  (transitioning — cautious entries allowed)",
            PositionRegime.RISK_OFF: "🔴 RISK_OFF (QQQ < MA200, MA declining — NO new entries)",
        }[regime]
        print(f"📊 Market Regime: {regime_label}")
        print(f"   {REGIME_INDEX}: ${close:.2f} | MA200: ${ma200:.2f} | MA Rising: {ma_rising}")
    else:
        print(f"📊 Market Regime: {regime.value}")

    return regime, params, params.get("allow_new_entries", True)


def filter_trades_by_same_day_exits(trades_df: pd.DataFrame, exited_tickers: set[str]) -> pd.DataFrame:
    """Block same-run re-entry for tickers that already generated exit actions."""
    if trades_df.empty or not exited_tickers:
        return trades_df

    keep_mask = ~trades_df["Ticker"].isin(sorted(exited_tickers))
    filtered_df = trades_df[keep_mask].copy()
    skipped = len(trades_df) - len(filtered_df)
    if skipped > 0:
        skipped_tickers = trades_df.loc[~keep_mask, "Ticker"].tolist()
        print(f"   🚫 Skipped {skipped} trade(s) due to same-day exits: {', '.join(skipped_tickers[:5])}")
        if len(skipped_tickers) > 5:
            print(f"      ... and {len(skipped_tickers) - 5} more")
    return filtered_df


def is_streak_option_alert(trade: pd.Series | dict) -> bool:
    """Identify the ranker's manual option guidance, which is never equity state."""
    return trade.get("Strategy") == "Streak_Position"


def split_streak_option_alerts(trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate manual option guidance from equity candidates before tracker handling."""
    if trades_df.empty:
        return trades_df.copy(), trades_df.copy()
    option_alerts = trades_df[trades_df.apply(is_streak_option_alert, axis=1)].copy()
    equity_trades = trades_df[~trades_df.apply(is_streak_option_alert, axis=1)].copy()
    return equity_trades, option_alerts


if __name__ == "__main__":
    # --------------------------------------------------
    # CLI Arguments
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description="Live Position Trading Scanner")
    parser.add_argument(
        "--strategies",
        help="Comma-separated list of strategies to scan (e.g. GapReversal_Position). "
             "If set, only these strategies run for new entries.",
    )
    parser.add_argument(
        "--skip-strategies",
        dest="skip_strategies",
        help="Comma-separated list of strategies to exclude from new entries "
             "(e.g. GapReversal_Position). Position monitoring always runs for all.",
    )
    parser.add_argument(
        "--skip-monitor",
        dest="skip_monitor",
        action="store_true",
        default=False,
        help="Skip position exit/pyramid monitoring (use for morning scans where "
             "the day's close is not yet available).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label appended to the email subject (e.g. 'Morning Gap Scan').",
    )
    args = parser.parse_args()

    # Apply strategy filters by modifying POSITION_MAX_PER_STRATEGY in-place.
    # scanner.py imported the same dict object so this propagates automatically.
    if args.strategies:
        allowed = {s.strip() for s in args.strategies.split(",")}
        for key in list(POSITION_MAX_PER_STRATEGY.keys()):
            if key not in allowed:
                POSITION_MAX_PER_STRATEGY[key] = 0
    elif args.skip_strategies:
        for s in args.skip_strategies.split(","):
            key = s.strip()
            if key in POSITION_MAX_PER_STRATEGY:
                POSITION_MAX_PER_STRATEGY[key] = 0

    scan_label = args.label or ("Morning Gap Scan" if args.strategies else "Evening Scan")

    print("="*80)
    print("🚀 LIVE POSITION TRADING SCANNER")
    print("="*80)
    print(f"📅 Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  [{scan_label}]")
    print(f"⚠️  Risk per trade: {POSITION_RISK_PER_TRADE_PCT}%")
    print(f"📊 Max positions: {POSITION_MAX_TOTAL} total")
    active = [k for k, v in POSITION_MAX_PER_STRATEGY.items() if v > 0]
    print(f"📊 Active strategies: {', '.join(active) if active else 'None'}")
    print("="*80 + "\n")

    # --------------------------------------------------
    # Step 1: Check Market Regime (same logic as backtester)
    # --------------------------------------------------
    regime, regime_params, allow_new_entries = check_market_regime()

    if not allow_new_entries:
        print("\n🔴 RISK_OFF — No new entries. Managing exits only.")
    elif regime == PositionRegime.NEUTRAL:
        print("\n🟡 NEUTRAL market — Cautious entries allowed (longs + quality shorts)")

    # --------------------------------------------------
    # Step 2: Monitor Open Positions for Exits/Actions
    # --------------------------------------------------
    print(f"\n📊 Current Open Positions: {position_tracker.get_position_count()}/{POSITION_MAX_TOTAL}")

    action_signals = {'exits': [], 'partials': [], 'pyramids': [], 'warnings': []}
    exited_tickers_today: set[str] = set()

    if args.skip_monitor:
        print("⏭️  Position monitoring skipped (morning scan — no close prices yet)")
    elif position_tracker.get_position_count() > 0:
        print(position_tracker)

        print("\n" + "="*80)
        print("🔍 MONITORING POSITIONS FOR EXIT/ACTION SIGNALS...")
        print("="*80)

        action_signals = monitor_positions(position_tracker)

        # Display action signals
        total_actions = len(action_signals['exits']) + len(action_signals['partials']) + len(action_signals['pyramids'])

        if total_actions > 0:
            print(f"\n⚠️  {total_actions} ACTION(S) REQUIRED:\n")

            # Exits (highest priority)
            if action_signals['exits']:
                print(f"🚨 EXITS ({len(action_signals['exits'])}):")
                for exit_sig in action_signals['exits']:
                    ticker = exit_sig['ticker']
                    exited_tickers_today.add(ticker)
                    print(f"   {ticker}: {exit_sig['type']} - {exit_sig['reason']}")
                    print(f"   → {exit_sig['action']}")
                    print()
                    
                    # GET POSITION FIRST (before removing)
                    pos = position_tracker.get_position(ticker)
                    strategy = pos.get('strategy') if pos else None
                    direction = pos.get('direction', 'LONG') if pos else 'LONG'
                    entry_price = float(pos.get('entry_price', 0)) if pos else 0.0
                    shares = float(pos.get('shares', 0)) if pos else 0.0
                    exit_price = float(exit_sig.get('current_price', 0))
                    profit_per_share = (
                        (exit_price - entry_price)
                        if direction != 'SHORT'
                        else (entry_price - exit_price)
                    )
                    profit_loss = profit_per_share * shares if shares > 0 else 0.0
                    days_held = int(exit_sig.get('days_held', 0))
                    strategy_tracker = get_strategy_tracker(strategy) if strategy else None
                     
                    # REMOVE THE POSITION FROM TRACKER
                    position_tracker.remove_position(ticker)
                     
                    # UPDATE PER-STRATEGY TRACKER / HISTORY
                    if strategy_tracker is not None:
                        strategy_tracker.close_position(
                            ticker=ticker,
                            exit_date=pd.Timestamp.today().strftime('%Y-%m-%d'),
                            exit_price=exit_price,
                            exit_reason=exit_sig['type'],
                            profit_loss=profit_loss,
                            r_multiple=exit_sig.get('current_r'),
                            days_held=days_held,
                            strategy=strategy,
                            entry_date=(
                                pd.to_datetime(pos.get('entry_date')).strftime('%Y-%m-%d')
                                if pos and pos.get('entry_date') is not None
                                else None
                            ),
                            entry_price=entry_price if entry_price > 0 else None,
                            setup_type=pos.get('setup_type') if pos else None,
                            trigger_level=pos.get('trigger_level') if pos else None,
                            leadership_stage=pos.get('leadership_stage') if pos else None,
                        )

            # Partial profits
            if action_signals['partials']:
                print(f"💰 PARTIAL PROFITS ({len(action_signals['partials'])}):")
                for partial in action_signals['partials']:
                    print(f"   {partial['ticker']}: {partial['reason']}")
                    print(f"   → {partial['action']}")
                    print()

            # Pyramid opportunities
            if action_signals['pyramids']:
                print(f"📈 PYRAMID OPPORTUNITIES ({len(action_signals['pyramids'])}):")
                for pyramid in action_signals['pyramids']:
                    print(f"   {pyramid['ticker']}: {pyramid['reason']}")
                    print(f"   → {pyramid['action']}")
                    print()
        else:
            print("\n✅ No exit/action signals - all positions healthy")

        # Display warnings (FYI only)
        if action_signals['warnings']:
            print(f"\n⚠️  Warnings ({len(action_signals['warnings'])}):")
            for warning in action_signals['warnings']:
                print(f"   {warning.get('message', warning.get('ticker', 'Unknown'))}")

    # Count positions per strategy
    strategy_counts = {}
    for ticker, pos in position_tracker.get_all_positions().items():
        strategy = pos.get('strategy', 'Unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    if strategy_counts:
        print("\n📊 Positions by Strategy:")
        for strategy, count in strategy_counts.items():
            max_for_strategy = POSITION_MAX_PER_STRATEGY.get(strategy, 5)
            print(f"   {strategy}: {count}/{max_for_strategy}")

    # --------------------------------------------------
    # Step 3: Run Position Trading Scanner
    # --------------------------------------------------
    print("\n" + "="*80)
    print("🔍 SCANNING S&P 500 FOR POSITION TRADES...")
    print("="*80 + "\n")

    # Load current S&P 500 tickers (with historical-file fallback during bootstrap)
    tickers = get_sp500_tickers()
    if not tickers:
        print("⚠️  Could not load S&P 500 constituents for the live scan")
        tickers = []

    # Run scanner as of today
    today = pd.Timestamp.today()
    signals = run_scan_as_of(today, tickers, rs_bought_tracker=rs_bought_tracker)

    print(f"\n✅ Scanner found {len(signals)} raw signals")

    # --------------------------------------------------
    # Step 4: Pre-buy Check (Format & Deduplicate)
    # --------------------------------------------------
    # RISK_OFF = no new entries (matches backtester exactly)
    if not allow_new_entries:
        print("\n🔴 RISK_OFF regime — skipping new entries, exits only.")
        trade_ready = pd.DataFrame()
    elif signals:
        trade_ready = pre_buy_check(
            signals,
            benchmark=REGIME_INDEX,
            as_of_date=None,
            strategy_trackers=strategy_trackers,
        )

        # Preserve the ranker's option alert outside all equity position filters.
        equity_trades, streak_alerts = split_streak_option_alerts(trade_ready)

        # Filter out equity positions we already hold
        if not equity_trades.empty:
            equity_trades = filter_trades_by_position(equity_trades, position_tracker, as_of_date=None)
            equity_trades = filter_trades_by_same_day_exits(equity_trades, exited_tickers_today)

        # Check equity-only position limits.
        if not equity_trades.empty:
            current_total = position_tracker.get_position_count()
            available_slots = max(0, POSITION_MAX_TOTAL - current_total)

            # Further filter by per-strategy limits
            filtered_trades = []
            for _, trade in equity_trades.iterrows():
                strategy = trade["Strategy"]
                current_count = strategy_counts.get(strategy, 0)
                max_for_strategy = POSITION_MAX_PER_STRATEGY.get(strategy, 5)

                if current_count < max_for_strategy and len(filtered_trades) < available_slots:
                    filtered_trades.append(trade)
                    strategy_counts[strategy] = current_count + 1

            equity_trades = pd.DataFrame(filtered_trades) if filtered_trades else pd.DataFrame()
        trade_ready = pd.concat([equity_trades, streak_alerts], ignore_index=True)
    else:
        trade_ready = pd.DataFrame()

    # --------------------------------------------------
    # Step 5: Display Results
    # --------------------------------------------------
    print("\n" + "="*80)
    print("📋 TRADE-READY SIGNALS")
    print("="*80)

    if not trade_ready.empty:
        print(f"\n✅ {len(trade_ready)} new position signal(s) ready:\n")

        # Calculate position sizing
        equity = POSITION_INITIAL_EQUITY  # From config (default $100k)
        risk_pct = POSITION_RISK_PER_TRADE_PCT / 100  # 2%
        risk_amount = equity * risk_pct

        print(f"💰 Account Equity: ${equity:,} | Risk per Trade: {risk_pct*100}% = ${risk_amount:,}\n")

        # Display format with position sizing
        for idx, trade in trade_ready.iterrows():
            ticker = trade['Ticker']
            entry = trade['Entry']
            stop = trade['StopLoss']
            target = trade['Target']
            if is_streak_option_alert(trade):
                print(f"   {idx+1}. {ticker:<6} | NEXT-DAY GREEN PREDICTION")
                print(f"      🎯 Buy a 7-day ITM call at next session open; exit at that session close")
                print(f"      Probability next green: {trade.get('ProbabilityNextGreen', 0):.1%}")
                print(f"      Why: {trade.get('PredictionReason', 'qualified Streak signal')}")
                print()
                continue

            # Skip if entry/stop are invalid (data corruption guard)
            if not entry or not stop or entry <= 0 or stop <= 0:
                continue

            # Calculate shares — enforce 1% minimum stop distance to prevent
            # position-size explosions when gap fill level is very close to entry.
            risk_per_share = abs(entry - stop)
            min_risk = entry * 0.01
            risk_per_share = max(risk_per_share, min_risk)
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            # Hard cap: never exceed 25% of capital by market value
            if entry > 0:
                shares = min(shares, int(equity * 0.25 / entry))
            position_size = shares * entry

            print(f"   {idx+1}. {ticker:<6} | {trade['Strategy']:<35}")
            print(f"      🎯 BUY {shares} shares at ${entry:.2f} = ${position_size:,.0f} position")
            print(f"      📉 Stop: ${stop:.2f} (risk: ${risk_per_share:.2f}/share)")
            print(f"      📈 Target: ${target:.2f} | Max Days: {trade.get('MaxDays', 150)}")
            print(f"      Score: {trade.get('Score', 0):.1f} | Priority: {trade.get('Priority', 999)}")
            print()
    else:
        print("\n⚠️  No trade-ready signals today")
        print("   - All active strategies checked")
        print("   - Either no setups found or all slots filled\n")

    # --------------------------------------------------
    # Step 6: Auto-Record Trades to Position Tracker
    # --------------------------------------------------
    if not trade_ready.empty:
        print("="*80)
        print("💾 Auto-Recording Trades to Position Tracker...")
        print("="*80 + "\n")
        
        for _, trade in trade_ready.iterrows():
            ticker = trade['Ticker']
            entry_price = trade['Entry']
            strategy = trade['Strategy']
            stop_loss = trade['StopLoss']
            target = trade['Target']
            if is_streak_option_alert(trade):
                print(f"✅ {ticker} prediction alert sent (not added to equity position tracker)")
                continue
            extra_fields = {
                'direction': trade.get('Direction', 'LONG'),
                'max_days': trade.get('MaxDays'),
                'entry_score': trade.get('EntryScore', trade.get('Score')),
                'setup_type': trade.get('SetupType'),
                'signal_type': trade.get('SignalType'),
                'trigger_level': trade.get('TriggerLevel'),
                'gap_fill_level': trade.get('GapFillLevel'),
                'gap_high': trade.get('GapHigh'),
                'zone_support': trade.get('ZoneSupport'),
                'zone_resistance': trade.get('ZoneResistance'),
                'gap_low': trade.get('GapLow'),
                'gap_mid': trade.get('GapMid'),
                'gap_support': trade.get('GapSupport'),
                'gap_resistance': trade.get('GapResistance'),
            }
            extra_fields = {
                key: value
                for key, value in extra_fields.items()
                if value is not None and not pd.isna(value)
            }
            
            success = position_tracker.add_position(
                ticker=ticker,
                entry_date=datetime.now(),
                entry_price=entry_price,
                strategy=strategy,
                stop_loss=stop_loss,
                target=target,
                **extra_fields,
            )
            
            if success:
                print(f"✅ {ticker} @ ${entry_price:.2f} ({strategy})")

                get_strategy_tracker(strategy).add_bought(
                    ticker=ticker,
                    entry_date=pd.Timestamp.today().strftime('%Y-%m-%d'),
                    entry_price=entry_price,
                    strategy=strategy,
                    setup_type=trade.get('SetupType'),
                    trigger_level=trade.get('TriggerLevel'),
                    leadership_stage=trade.get('LeadershipStage'),
                )
            else:
                print(f"⚠️  {ticker} - already recorded or error")
        
        print()

    # --------------------------------------------------
    # Step 7: Send Email Alert (only if there is something actionable)
    # --------------------------------------------------
    has_new_trades = not trade_ready.empty
    has_action_signals = (
        len(action_signals.get('exits', [])) > 0
        or len(action_signals.get('partials', [])) > 0
        or len(action_signals.get('pyramids', [])) > 0
    )

    if has_new_trades or has_action_signals:
        print("="*80)
        print("📧 Sending Email Alert...")
        print("="*80 + "\n")

        send_email_alert(
            trade_df=trade_ready,
            all_signals=signals if signals else [],
            subject_prefix=f"📊 {scan_label}",
            position_tracker=position_tracker,
            action_signals=action_signals
        )
    else:
        print("📭 No actionable signals — email skipped")

    print("\n" + "="*80)
    print("✨ Scan Complete")
    print("="*80)
