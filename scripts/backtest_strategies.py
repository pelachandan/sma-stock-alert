"""
Unified Strategy Backtest Runner
=================================
Run one strategy or all strategies through the walk-forward engine and get
a clean per-strategy, long/short, and yearly P&L breakdown.

Usage:
    python scripts/backtest_strategies.py                      # all active strategies
    python scripts/backtest_strategies.py --strategy gap       # GapReversal only
    python scripts/backtest_strategies.py --strategy gapcont   # GapContinuation only
    python scripts/backtest_strategies.py --strategy rally     # RallyPattern only
    python scripts/backtest_strategies.py --strategy rs        # RS Ranker only
    python scripts/backtest_strategies.py --strategy all       # explicitly all
    python scripts/backtest_strategies.py --start 2020-01-01
    python scripts/backtest_strategies.py --capital 200000
    python scripts/backtest_strategies.py --output my_results.csv

Strategy aliases:
    gap       → GapReversal_Position
    gapcont   → GapContinuation_Position
    rally     → RallyPattern_Position
    rs        → RelativeStrength_Ranker_Position
    high52    → High52_Position
    bigbase   → BigBase_Breakout_Position
    all       → all strategies (active + backtest-enabled)
"""
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import argparse
import logging
import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting.engine import WalkForwardBacktester
from src.config.settings import BACKTEST_START_DATE, BACKTEST_SCAN_FREQUENCY
from scripts.download_history import download_ticker, was_update_session_today, mark_update_session
import src.config.settings as cfg


# ─── Strategy aliases & max positions for backtest ────────────────────────────
STRATEGY_ALIASES = {
    "gap":     "GapReversal_Position",
    "gapcont": "GapContinuation_Position",
    "rally":   "RallyPattern_Position",
    "rs":      "RelativeStrength_Ranker_Position",
    "high52":  "High52_Position",
    "bigbase": "BigBase_Breakout_Position",
    "ema":     "EMA_Crossover_Position",
    "trend":   "TrendContinuation_Position",
}

# Backtest max positions per strategy (overrides 0 for isolated runs)
# Only strategies listed here are enabled when running --strategy all
BACKTEST_MAX_POSITIONS = {
    "GapReversal_Position": 5,
    "GapContinuation_Position": 5,
    "RallyPattern_Position": 5,
    "RelativeStrength_Ranker_Position": 10,
    # High52_Position: disabled — needs further tuning
    # TrendContinuation_Position: disabled — needs further tuning
    # BigBase_Breakout_Position: disabled — not yet validated
    # EMA_Crossover_Position: disabled — not yet validated
}

RALLY_STAGE_BACKTEST_CAPS = {
    "emerging": 10,
    "confirmed": 10,
}


# ─── Logging ──────────────────────────────────────────────────────────────────
class _TeeStream:
    def __init__(self, *streams) -> None:
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def fileno(self) -> int:
        return self._streams[0].fileno()


def _enable_debug_capture(strategy_key: str) -> Path:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    debug_log_path = log_dir / f"backtest_debug_{strategy_key}_{timestamp}.log"
    debug_file = debug_log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, debug_file)
    sys.stderr = _TeeStream(sys.stderr, debug_file)
    return debug_log_path


def _setup_logging() -> logging.Logger:
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = logging.getLogger("backtest_strategies")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_dir / "backtest_strategies.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


# ─── CLI ──────────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="Unified Strategy Backtest Runner")
    p.add_argument(
        "--strategy", default="all",
        help=f"Strategy alias or 'all'. Aliases: {', '.join(STRATEGY_ALIASES.keys())}",
    )
    p.add_argument("--start", default=str(BACKTEST_START_DATE), help="Start date YYYY-MM-DD")
    p.add_argument(
        "--freq", default=BACKTEST_SCAN_FREQUENCY,
        choices=["B", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI"],
        help="Scan frequency (B=daily)",
    )
    p.add_argument("--capital", type=float, default=100_000, help="Starting capital")
    p.add_argument(
        "--direction", default=None, choices=["long", "short", "both"],
        help="Trade direction for GapReversal (long/short/both). Only applies when --strategy gap.",
    )
    p.add_argument("--output", default=None, help="Output CSV (default: auto-named)")
    p.add_argument("--no-download", action="store_true", help="Skip data download")
    return p.parse_args()


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    wins = df["Outcome"] == "Win"
    losses = ~wins
    total = len(df)
    win_count = int(wins.sum())
    loss_count = int(losses.sum())
    net_pnl_col = "PnL_$"
    gross_pnl_col = "GrossPnL_$" if "GrossPnL_$" in df.columns else "PnL_$"
    gross_profit = df.loc[wins, net_pnl_col].sum()
    gross_loss = abs(df.loc[losses, net_pnl_col].sum())
    cumulative = df[net_pnl_col].cumsum()
    max_drawdown = (cumulative - cumulative.cummax()).min()
    total_brokerage = df["Brokerage_$"].sum() if "Brokerage_$" in df.columns else 0
    total_tax = df["Tax_$"].sum() if "Tax_$" in df.columns else 0
    total_gross = df[gross_pnl_col].sum() if gross_pnl_col in df.columns else df[net_pnl_col].sum()
    return {
        "TotalTrades": total,
        "Wins": win_count,
        "Losses": loss_count,
        "WinRate%": round(win_count / total * 100, 1),
        "AvgR": round(df["RMultiple"].mean(), 2),
        "AvgWinR": round(df.loc[wins, "RMultiple"].mean(), 2) if win_count else 0,
        "AvgLossR": round(df.loc[losses, "RMultiple"].mean(), 2) if loss_count else 0,
        "MaxR": round(df["RMultiple"].max(), 2),
        "MinR": round(df["RMultiple"].min(), 2),
        "ProfitFactor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "GrossPnL$": round(total_gross, 2),
        "Brokerage$": round(total_brokerage, 2),
        "Tax$": round(total_tax, 2),
        "TotalPnL$": round(df[net_pnl_col].sum(), 2),  # Net after brokerage + tax
        "MaxDrawdown$": round(max_drawdown, 2),
        "AvgHoldDays": round(df["HoldingDays"].mean(), 1),
    }


# ─── Print helpers ────────────────────────────────────────────────────────────
W = 72
SEP = "=" * W
DASH = "-" * W


def _print_strategy_block(strategy_name: str, df: pd.DataFrame):
    print(f"\n{'─' * W}")
    print(f"  📌 {strategy_name}")
    print(f"{'─' * W}")

    if df.empty:
        print("  ⚠️  No trades.")
        return

    m = compute_metrics(df)
    print(f"\n  📊 OVERALL PERFORMANCE  ({m['TotalTrades']} trades)")
    print(f"  {'Win Rate':<24} {m['WinRate%']}%  ({m['Wins']}W / {m['Losses']}L)")
    print(f"  {'Avg R':<24} {m['AvgR']:+.2f}   win: {m['AvgWinR']:+.2f}  loss: {m['AvgLossR']:+.2f}")
    print(f"  {'Best / Worst R':<24} {m['MaxR']:+.2f} / {m['MinR']:+.2f}")
    print(f"  {'Profit Factor':<24} {m['ProfitFactor']:.2f}")
    print(f"  {'Gross PnL':<24} ${m['GrossPnL$']:+,.2f}")
    if m.get('Brokerage$', 0) or m.get('Tax$', 0):
        print(f"  {'  Brokerage cost':<24} -${abs(m['Brokerage$']):,.2f}")
        print(f"  {'  Tax paid':<24} -${abs(m['Tax$']):,.2f}")
        print(f"  {'Net PnL (in-hand)':<24} ${m['TotalPnL$']:+,.2f}  ← actual take-home")
    else:
        print(f"  {'Total PnL':<24} ${m['TotalPnL$']:+,.2f}")
    print(f"  {'Max Drawdown':<24} ${m['MaxDrawdown$']:,.2f}")
    print(f"  {'Avg Hold Days':<24} {m['AvgHoldDays']:.1f}")

    # Long vs Short
    if "Direction" in df.columns:
        directions = df["Direction"].unique()
        if len(directions) > 1:
            print(f"\n  📈 LONG vs SHORT")
            print(f"  {DASH}")
            print(f"  {'Dir':<6} {'Trades':>6} {'WinRate':>8} {'AvgR':>7} {'AvgWin':>8} {'AvgLoss':>8} {'PnL':>14}")
            print(f"  {DASH}")
            for d in ["LONG", "SHORT"]:
                sub = df[df["Direction"] == d]
                if sub.empty:
                    continue
                dm = compute_metrics(sub)
                print(f"  {d:<6} {dm['TotalTrades']:>6} {dm['WinRate%']:>7.1f}%  "
                      f"{dm['AvgR']:>+6.2f}  {dm['AvgWinR']:>+7.2f}  {dm['AvgLossR']:>+7.2f}  "
                      f"${dm['TotalPnL$']:>+12,.2f}")

    # Yearly breakdown
    if "Year" in df.columns:
        print(f"\n  📅 YEARLY BREAKDOWN")
        print(f"  {DASH}")
        print(f"  {'Year':<6} {'Trades':>6}  {'WinRate':>8}  {'AvgR':>8}  {'PnL':>14}")
        print(f"  {DASH}")
        by_year = df.groupby("Year").agg(
            Trades=("RMultiple", "count"),
            Wins=("Outcome", lambda x: (x == "Win").sum()),
            AvgR=("RMultiple", "mean"),
            PnL=("PnL_$", "sum"),
        )
        for year, row in by_year.iterrows():
            wr = row["Wins"] / row["Trades"] * 100
            print(f"  {year:<6} {int(row['Trades']):>6}  {wr:>7.1f}%  {row['AvgR']:>+8.2f}  ${row['PnL']:>+12,.2f}")

    # Exit reason
    if "ExitReason" in df.columns:
        print(f"\n  🚪 EXIT REASON BREAKDOWN")
        print(f"  {DASH}")
        print(f"  {'Exit Reason':<32} {'Cnt':>4}  {'AvgR':>7}  {'PnL':>14}")
        print(f"  {DASH}")
        by_exit = (
            df.groupby("ExitReason")
            .agg(Count=("RMultiple", "count"), AvgR=("RMultiple", "mean"), PnL=("PnL_$", "sum"))
            .sort_values("Count", ascending=False)
        )
        for reason, row in by_exit.iterrows():
            print(f"  {reason:<32} {int(row['Count']):>4}  {row['AvgR']:>+6.2f}  ${row['PnL']:>+12,.2f}")


def _print_combined_summary(trades: pd.DataFrame):
    print(f"\n{SEP}")
    print("  🏆 COMBINED PORTFOLIO SUMMARY (ALL STRATEGIES)")
    print(SEP)

    m = compute_metrics(trades)
    print(f"\n  📊 TOTAL  ({m['TotalTrades']} trades across all strategies)")
    print(f"  {'Win Rate':<24} {m['WinRate%']}%  ({m['Wins']}W / {m['Losses']}L)")
    print(f"  {'Avg R':<24} {m['AvgR']:+.2f}")
    print(f"  {'Profit Factor':<24} {m['ProfitFactor']:.2f}")
    print(f"  {'Gross PnL':<24} ${m['GrossPnL$']:+,.2f}")
    if m.get('Brokerage$', 0) or m.get('Tax$', 0):
        print(f"  {'  Brokerage cost':<24} -${abs(m['Brokerage$']):,.2f}")
        print(f"  {'  Tax paid':<24} -${abs(m['Tax$']):,.2f}")
        print(f"  {'Net PnL (in-hand)':<24} ${m['TotalPnL$']:+,.2f}  ← actual take-home")
    else:
        print(f"  {'Total PnL':<24} ${m['TotalPnL$']:+,.2f}")
    print(f"  {'Max Drawdown':<24} ${m['MaxDrawdown$']:,.2f}")

    # Per-strategy contribution
    print(f"\n  📌 PER-STRATEGY CONTRIBUTION")
    print(f"  {DASH}")
    print(f"  {'Strategy':<40} {'Trades':>6}  {'WR%':>6}  {'AvgR':>7}  {'PnL':>14}")
    print(f"  {DASH}")
    for strat, grp in trades.groupby("Strategy"):
        sm = compute_metrics(grp)
        print(f"  {strat:<40} {sm['TotalTrades']:>6}  {sm['WinRate%']:>5.1f}%  "
              f"{sm['AvgR']:>+6.2f}  ${sm['TotalPnL$']:>+12,.2f}")

    # Yearly combined
    if "Year" in trades.columns:
        print(f"\n  📅 YEARLY (ALL STRATEGIES)")
        print(f"  {DASH}")
        print(f"  {'Year':<6} {'Trades':>6}  {'WinRate':>8}  {'AvgR':>8}  {'PnL':>14}")
        print(f"  {DASH}")
        by_year = trades.groupby("Year").agg(
            Trades=("RMultiple", "count"),
            Wins=("Outcome", lambda x: (x == "Win").sum()),
            AvgR=("RMultiple", "mean"),
            PnL=("PnL_$", "sum"),
        )
        for year, row in by_year.iterrows():
            wr = row["Wins"] / row["Trades"] * 100
            print(f"  {year:<6} {int(row['Trades']):>6}  {wr:>7.1f}%  {row['AvgR']:>+8.2f}  ${row['PnL']:>+12,.2f}")

    print(f"\n{SEP}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = _parse_args()
    debug_log_path = _enable_debug_capture(args.strategy.lower())
    os.environ["STOCK_ALERT_BACKTEST_DEBUG"] = "1"
    os.environ["STOCK_ALERT_BACKTEST_DEBUG_LOG"] = str(debug_log_path)
    log = _setup_logging()
    print(f"📝 Debug log: {debug_log_path}")
    log.info("Debug capture enabled: %s", debug_log_path)

    # Resolve strategy selection
    strategy_key = args.strategy.lower()
    if strategy_key in STRATEGY_ALIASES:
        target_strategy = STRATEGY_ALIASES[strategy_key]
        run_all = False
    elif strategy_key == "all":
        target_strategy = None
        run_all = True
    else:
        aliases = ", ".join(list(STRATEGY_ALIASES.keys()) + ["all"])
        print(f"❌ Unknown strategy '{args.strategy}'. Valid aliases: {aliases}")
        sys.exit(1)

    # Apply config overrides
    strategy_bucket_limits = None
    if run_all:
        # Enable all known strategies for backtest (keep production-disabled ones enabled for testing)
        for strat, max_pos in BACKTEST_MAX_POSITIONS.items():
            if cfg.POSITION_MAX_PER_STRATEGY.get(strat, 0) == 0:
                cfg.POSITION_MAX_PER_STRATEGY[strat] = max_pos
        label = "ALL STRATEGIES"
    else:
        # Isolate: zero everything, enable only the target
        for strat in list(cfg.POSITION_MAX_PER_STRATEGY.keys()):
            cfg.POSITION_MAX_PER_STRATEGY[strat] = 0
        isolated_max_positions = BACKTEST_MAX_POSITIONS.get(target_strategy, 5)
        if target_strategy == "RallyPattern_Position":
            isolated_max_positions = sum(RALLY_STAGE_BACKTEST_CAPS.values())
            strategy_bucket_limits = {target_strategy: dict(RALLY_STAGE_BACKTEST_CAPS)}
        cfg.POSITION_MAX_PER_STRATEGY[target_strategy] = isolated_max_positions
        if target_strategy == "GapReversal_Position":
            direction = args.direction or "both"
            cfg.GAP_REVERSAL_DIRECTION = direction
            label = f"{target_strategy} [{direction.upper()}]"
        else:
            label = target_strategy

    # Auto output name
    output_path = args.output or f"backtest_{strategy_key}.csv"

    print(f"\n{SEP}")
    print(f"  BACKTEST: {label}")
    print(f"  Start: {args.start}  |  Freq: {args.freq}  |  Capital: ${args.capital:,.0f}")
    print(SEP)

    # Load S&P 500 universe
    sp500 = pd.read_csv(ROOT / "data" / "sp500_constituents.csv")
    tickers = sp500["Symbol"].tolist()
    print(f"Universe: {len(tickers)} S&P 500 tickers")

    # Data download
    if not args.no_download:
        if was_update_session_today():
            print("⚡ Data already updated today — skipping download")
        else:
            import gc
            print("🔄 Updating historical data...")
            for i, ticker in enumerate(tickers, 1):
                if i % 50 == 0:
                    print(f"  [{i}/{len(tickers)}]")
                download_ticker(ticker)
                if i % 10 == 0:
                    gc.collect()
            gc.collect()
            download_ticker("SPY")
            download_ticker("QQQ")
            gc.collect()
            mark_update_session()
            print("✅ Data update complete!\n")

    # Run backtest
    print(f"\n🚀 Running backtest...")
    log.info(f"Backtest start: strategy={label} start={args.start} freq={args.freq}")
    t0 = time.time()

    bt = WalkForwardBacktester(
        tickers=tickers,
        start_date=args.start,
        scan_frequency=args.freq,
        initial_capital=args.capital,
        strategy_bucket_limits=strategy_bucket_limits,
    )
    try:
        trades = bt.run()
    except Exception as e:
        log.exception(f"Backtest crashed: {e}")
        raise

    elapsed = time.time() - t0
    print(f"\n⏱️  Backtest completed in {elapsed:.1f}s")
    print(f"📊 Total trades: {len(trades)}")
    print(f"📝 Debug log saved to: {debug_log_path}")

    if trades.empty:
        print("\n⚠️  No trades — check filters, thresholds, or data quality.")
        return

    # Save CSV
    trades.to_csv(output_path, index=False)
    print(f"💾 Trade log saved to: {output_path}")
    log.info(f"Saved {len(trades)} trades to {output_path}")

    # ─── Print results per strategy ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DETAILED RESULTS BY STRATEGY")
    print(SEP)

    strategies_found = sorted(trades["Strategy"].unique())
    for strat in strategies_found:
        _print_strategy_block(strat, trades[trades["Strategy"] == strat].copy())

    # Combined summary when multiple strategies ran
    if len(strategies_found) > 1:
        _print_combined_summary(trades)
    else:
        print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
