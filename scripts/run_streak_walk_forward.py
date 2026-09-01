"""Walk-forward report for the daily one-pick next-day-green ranker."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.market import download_historical, get_historical_data
from src.strategies.streak import StreakPosition


def run_daily_walk_forward(
    ranker: StreakPosition,
    calendar: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    notional: float,
) -> pd.DataFrame:
    """Evaluate each close-time pick at its following session open and close."""
    trades = []
    for signal_date in calendar[(calendar >= start) & (calendar <= end)]:
        signals = ranker.run([], as_of_date=signal_date)
        if len(signals) != 1:
            continue
        signal = signals[0]
        data = get_historical_data(signal["Ticker"]).sort_index()
        later = data[data.index > signal_date]
        if later.empty:
            continue
        entry_date = pd.Timestamp(later.index[0])
        entry = float(later.iloc[0]["Open"])
        exit_price = float(later.iloc[0]["Close"])
        if entry <= 0:
            continue
        intraday_return = exit_price / entry - 1
        trades.append(
            {
                "SignalDate": signal_date,
                "EntryDate": entry_date,
                "ExitDate": entry_date,
                "Ticker": signal["Ticker"],
                "ProbabilityNextGreen": signal["ProbabilityNextGreen"],
                "Entry": entry,
                "Exit": exit_price,
                "OpenToCloseReturn": intraday_return,
                "PnL_$": notional * intraday_return,
            }
        )
    return pd.DataFrame(trades)


def summarize_top_picks(trades: pd.DataFrame, notional: float) -> dict[str, float | int]:
    """Summarize next-open-to-close performance using a fixed daily notional."""
    if trades.empty:
        return {"NumberOfPicks": 0}
    returns = trades["OpenToCloseReturn"]
    equity = (1 + returns).cumprod()
    return {
        "NumberOfPicks": len(trades),
        "WinRate%": round((returns > 0).mean() * 100, 2),
        "AverageOpenToCloseReturn%": round(returns.mean() * 100, 3),
        "MedianOpenToCloseReturn%": round(returns.median() * 100, 3),
        "TotalFixedNotionalPnL_$": round(trades["PnL_$"].sum(), 2),
        "FixedNotionalPerPick_$": round(notional, 2),
        "MaximumDrawdown%": round(((equity / equity.cummax()) - 1).min() * 100, 3),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward-test one daily Streak ranker pick from the fixed eight-stock universe."
    )
    parser.add_argument("--start", required=True, help="First close date to evaluate, YYYY-MM-DD.")
    parser.add_argument("--end", help="Last close date to evaluate, YYYY-MM-DD.")
    parser.add_argument("--notional", type=float, default=5_000, help="Fixed dollars per pick (default: 5000).")
    parser.add_argument("--output", type=Path, help="Optional CSV for daily picks.")
    parser.add_argument("--no-download", action="store_true", help="Use cached daily data.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.notional <= 0:
        raise ValueError("--notional must be positive.")
    if not args.no_download:
        for ticker in (*StreakPosition.ALLOWED_TICKERS, "QQQ"):
            print(f"{ticker}: refreshing daily history")
            download_historical(ticker, period="3y")

    qqq = get_historical_data("QQQ").sort_index()
    if qqq.empty:
        raise ValueError("QQQ daily history is required to establish the walk-forward calendar.")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize() if args.end else pd.Timestamp(qqq.index.max()).normalize()
    if start > end:
        raise ValueError("--start must not be after --end.")

    trades = run_daily_walk_forward(
        StreakPosition(), pd.DatetimeIndex(qqq.index).normalize(), start=start, end=end, notional=args.notional
    )
    print("\nDAILY TOP-PICK WALK-FORWARD RESULTS")
    print(pd.Series(summarize_top_picks(trades, args.notional)).to_string())
    if args.output:
        trades.to_csv(args.output, index=False)
        print(f"\nSaved {len(trades)} picks to {args.output}")


if __name__ == "__main__":
    main()
