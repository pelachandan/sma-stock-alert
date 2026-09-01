"""Analyze daily green/red streak continuation and backtest exact Streak trades."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.market import get_historical_data

STREAK_LENGTHS = range(1, 9)
FORWARD_DAYS = (1, 2, 3, 5)


def prepare_streak_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily direction, reset-on-flat streak lengths, and EMA trend columns."""
    working = df.copy().sort_index()
    working["Close"] = pd.to_numeric(working["Close"], errors="coerce")
    working = working.dropna(subset=["Close"])
    working["Return"] = working["Close"].pct_change()
    working["Direction"] = "FLAT"
    working.loc[working["Return"] > 0, "Direction"] = "GREEN"
    working.loc[working["Return"] < 0, "Direction"] = "RED"

    groups = working["Direction"].ne(working["Direction"].shift()).cumsum()
    working["StreakLength"] = working.groupby(groups).cumcount() + 1
    working.loc[working["Direction"] == "FLAT", "StreakLength"] = 0
    working["EMA20"] = working["Close"].ewm(span=20, adjust=False).mean()
    working["EMA50"] = working["Close"].ewm(span=50, adjust=False).mean()
    return working


def analyze_streaks(data: pd.DataFrame) -> pd.DataFrame:
    """Return continuation statistics for exact streak lengths one through eight."""
    rows = []
    for direction in ("GREEN", "RED"):
        multiplier = 1 if direction == "GREEN" else -1
        for length in STREAK_LENGTHS:
            matches = data[
                (data["Direction"] == direction) & (data["StreakLength"] == length)
            ].copy()
            matches["NextReturn"] = data["Close"].shift(-1) / data["Close"] - 1
            matches = matches.dropna(subset=["NextReturn"])
            directional_return = matches["NextReturn"] * multiplier
            rows.append({
                "Pattern": ("G" if direction == "GREEN" else "R") * length,
                "Occurrences": len(matches),
                "NextSameDirection%": round((directional_return > 0).mean() * 100, 2)
                if len(matches) else 0.0,
                "NextReversal%": round((directional_return < 0).mean() * 100, 2)
                if len(matches) else 0.0,
                "AvgNextDayReturn%": round(directional_return.mean() * 100, 3)
                if len(matches) else 0.0,
                "MedianNextDayReturn%": round(directional_return.median() * 100, 3)
                if len(matches) else 0.0,
            })
    return pd.DataFrame(rows)


def backtest_streak(
    data: pd.DataFrame, *, direction: str, length: int, use_trend_filter: bool, transaction_cost: float
) -> pd.DataFrame:
    """Backtest signals entered next session open and exited that session's close."""
    if "Open" not in data:
        raise ValueError("Streak intraday backtest requires an Open column.")
    streak_direction = "GREEN" if direction == "LONG" else "RED"
    trend_ok = (
        (data["Close"] > data["EMA20"]) & (data["EMA20"] > data["EMA50"])
        if direction == "LONG"
        else (data["Close"] < data["EMA20"]) & (data["EMA20"] < data["EMA50"])
    )
    entries = data[(data["Direction"] == streak_direction) & (data["StreakLength"] == length)].copy()
    if use_trend_filter:
        entries = entries[trend_ok.loc[entries.index]]

    rows = []
    for date, row in entries.iterrows():
        position = data.index.get_loc(date)
        if position + 1 >= len(data):
            continue
        exit_row = data.iloc[position + 1]
        entry_price = exit_row["Open"]
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        gross_return = (exit_row["Close"] / entry_price - 1) * (1 if direction == "LONG" else -1)
        trade = {
            "SignalDate": date,
            "EntryDate": exit_row.name,
            "ExitDate": exit_row.name,
            "Direction": direction,
            "Entry": entry_price,
            "Exit": exit_row["Close"],
            "GrossReturn": gross_return,
            "NetReturn": gross_return - transaction_cost,
        }
        for days in FORWARD_DAYS:
            if position + days < len(data):
                forward_return = data["Close"].iloc[position + days] / entry_price - 1
                trade[f"Forward{days}DReturn"] = forward_return * (1 if direction == "LONG" else -1)
        rows.append(trade)
    return pd.DataFrame(rows)


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int]:
    """Calculate the complete gross and cost-adjusted Streak backtest summary."""
    if trades.empty:
        return {"NumberOfTrades": 0}
    returns = trades["GrossReturn"]
    winners = returns > 0
    losers = returns < 0
    gross_profit = returns[winners].sum()
    gross_loss = abs(returns[losers].sum())
    gross_compounded = (1 + returns).prod() - 1
    net_total = trades["NetReturn"].sum()
    compounded = (1 + trades["NetReturn"]).prod() - 1
    equity = (1 + trades["NetReturn"]).cumprod()
    drawdown = equity / equity.cummax() - 1
    result: dict[str, float | int] = {
        "NumberOfTrades": len(trades),
        "WinningTrades": int(winners.sum()),
        "LosingTrades": int(losers.sum()),
        "WinRate%": round(winners.mean() * 100, 2),
        "AverageReturn%": round(returns.mean() * 100, 3),
        "MedianReturn%": round(returns.median() * 100, 3),
        "TotalReturn%": round(returns.sum() * 100, 3),
        "TotalNetReturn%": round(net_total * 100, 3),
        "CompoundedGrossReturn%": round(gross_compounded * 100, 3),
        "CompoundedNetReturn%": round(compounded * 100, 3),
        "BestTrade%": round(returns.max() * 100, 3),
        "WorstTrade%": round(returns.min() * 100, 3),
        "ProfitFactor": round(gross_profit / gross_loss, 3) if gross_loss else float("inf"),
        "MaximumDrawdown%": round(drawdown.min() * 100, 3),
        "AverageWinner%": round(returns[winners].mean() * 100, 3) if winners.any() else 0.0,
        "AverageLoser%": round(returns[losers].mean() * 100, 3) if losers.any() else 0.0,
    }
    for days in FORWARD_DAYS:
        column = f"Forward{days}DReturn"
        result[f"{days}DayForwardReturn%"] = round(trades[column].mean() * 100, 3) if column in trades else 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze and backtest daily close-to-close Streak signals.")
    parser.add_argument("--ticker", default="MSFT", help="Ticker to load from data/historical (default: MSFT)")
    parser.add_argument("--csv", type=Path, help="Optional OHLCV CSV path; first column must be the date index")
    parser.add_argument("--transaction-cost", type=float, default=0.001, help="Round-trip cost as decimal (default: 0.001)")
    parser.add_argument("--output", type=Path, help="Optional CSV output path for all trade records")
    args = parser.parse_args()

    if args.transaction_cost < 0:
        parser.error("--transaction-cost must be non-negative")
    data = pd.read_csv(args.csv, index_col=0, parse_dates=True) if args.csv else get_historical_data(args.ticker)
    if data.empty or "Close" not in data:
        raise ValueError(f"No daily Close data available for {args.csv or args.ticker}.")
    data = prepare_streak_data(data)
    print("\nSTREAK ANALYSIS")
    print(analyze_streaks(data).to_string(index=False))

    all_trades = []
    for direction, length in (("LONG", 2), ("SHORT", 3)):
        for filtered in (False, True):
            trades = backtest_streak(
                data, direction=direction, length=length,
                use_trend_filter=filtered, transaction_cost=args.transaction_cost,
            )
            variant = f"{direction} exact {length} {'EMA-filtered' if filtered else 'raw'}"
            print(f"\n{variant}")
            print(pd.Series(summarize_trades(trades)).to_string())
            if not trades.empty:
                trades = trades.assign(Variant=variant)
                all_trades.append(trades)
    if args.output and all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
