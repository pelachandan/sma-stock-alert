import pandas as pd
import pytest

from scripts.analyze_streak import analyze_streaks, backtest_streak, prepare_streak_data, summarize_trades


def _data():
    index = pd.date_range("2024-01-02", periods=60, freq="B")
    close = [100 + (index * 0.2) + (0.1 if index % 2 else 0) for index in range(60)]
    close[-5:] = [111, 110, 111, 112, 114]
    return pd.DataFrame({"Open": close, "Close": close}, index=index)


def test_streak_analysis_counts_exact_patterns_and_reports_forward_returns():
    data = prepare_streak_data(_data())

    analysis = analyze_streaks(data)
    green_two = analysis.loc[analysis["Pattern"] == "GG"].iloc[0]
    trades = backtest_streak(
        data, direction="LONG", length=2, use_trend_filter=True, transaction_cost=0.001
    )
    summary = summarize_trades(trades)

    assert green_two["Occurrences"] >= 1
    assert not trades.empty
    assert (trades["EntryDate"] == trades["ExitDate"]).all()
    assert (trades["EntryDate"] > trades["SignalDate"]).all()
    assert "Forward5DReturn" in trades.columns
    assert summary["NumberOfTrades"] == len(trades)
    assert "ProfitFactor" in summary
    assert "TotalNetReturn%" in summary
    assert "5DayForwardReturn%" in summary


def test_streak_backtest_applies_round_trip_transaction_cost():
    data = prepare_streak_data(_data())

    trades = backtest_streak(
        data, direction="LONG", length=2, use_trend_filter=False, transaction_cost=0.001
    )

    assert (trades["GrossReturn"] - trades["NetReturn"]).tolist() == pytest.approx(
        [0.001] * len(trades)
    )
