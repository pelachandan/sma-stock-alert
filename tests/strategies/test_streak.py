import pandas as pd

import src.scanning.validator as validator
import src.strategies.streak as streak_module
from src.strategies.streak import StreakPosition


def _history(seed: float, periods: int = 150) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=periods, freq="B")
    steps = pd.Series(range(periods), index=index, dtype=float)
    close = seed + steps * 0.15 + ((steps % 7) - 3) * 0.35
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000 + (steps % 11) * 10_000,
        },
        index=index,
    )


def _market_data() -> dict[str, pd.DataFrame]:
    data = {
        ticker: _history(100 + index * 10)
        for index, ticker in enumerate(StreakPosition.ALLOWED_TICKERS)
    }
    data["QQQ"] = _history(400)
    return data


def test_streak_ranker_returns_one_allowed_next_session_option_alert(monkeypatch):
    data = _market_data()
    monkeypatch.setattr(streak_module, "get_historical_data", lambda ticker: data.get(ticker, pd.DataFrame()))

    signals = StreakPosition().run([], as_of_date=data["QQQ"].index[-1])

    assert len(signals) == 1
    signal = signals[0]
    assert signal["Ticker"] in StreakPosition.ALLOWED_TICKERS
    assert signal["Strategy"] == "Streak_Position"
    assert signal["Direction"] == "LONG"
    assert signal["Prediction"] == "GREEN"
    assert signal["EntryTiming"] == "NEXT_SESSION_OPEN"
    assert signal["MaxDays"] == 1
    assert 0 <= signal["ProbabilityNextGreen"] <= 1
    assert "Rolling one-year" in signal["PredictionReason"]


def test_streak_ranker_does_not_use_future_bars_in_training_or_features(monkeypatch):
    data = _market_data()
    as_of = data["QQQ"].index[130]
    monkeypatch.setattr(streak_module, "get_historical_data", lambda ticker: data.get(ticker, pd.DataFrame()))
    ranker = StreakPosition()

    before = ranker.run([], as_of_date=as_of)[0]
    for history in data.values():
        history.loc[history.index > as_of, ["Open", "High", "Low", "Close"]] *= 100
        history.loc[history.index > as_of, "Volume"] *= 100
    after = ranker.run([], as_of_date=as_of)[0]

    assert (after["Ticker"], after["ProbabilityNextGreen"], after["PredictionReason"]) == (
        before["Ticker"],
        before["ProbabilityNextGreen"],
        before["PredictionReason"],
    )


def test_streak_ranker_requires_the_fixed_eight_ticker_universe(monkeypatch):
    data = _market_data()
    data.pop("AMD")
    monkeypatch.setattr(streak_module, "get_historical_data", lambda ticker: data.get(ticker, pd.DataFrame()))

    assert StreakPosition().run(["TEST"], as_of_date=data["QQQ"].index[-1]) == []


def test_streak_prediction_metadata_survives_validation_without_displacing_equity(monkeypatch):
    data = _market_data()
    as_of = data["QQQ"].index[-1]
    monkeypatch.setattr(streak_module, "get_historical_data", lambda ticker: data.get(ticker, pd.DataFrame()))
    monkeypatch.setattr(validator, "get_historical_data", lambda ticker: data.get(ticker, pd.DataFrame()))
    prediction = StreakPosition().run([], as_of_date=as_of)[0]
    equity = {
        "Ticker": prediction["Ticker"],
        "Strategy": "GapContinuation_Position",
        "Entry": prediction["Entry"],
        "StopLoss": prediction["StopLoss"],
        "Target": prediction["Entry"] * 1.02,
        "Score": 50,
        "Direction": "LONG",
    }

    validated = validator.pre_buy_check([equity, prediction], as_of_date=as_of)

    assert len(validated) == 2
    validated_prediction = validated.loc[validated["Strategy"] == "Streak_Position"].iloc[0]
    assert validated_prediction["EntryTiming"] == "NEXT_SESSION_OPEN"
    assert validated_prediction["Prediction"] == "GREEN"
    assert validated_prediction["ProbabilityNextGreen"] == prediction["ProbabilityNextGreen"]
