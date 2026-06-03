"""
RelativeStrength Ranker - Bought Ticker Tracker
================================================
Maintains list of tickers recommended as buys by RelativeStrength_Ranker_Position.

Purpose:
- Prevent duplicate BUY alerts for same ticker
- Track pyramiding opportunities (+1.5R threshold)
- Track exit signals (time stops, trailing MA)
- Manage position lifecycle: bought → pyramided → closed

File format: data/rs_ranker_bought.json
{
  "TICKER": {
    "entry_date": "2026-01-15",
    "entry_price": 150.25,
    "status": "bought|pyramided|closed",
    "pyramids": [  # List of pyramid adds
      {"date": "2026-02-01", "price": 165.50, "amount": 0.5}
    ],
    "exit_date": "2026-03-10",
    "exit_price": 180.00,
    "exit_reason": "TimeStop_150d|EMA21_Trail|MA100_Trail|TargetHit",
    "profit_loss": 4500.00
  }
}
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


STRATEGY_FILE_KEY_OVERRIDES = {
    "RelativeStrength_Ranker_Position": "rs_ranker",
}


def strategy_file_key(strategy_name: str) -> str:
    """Return the file-name-safe key for a strategy."""
    if strategy_name in STRATEGY_FILE_KEY_OVERRIDES:
        return STRATEGY_FILE_KEY_OVERRIDES[strategy_name]

    base = str(strategy_name).strip()
    base = re.sub(r"_Position$", "", base)
    base = base.replace("%B", "PercentB")
    base = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base)
    base = re.sub(r"[^A-Za-z0-9]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base.lower()


def tracker_file_path_for_strategy(strategy_name: str, *, backtest: bool = False) -> str:
    key = strategy_file_key(strategy_name)
    prefix = "data\\backtest" if backtest else "data"
    return f"{prefix}\\{key}_bought.json"


def history_file_path_for_strategy(strategy_name: str, *, backtest: bool = False) -> str:
    key = strategy_file_key(strategy_name)
    prefix = "data\\backtest" if backtest else "data"
    return f"{prefix}\\{key}_trade_history.json"


class StrategyStateTracker:
    """Manages per-strategy bought ticker list and closed trade history."""

    def __init__(
        self,
        strategy_name: str,
        file_path: str | None = None,
        history_file_path: str | None = None,
        load_from_file: bool = True,
    ):
        """Initialize tracker.
        
        Args:
            strategy_name: Strategy name (e.g. RelativeStrength_Ranker_Position)
            file_path: Optional explicit path to bought-state JSON
            history_file_path: Optional explicit path to trade history JSON
            load_from_file: If True, load existing data from file. If False, start fresh.
                           Use False for diagnostic/test runs to avoid loading stale data.
        """
        self.strategy_name = strategy_name
        resolved_file_path = file_path or tracker_file_path_for_strategy(strategy_name)
        resolved_history_file_path = history_file_path or history_file_path_for_strategy(
            strategy_name,
            backtest="backtest" in str(resolved_file_path),
        )
        self.file_path = Path(resolved_file_path)
        self.history_file_path = str(resolved_history_file_path)
        self.bought_tickers = {}

        if load_from_file:
            self._load()

    def _gcs_path(self) -> str | None:
        """Return GCS config path for this tracker, or None if it's a backtest tracker."""
        if "backtest" in str(self.file_path):
            return None
        return f"config/{self.file_path.name}"

    def _load(self) -> None:
        """Load bought list from JSON file, falling back to GCS if missing locally."""
        if not self.file_path.exists():
            gcs_path = self._gcs_path()
            if gcs_path:
                from src.storage.gcs import download_file
                download_file(gcs_path, self.file_path)

        if self.file_path.exists():
            try:
                with open(self.file_path, 'r') as f:
                    self.bought_tickers = json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading {self.strategy_name} tracker: {e}")
                self.bought_tickers = {}
        else:
            self.bought_tickers = {}

    def _save(self) -> None:
        """Save bought list to JSON file and push to GCS."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, 'w') as f:
                json.dump(self.bought_tickers, f, indent=2)
            gcs_path = self._gcs_path()
            if gcs_path:
                from src.storage.gcs import upload_file
                upload_file(self.file_path, gcs_path)
        except Exception as e:
            print(f"⚠️  Error saving {self.strategy_name} tracker: {e}")

    def add_bought(
        self,
        ticker: str,
        entry_date: str,
        entry_price: float,
        strategy: str | None = None,
        **kwargs,
    ) -> None:
        """
        Record a ticker as recommended for buy.
        Handles both new entries and re-entries after cooldown.

        Args:
            ticker: Stock ticker symbol
            entry_date: Date recommended (YYYY-MM-DD)
            entry_price: Entry price
            strategy: Strategy name (defaults to tracker's configured strategy)
        """
        strategy = strategy or self.strategy_name
        self.bought_tickers[ticker] = {
            "entry_date": entry_date,
            "entry_price": entry_price,
            "strategy": strategy,
            "status": "bought",
            "pyramids": [],
            "exit_date": None,
            "exit_price": None,
            "exit_reason": None,
            "profit_loss": None,
            **kwargs,
        }
        self._save()

    def add_pyramid(self, ticker: str, date: str, price: float, size_pct: float = 0.5) -> None:
        """
        Record a pyramid add to existing position.

        Args:
            ticker: Stock ticker symbol
            date: Pyramid date (YYYY-MM-DD)
            price: Pyramid entry price
            size_pct: Size relative to original position (default 50%)
        """
        if ticker in self.bought_tickers:
            self.bought_tickers[ticker]["pyramids"].append({
                "date": date,
                "price": price,
                "amount": size_pct
            })
            # Mark as pyramided once first add is made
            if len(self.bought_tickers[ticker]["pyramids"]) == 1:
                self.bought_tickers[ticker]["status"] = "pyramided"
            self._save()

    def close_position(
        self,
        ticker: str,
        exit_date: str,
        exit_price: float,
        exit_reason: str,
        profit_loss: Optional[float] = None,
        r_multiple: Optional[float] = None,
        days_held: int = 0,
        strategy: str | None = None,
        entry_date: str | None = None,
        entry_price: float | None = None,
        **kwargs,
    ) -> None:
        """
        Record position closure and move to history.

        Args:
            ticker: Stock ticker symbol
            exit_date: Exit date (YYYY-MM-DD)
            exit_price: Exit price
            exit_reason: Reason for exit
            profit_loss: Profit/loss in dollars (optional)
            r_multiple: R-multiple (optional)
            days_held: Days held (optional)
            strategy: Strategy name override when no active tracker row exists
            entry_date: Entry date override when no active tracker row exists
            entry_price: Entry price override when no active tracker row exists
        """
        trade_data = self.bought_tickers.get(ticker, {})
        resolved_strategy = trade_data.get("strategy", strategy or self.strategy_name)
        resolved_entry_date = trade_data.get("entry_date", entry_date)
        resolved_entry_price = trade_data.get("entry_price", entry_price)

        if resolved_entry_date is None or resolved_entry_price is None:
            return

        from src.scanning.trade_history import TradeHistory

        history = TradeHistory(file_path=self.history_file_path)
        history.append_trade(
            ticker=ticker,
            strategy=resolved_strategy,
            entry_date=resolved_entry_date,
            entry_price=resolved_entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=profit_loss or 0,
            r_multiple=r_multiple or 0,
            days_held=days_held
        )

        if ticker in self.bought_tickers:
            self.bought_tickers[ticker].update({
                "status": "closed",
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "profit_loss": profit_loss,
                **kwargs,
            })
            self._save()
        else:
            self.bought_tickers[ticker] = {
                "entry_date": resolved_entry_date,
                "entry_price": resolved_entry_price,
                "strategy": resolved_strategy,
                "status": "closed",
                "pyramids": [],
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "profit_loss": profit_loss,
                **kwargs,
            }
            self._save()

    def is_bought(self, ticker: str) -> bool:
        """Check if ticker is in active position (not closed)."""
        if ticker not in self.bought_tickers:
            return False
        status = self.bought_tickers[ticker].get("status")
        return status in ("bought", "pyramided")

    def is_closed(self, ticker: str) -> bool:
        """Check if ticker position is closed."""
        if ticker not in self.bought_tickers:
            return False
        return self.bought_tickers[ticker].get("status") == "closed"

    def can_buy_again(self, ticker: str, cooldown_days: int = 30, as_of_date=None) -> bool:
        """
        Check if a closed ticker can be recommended again (cooldown expired).
        
        Args:
            ticker: Stock ticker symbol
            cooldown_days: Number of days to wait before allowing re-entry (default 30)
            as_of_date: Date to check from (default: today). Used for backtesting.
        
        Returns:
            True if ticker can be traded again, False if still in cooldown
        """
        if not self.is_closed(ticker):
            return True  # Not closed, can trade
        
        # Closed - check cooldown
        ticker_data = self.bought_tickers.get(ticker)
        if not ticker_data or not ticker_data.get("exit_date"):
            return False  # No exit date, block it
        
        from datetime import datetime, timedelta
        import pandas as pd
        
        exit_date = datetime.strptime(ticker_data["exit_date"], "%Y-%m-%d")
        
        # Use provided date (for backtesting) or current date (for live)
        if as_of_date is not None:
            current_date = pd.to_datetime(as_of_date)
            if hasattr(current_date, 'to_pydatetime'):
                current_date = current_date.to_pydatetime()
        else:
            current_date = datetime.now()
        
        days_since_exit = (current_date - exit_date).days
        
        return days_since_exit >= cooldown_days

    def has_recent_stop(self, ticker: str, trading_days_lookback: int = 5, as_of_date=None) -> bool:
        """
        Check if ticker was stopped out recently (within N trading days).
        
        Args:
            ticker: Stock ticker symbol
            trading_days_lookback: Number of trading days to look back (default 5)
            as_of_date: Current date for calculation (default: today)
        
        Returns:
            True if ticker exited via StopLoss within the lookback period
        """
        if ticker not in self.bought_tickers:
            return False
        
        ticker_data = self.bought_tickers[ticker]
        exit_reason = ticker_data.get("exit_reason")
        exit_date = ticker_data.get("exit_date")
        
        # Only block if exit was a StopLoss
        if exit_reason != "StopLoss" or not exit_date:
            return False
        
        from datetime import datetime
        import pandas as pd
        
        exit_dt = datetime.strptime(exit_date, "%Y-%m-%d")
        
        # Use provided date or current date
        if as_of_date is not None:
            current_dt = pd.to_datetime(as_of_date)
            if hasattr(current_dt, 'to_pydatetime'):
                current_dt = current_dt.to_pydatetime()
        else:
            current_dt = datetime.now()
        
        # Calculate trading days since exit (rough: assume ~1 trading day per calendar day)
        trading_days_since = (current_dt - exit_dt).days
        
        return trading_days_since <= trading_days_lookback

    def get_bought_tickers(self) -> List[str]:
        """Get list of all active bought tickers (not closed)."""
        return [
            ticker for ticker, data in self.bought_tickers.items()
            if data.get("status") in ("bought", "pyramided")
        ]

    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Get full info for a ticker."""
        return self.bought_tickers.get(ticker)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        active = sum(1 for t in self.bought_tickers.values() if t.get("status") in ("bought", "pyramided"))
        closed = sum(1 for t in self.bought_tickers.values() if t.get("status") == "closed")
        total_profit = sum(
            t.get("profit_loss", 0) or 0
            for t in self.bought_tickers.values()
            if t.get("status") == "closed"
        )
        
        return {
            "total_recommendations": len(self.bought_tickers),
            "active_positions": active,
            "closed_positions": closed,
            "total_profit_closed": total_profit,
            "active_tickers": self.get_bought_tickers()
        }

    def clear_all(self) -> None:
        """Clear all data (use with caution)."""
        self.bought_tickers = {}
        self._save()


class RSBoughtTracker(StrategyStateTracker):
    """Backward-compatible RS Ranker tracker."""

    def __init__(self, file_path: str = "data\\rs_ranker_bought.json", load_from_file: bool = True):
        super().__init__(
            strategy_name="RelativeStrength_Ranker_Position",
            file_path=file_path,
            history_file_path=history_file_path_for_strategy(
                "RelativeStrength_Ranker_Position",
                backtest="backtest" in str(file_path),
            ),
            load_from_file=load_from_file,
        )
