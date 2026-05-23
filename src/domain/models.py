"""
Domain Models
=============
Core data structures for positions, signals, and trades.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


class StrategyType(str, Enum):
    """Available strategy types."""
    RELATIVE_STRENGTH = "RelativeStrength_Ranker_Position"
    HIGH_52W = "High52_Position"
    BIGBASE_BREAKOUT = "BigBase_Breakout_Position"
    EMA_CROSSOVER = "EMA_Crossover_Position"
    EMA_STACK_ALIGNMENT = "EMA_StackAlignment_Position"
    TREND_CONTINUATION = "TrendContinuation_Position"
    PERCENT_B = "%B_MeanReversion_Position"


class SignalType(str, Enum):
    """Signal types."""
    BUY = "BUY"
    SELL = "SELL"
    EXIT = "EXIT"
    PYRAMID = "PYRAMID"
    PARTIAL = "PARTIAL"


class PositionStatus(str, Enum):
    """Position statuses."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    STOPPED_OUT = "STOPPED_OUT"


@dataclass
class Signal:
    """Trading signal from a strategy."""
    ticker: str
    close: float
    score: float
    strategy: str
    volume: int
    date: datetime
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    target: Optional[float] = None
    normalized_score: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'Ticker': self.ticker,
            'Close': self.close,
            'Score': self.score,
            'Strategy': self.strategy,
            'Volume': self.volume,
            'Date': self.date,
            'Entry': self.entry,
            'StopLoss': self.stop_loss,
            'Target': self.target,
            'NormalizedScore': self.normalized_score,
        }


@dataclass
class Position:
    """Open trading position."""
    ticker: str
    entry_price: float
    entry_date: datetime
    strategy: str
    status: PositionStatus = PositionStatus.OPEN
    stop_loss: float = 0.0
    target: float = 0.0
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    shares: int = 0
    pyramid_adds: int = 0
    partial_exits: int = 0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'ticker': self.ticker,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date,
            'strategy': self.strategy,
            'status': self.status.value,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'exit_price': self.exit_price,
            'exit_date': self.exit_date,
            'shares': self.shares,
            'pyramid_adds': self.pyramid_adds,
            'partial_exits': self.partial_exits,
            'metadata': self.metadata,
        }


@dataclass
class Trade:
    """Completed or historical trade."""
    ticker: str
    entry_price: float
    entry_date: datetime
    exit_price: float
    exit_date: datetime
    strategy: str
    stop_loss: float
    target: float
    reason: str = ""  # Why it exited
    r_multiple: float = 0.0
    profit: float = 0.0
    shares: int = 0
    pyramids: int = 0
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'Ticker': self.ticker,
            'EntryPrice': self.entry_price,
            'EntryDate': self.entry_date,
            'ExitPrice': self.exit_price,
            'ExitDate': self.exit_date,
            'Strategy': self.strategy,
            'StopLoss': self.stop_loss,
            'Target': self.target,
            'Reason': self.reason,
            'R_Multiple': self.r_multiple,
            'Profit': self.profit,
            'Shares': self.shares,
            'Pyramids': self.pyramids,
        }
