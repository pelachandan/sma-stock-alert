"""
Strategy Registry and Factory
=============================
Central registry for all available strategies.
Provides factory pattern for instantiating strategies by name.
"""

from typing import Dict, Type, List, Any
from .base import BaseStrategy


class StrategyRegistry:
    """Registry for all available strategies."""
    
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseStrategy]) -> None:
        """
        Register a strategy.
        
        Args:
            name: Name to register strategy under
            strategy_class: Strategy class to register
        """
        cls._strategies[name] = strategy_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseStrategy]:
        """
        Get a strategy class by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy class
            
        Raises:
            KeyError: If strategy not found
        """
        if name not in cls._strategies:
            raise KeyError(f"Strategy '{name}' not registered. Available: {list(cls._strategies.keys())}")
        return cls._strategies[name]
    
    @classmethod
    def create(cls, name: str) -> BaseStrategy:
        """
        Create an instance of a strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Instantiated strategy
        """
        strategy_class = cls.get(name)
        return strategy_class()
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Get list of registered strategy names."""
        return list(cls._strategies.keys())
    
    @classmethod
    def list_active(cls) -> List[str]:
        """Get list of active (enabled) strategies from config."""
        from src.config import POSITION_MAX_PER_STRATEGY
        
        return [name for name, limit in POSITION_MAX_PER_STRATEGY.items() if limit > 0]


def register_builtin_strategies() -> None:
    """Register all built-in strategies."""
    try:
        from .relative_strength import RelativeStrengthRanker
        StrategyRegistry.register("RelativeStrength_Ranker_Position", RelativeStrengthRanker)
    except ImportError:
        pass

    try:
        from .high_52w_strategy import High52Strategy
        StrategyRegistry.register("High52_Position", High52Strategy)
    except ImportError:
        pass

    try:
        from .consolidation_breakout import ConsolidationBreakout
        StrategyRegistry.register("BigBase_Breakout_Position", ConsolidationBreakout)
    except ImportError:
        pass

    try:
        from .ema_signals import EMACrossover
        StrategyRegistry.register("EMA_Crossover_Position", EMACrossover)
    except ImportError:
        pass

    try:
        from .ema_stack_alignment import EMAStackAlignment
        StrategyRegistry.register("EMA_StackAlignment_Position", EMAStackAlignment)
    except ImportError:
        pass

    try:
        from .gap_reversal import GapReversalPosition
        StrategyRegistry.register("GapReversal_Position", GapReversalPosition)
    except ImportError:
        pass

    try:
        from .gap_continuation import GapContinuationPosition
        StrategyRegistry.register("GapContinuation_Position", GapContinuationPosition)
    except ImportError:
        pass

    try:
        from .rally_pattern import RallyPatternPosition
        StrategyRegistry.register("RallyPattern_Position", RallyPatternPosition)
    except ImportError:
        pass

    try:
        from .streak import StreakPosition
        StrategyRegistry.register("Streak_Position", StreakPosition)
    except ImportError:
        pass


# Auto-register strategies on import
register_builtin_strategies()
