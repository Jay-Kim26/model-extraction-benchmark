"""Benchmark state management."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from mebench.core.types import QueryBatch, OracleOutput


@dataclass
class BenchmarkState:
    """Global benchmark state passed to attacks.

    This provides explicit state management instead of global mutable state.

    Attributes:
        query_count: Total queries sent so far
        budget_remaining: Remaining query budget
        checkpoint_reached: Whether any checkpoint has been reached
        attack_state: Attack-specific internal state
        metadata: Additional run metadata
    """

    query_count: int = 0
    budget_remaining: int = 0
    checkpoint_reached: bool = False
    attack_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
