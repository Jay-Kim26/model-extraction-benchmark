"""Base attack interface."""

from abc import ABC, abstractmethod
from typing import Optional
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState


class BaseAttack(ABC):
    """Base class for model extraction attacks.

    All attacks must implement propose() and optionally observe().
    The engine does not know about attack internals (pool selection vs generation).
    """

    def __init__(self, config: dict, state: BenchmarkState):
        """Initialize attack with configuration and state.

        Args:
            config: Attack-specific configuration
            state: Global benchmark state
        """
        self.config = config
        self.state = state

    @abstractmethod
    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        """Propose k queries to send to oracle.

        Args:
            k: Number of queries to propose
            state: Current benchmark state

        Returns:
            QueryBatch with k queries
        """
        ...

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        """Observe oracle response and update internal state.

        Args:
            query_batch: The query batch that was sent
            oracle_output: Oracle response
            state: Current benchmark state
        """
        # Default: no-op for attacks that don't need to observe
        pass
