"""Random selection baseline attack."""

import torch
from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader


class RandomBaseline(BaseAttack):
    """Attack that randomly samples from surrogate or seed pool."""

    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)
        self.dataloader = None
        self.iterator = None

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        if self.dataloader is None:
            self.dataloader = create_dataloader(
                state.metadata.get("dataset_config", {}),
                batch_size=int(k),
                shuffle=True,
            )
            self.iterator = iter(self.dataloader)

        try:
            x_batch, _ = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            x_batch, _ = next(self.iterator)

        # Truncate if k is smaller than batch_size
        x_batch = x_batch[:k]
        
        # Contract: Inputs in [0, 1].
        return QueryBatch(x=x_batch, meta={"synthetic": False})

    def observe(
        self,
        query_batch: QueryBatch,
        oracle_output: OracleOutput,
        state: BenchmarkState,
    ) -> None:
        # Random baseline doesn't update state based on output
        pass
