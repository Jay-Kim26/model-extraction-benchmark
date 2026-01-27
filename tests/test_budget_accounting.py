"""Test budget accounting (1 query = 1 image)."""

import pytest
import torch
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.oracles.oracle import Oracle


def test_budget_increments_by_batch_size():
    """Verify budget decrements by number of images, not API calls."""
    # Setup
    class DummyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3)
            self.fc = torch.nn.Linear(64 * 30 * 30, 10)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    victim = DummyVictim()
    config = {
        "input_size": [32, 32],
        "channels": 3,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "output_mode": "soft_prob",
        "temperature": 1.0,
        "output_modes_supported": ["soft_prob", "hard_top1"],
    }
    state = BenchmarkState(budget_remaining=1000)
    oracle = Oracle(victim, config, state)

    # Query batch of 10 images
    x_batch = torch.randn(10, 3, 32, 32)
    query_batch = QueryBatch(x=x_batch)

    initial_budget = state.budget_remaining
    oracle.query(query_batch.x)
    final_budget = state.budget_remaining

    # Verify budget decreased by batch size (10)
    assert initial_budget - final_budget == 10
    assert oracle.query_count == 10


def test_multiple_queries_accumulate():
    """Verify multiple queries accumulate correctly."""
    class DummyVictim(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 64, 3)
            self.fc = torch.nn.Linear(64 * 30 * 30, 10)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    victim = DummyVictim()
    config = {
        "input_size": [32, 32],
        "channels": 3,
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "output_mode": "soft_prob",
        "temperature": 1.0,
        "output_modes_supported": ["soft_prob", "hard_top1"],
    }
    state = BenchmarkState(budget_remaining=1000)
    oracle = Oracle(victim, config, state)

    # Query 3 batches of different sizes
    oracle.query(torch.randn(5, 3, 32, 32))
    oracle.query(torch.randn(15, 3, 32, 32))
    oracle.query(torch.randn(30, 3, 32, 32))

    assert oracle.query_count == 5 + 15 + 30  # 50 total
    assert state.budget_remaining == 1000 - 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
