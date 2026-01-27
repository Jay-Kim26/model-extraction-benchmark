"""Test Track A training loop determinism."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from mebench.core.query_storage import QueryStorage
from mebench.eval.evaluator import Evaluator
from mebench.models.substitute_factory import create_substitute
from mebench.core.state import BenchmarkState
from pathlib import Path
import tempfile


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, num_samples=1000, num_classes=10, output_mode="soft_prob"):
        self.x = torch.randn(num_samples, 3, 32, 32)
        if output_mode == "soft_prob":
            # 2D probability distribution [N, K]
            self.y = torch.rand(num_samples, num_classes)
        else:
            # 1D class labels [N]
            self.y = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def test_track_a_training_determinism():
    """Test that Track A training produces same results with fixed seed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config = {
            "run": {"name": "test", "seeds": [0], "device": "cpu"},
            "victim": {"output_mode": "soft_prob", "temperature": 1.0},
            "substitute": {
                "arch": "resnet18",
                "init_seed": 1234,
                "trackA": {"batch_size": 32, "steps_coeff_c": 20},  # 0.2 * B
                "optimizer": {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
                "patience": 5,
            },
        }

            # Create dataset
        dataset = SimpleDataset(num_samples=500, output_mode="soft_prob")

        # Create query storage
        cache_dir = Path(tmpdir) / "cache"
        query_storage = QueryStorage(cache_dir, output_mode="soft_prob")

        # Add all data to storage
        for i in range(0, 500, 32):
            x_batch = dataset.x[i:i+32]
            y_batch = dataset.y[i:i+32]
            query_storage.add_batch(x_batch, y_batch)

        query_storage.save()

        # Run training with seed 42
        torch.manual_seed(42)
        state1 = BenchmarkState()
        evaluator1 = Evaluator(config, state1, query_storage)

        model1 = create_substitute(
            arch=config["substitute"]["arch"],
            num_classes=10,
            input_channels=3,
        ).to("cpu")

        optimizer1 = optim.SGD(
            model1.parameters(),
            lr=float(config["substitute"]["optimizer"]["lr"]),
            momentum=float(config["substitute"]["optimizer"]["momentum"]),
            weight_decay=float(config["substitute"]["optimizer"]["weight_decay"]),
        )

        # Small number of steps for testing
        num_steps = 50
        evaluator1._train_track_a(model1, optimizer1, num_steps, 32)

        # Get final state
        params1 = {k: v.cpu().clone() for k, v in model1.state_dict().items()}

        # Reset and run again with same seed
        torch.manual_seed(42)
        state2 = BenchmarkState()
        evaluator2 = Evaluator(config, state2, query_storage)

        model2 = create_substitute(
            arch=config["substitute"]["arch"],
            num_classes=10,
            input_channels=3,
        ).to("cpu")

        optimizer2 = optim.SGD(
            model2.parameters(),
            lr=float(config["substitute"]["optimizer"]["lr"]),
            momentum=float(config["substitute"]["optimizer"]["momentum"]),
            weight_decay=float(config["substitute"]["optimizer"]["weight_decay"]),
        )

        evaluator2._train_track_a(model2, optimizer2, num_steps, 32)

        params2 = {k: v.cpu().clone() for k, v in model2.state_dict().items()}

        # Check all parameters are identical
        for key in params1:
            assert torch.allclose(params1[key], params2[key], atol=1e-5), \
                f"Parameter {key} differs between runs"

        print("Track A training is deterministic!")


def test_query_storage_save_load():
    """Test that query storage correctly saves and loads data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        storage = QueryStorage(cache_dir, output_mode="soft_prob")

        # Add batches
        x1 = torch.randn(10, 3, 32, 32)
        y1 = torch.randn(10, 10)  # Soft labels
        storage.add_batch(x1, y1)

        x2 = torch.randn(5, 3, 32, 32)
        y2 = torch.randn(5, 10)
        storage.add_batch(x2, y2)

        assert storage.count == 15

        # Save
        storage.save()

        # Load into new storage
        storage2 = QueryStorage(cache_dir, output_mode="soft_prob")
        storage2.load()

        # Verify
        assert storage2.count == 15
        assert storage2.queries.shape[0] == 15

        # Check first batch
        assert torch.allclose(storage2.queries[:10], x1)
        assert torch.allclose(storage2.labels[:10], y1)

        print("Query storage save/load works correctly!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
