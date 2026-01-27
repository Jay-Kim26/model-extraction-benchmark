"""Test reproducibility with fixed seeds."""

import pytest
import torch
import numpy as np
from mebench.core.seed import set_seed


def test_set_seed_affects_torch():
    """Verify set_seed affects PyTorch RNG."""
    set_seed(42)

    # Generate random tensor
    x1 = torch.randn(5)

    # Reset seed
    set_seed(42)

    # Generate again - should be identical
    x2 = torch.randn(5)

    assert torch.allclose(x1, x2)


def test_set_seed_affects_numpy():
    """Verify set_seed affects NumPy RNG."""
    set_seed(42)

    # Generate random array
    x1 = np.random.randn(5)

    # Reset seed
    set_seed(42)

    # Generate again - should be identical
    x2 = np.random.randn(5)

    assert np.allclose(x1, x2)


def test_different_seeds_produce_different_outputs():
    """Different seeds should produce different outputs."""
    set_seed(42)
    x1 = torch.randn(5)

    set_seed(123)
    x2 = torch.randn(5)

    assert not torch.allclose(x1, x2)


def test_deterministic_cudnn_settings():
    """Verify CuDNN deterministic settings are applied."""
    set_seed(42)

    # These settings should be applied by set_seed
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
