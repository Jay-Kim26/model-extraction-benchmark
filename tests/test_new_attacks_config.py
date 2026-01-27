"""Tests for new attack configurations."""

import pytest

from mebench.core.validate import validate_config


def test_swiftthief_requires_soft_prob():
    """SwiftThief requires soft_prob output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "swiftthief", "output_mode": "soft_prob"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(config)  # Should not raise


def test_swiftthief_rejects_hard_top1():
    """SwiftThief rejects hard_top1 output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "swiftthief", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    with pytest.raises(ValueError, match="swiftthief requires soft_prob"):
        validate_config(config)


def test_dissector_requires_hard_top1():
    """Dissector requires hard_top1 output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "blackbox_dissector", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(config)  # Should not raise


def test_dissector_rejects_soft_prob():
    """Dissector rejects soft_prob output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "blackbox_dissector", "output_mode": "soft_prob"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    with pytest.raises(ValueError, match="blackbox_dissector requires hard_top1"):
        validate_config(config)


def test_cloudleak_requires_soft_prob():
    """CloudLeak requires soft_prob output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "seed", "seed_size": 1000, "seed_name": "MNIST"},
        "attack": {"name": "cloudleak", "output_mode": "soft_prob"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    validate_config(config)  # Should not raise


def test_cloudleak_rejects_hard_top1():
    """CloudLeak rejects hard_top1 output mode."""
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "hard_top1", "temperature": 1.0},
        "dataset": {"data_mode": "seed", "seed_size": 1000, "seed_name": "MNIST"},
        "attack": {"name": "cloudleak", "output_mode": "hard_top1"},
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    with pytest.raises(ValueError, match="cloudleak requires soft_prob"):
        validate_config(config)


def test_new_attacks_mode_mismatch():
    """New attacks reject mode mismatch."""
    # SwiftThief with hard mode
    config = {
        "run": {"name": "test", "seeds": [0]},
        "victim": {"output_mode": "soft_prob", "temperature": 1.0},
        "dataset": {"data_mode": "surrogate", "seed_size": 10000, "seed_name": "CIFAR10"},
        "attack": {"name": "swiftthief", "output_mode": "hard_top1"},  # Mismatch
        "budget": {"max_budget": 10000, "checkpoints": [1000, 10000]},
    }
    with pytest.raises(ValueError, match="Mode mismatch"):
        validate_config(config)
