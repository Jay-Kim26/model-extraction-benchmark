"""Deterministic selection tests for ActiveThief strategies."""

import types
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mebench.attackers.activethief import ActiveThief
from mebench.core.state import BenchmarkState


class IndexedDataset(Dataset):
    """Dataset backed by a list of tensors."""

    def __init__(self, items):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx], 0


class EntropyModel(nn.Module):
    """Model that returns high/low entropy logits based on input mean."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        means = x.view(x.size(0), -1).mean(dim=1)
        logits = torch.zeros(x.size(0), 2)
        for i, m in enumerate(means):
            if m.item() < 1.0:
                logits[i] = torch.tensor([0.0, 0.0])
            else:
                logits[i] = torch.tensor([10.0, 0.0])
        return logits


class FeatureModel(nn.Module):
    """Model exposing a simple features method."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return torch.zeros(flat.size(0), 2)


def _make_attack() -> ActiveThief:
    attack = ActiveThief.__new__(ActiveThief)
    attack.strategy = "uncertainty"
    attack.pool_dataset = None
    return attack


def test_select_uncertainty_prefers_high_entropy() -> None:
    attack = _make_attack()
    state = BenchmarkState()

    high_entropy = torch.zeros(1, 1, 2)
    low_entropy = torch.ones(1, 1, 2) * 10.0
    attack.pool_dataset = IndexedDataset([high_entropy, low_entropy])

    state.attack_state["unlabeled_indices"] = [0, 1]
    state.attack_state["substitute"] = EntropyModel()

    selected = attack._select_uncertainty(1, state)

    assert selected == [0]


def test_select_k_center_prefers_farthest_point() -> None:
    attack = _make_attack()
    state = BenchmarkState()

    labeled = torch.tensor([[[0.0, 0.0]]])
    near = torch.tensor([[[0.1, 0.1]]])
    far = torch.tensor([[[10.0, 10.0]]])
    attack.pool_dataset = IndexedDataset([labeled, near, far])

    state.attack_state["labeled_indices"] = [0]
    state.attack_state["unlabeled_indices"] = [1, 2]
    state.attack_state["substitute"] = FeatureModel()

    selected = attack._select_k_center(1, state)

    assert selected == [2]


def test_select_dfal_prefers_smallest_perturbation() -> None:
    attack = _make_attack()
    state = BenchmarkState()

    close = torch.tensor([[[0.1, 0.1]]])
    far = torch.tensor([[[1.0, 1.0]]])
    attack.pool_dataset = IndexedDataset([close, far])

    state.attack_state["unlabeled_indices"] = [0, 1]
    state.attack_state["substitute"] = FeatureModel()

    def fake_compute_input_gradient(self, model, x, target_class):
        scale = float(x.mean().item())
        return torch.ones_like(x) * scale

    attack._compute_input_gradient = types.MethodType(fake_compute_input_gradient, attack)

    selected = attack._select_dfal(1, state)

    assert selected == [0]
