"""SwiftThief attack implementation."""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torchvision.transforms as transforms

from mebench.attackers.base import BaseAttack
from mebench.core.types import QueryBatch, OracleOutput
from mebench.core.state import BenchmarkState
from mebench.data.loaders import create_dataloader
from mebench.models.substitute_factory import create_substitute


# ============================================================
# Repo: utils/datasetmodules/normalize.py
# ============================================================

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStdSimSiam(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, im_aug1: torch.Tensor, im_aug2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean[None, :, None, None]
        std = self.std[None, :, None, None]
        return im_aug1.sub(mean).div(std), im_aug2.sub(mean).div(std)


# ============================================================
# Repo: contrastive_learning/simsiam/criterion.py
# ============================================================

class SimSiamLoss(nn.Module):
    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()
            p = F.normalize(p, dim=1)
            z = F.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()
        elif self.ver == 'simplified':
            z = z.detach()
            return -F.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, z1, z2, p1, p2):
        loss1 = self.asymmetric_loss(p1, z2)
        loss2 = self.asymmetric_loss(p2, z1)
        return 0.5 * loss1 + 0.5 * loss2


class SoftSupSimSiamLossV17(nn.Module):
    def __init__(self, device, num_classes):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

    def forward(self, p, z, targets):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)

        dot_product = -torch.mm(p, z.T)

        entr = -(targets * targets.log()).sum(dim=1)
        entr[torch.isnan(entr)] = 0.
        norm_entr = entr / torch.log(torch.tensor(self.num_classes, device=self.device))
        reversed_norm_entr = 1 - norm_entr
        mask_similar_class1 = torch.outer(reversed_norm_entr, reversed_norm_entr)

        mask_similar_class2 = F.cosine_similarity(
            targets.T.repeat(len(targets), 1, 1),
            targets.unsqueeze(2)
        ).to(self.device)

        mask_anchor_out = (1 - torch.eye(dot_product.shape[0], device=self.device))
        mask_combined = mask_similar_class1 * mask_similar_class2 * mask_anchor_out

        dot_product_selected = dot_product * mask_combined
        selected = dot_product_selected[dot_product_selected.nonzero(as_tuple=True)]
        # repo는 mean()만 하지만, NaN 방지용 최소 가드
        if selected.numel() == 0:
            return torch.zeros((), device=self.device)
        return selected.mean()


class CL_FGSM(nn.Module):
    def __init__(self, model, eps, device):
        super().__init__()
        self.device = device
        self.model = model
        self.eps = eps

    def asymmetric_loss(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()

    def forward(self, x1, x2):
        self.model.eval()
        x1.requires_grad = True

        outs = self.model(im_aug1=x1, im_aug2=x2)
        loss1 = self.asymmetric_loss(outs['p1'], outs['z2'])
        loss2 = self.asymmetric_loss(outs['p2'], outs['z1'])
        loss = 0.5 * loss1 + 0.5 * loss2

        loss.backward()
        adv_x1 = x1 + self.eps * x1.grad.sign()
        return adv_x1.detach()


class SimSiamLoss_cost_sensitive(nn.Module):
    def __init__(self, costs: torch.Tensor):
        super().__init__()
        self.costs = costs

    def asymmetric_loss(self, p, z, targets):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -((p * z).sum(dim=1) * self.costs[targets]).mean()

    def forward(self, z1, z2, p1, p2, targets):
        loss1 = self.asymmetric_loss(p1, z2, targets)
        loss2 = self.asymmetric_loss(p2, z1, targets)
        return 0.5 * loss1 + 0.5 * loss2


# ============================================================
# Your requested head format (keep as-is)
# ============================================================

class SimSiamProjectionHead(nn.Module):
    """3-layer MLP projector (as you wrote)."""

    def __init__(self, in_dim: int, proj_dim: int = 2048, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimSiamPredictorHead(nn.Module):
    """Predictor head (as you wrote)."""

    def __init__(self, in_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiftThiefSubstitute(nn.Module):
    """Substitute model wrapper exposing a representation function."""

    def __init__(self, backbone: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # channel mismatch handling (keep)
        first_layer = None
        if hasattr(self.backbone, "conv1"):
            first_layer = self.backbone.conv1
        elif isinstance(self.backbone, nn.Sequential) and len(self.backbone) > 0:
            if isinstance(self.backbone[0], nn.Conv2d):
                first_layer = self.backbone[0]

        if first_layer is not None:
            if isinstance(first_layer, nn.Sequential) and len(first_layer) > 0:
                first_layer = first_layer[0]
            if hasattr(first_layer, 'in_channels') and x.shape[1] == 3 and first_layer.in_channels == 1:
                x = x.mean(dim=1, keepdim=True)

        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.classifier(feats)


# ============================================================
# SimSiamWrapper (for CL_FGSM) - matches repo forward signature
# ============================================================

class _SimSiamWrapper(nn.Module):
    def __init__(self, substitute: SwiftThiefSubstitute, projector: nn.Module, predictor: nn.Module):
        super().__init__()
        self.substitute = substitute
        self.projector = projector
        self.predictor = predictor

    def forward(self, im_aug1, im_aug2=None):
        if im_aug2 is None:
            z1 = self.projector(self.substitute.features(im_aug1))
            p1 = self.predictor(z1)
            return p1

        z1 = self.projector(self.substitute.features(im_aug1))
        z2 = self.projector(self.substitute.features(im_aug2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}


# ============================================================
# SwiftThief Attack
# ============================================================

class SwiftThief(BaseAttack):
    def __init__(self, config: dict, state: BenchmarkState):
        super().__init__(config, state)

        # Hyperparameters
        self.I = int(config.get("I", 10))
        self.initial_seed_ratio = float(config.get("initial_seed_ratio", 0.1))
        self.lambda1 = float(config.get("lambda1", 1.0))
        self.lambda2 = float(config.get("lambda2", 0.01))  # matches repo: 0.01 * loss3
        self.lambda3 = float(config.get("lambda3", 1.0))
        self.fgsm_epsilon = float(config.get("fgsm_epsilon", 0.01))
        self.projection_dim = int(config.get("projection_dim", 2048))

        # Sampling
        self.kde_sigma = float(config.get("kde_sigma", 1.0))
        self.max_pool_eval = int(config.get("max_pool_eval", 2000))

        # Training
        self.batch_size = int(config.get("batch_size", 256))
        self.lr = float(config.get("lr", 0.06))          # repo CL uses 0.06 SGD
        self.momentum = float(config.get("momentum", 0.9))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.cl_epochs = int(config.get("cl_epochs", 40))  # repo: 40 epochs for CL stage
        self.max_epochs = int(config.get("max_epochs", 40))  # keep compatibility (used for CL loop)
        self.patience = int(config.get("patience", 50))

        self._initialize_state(state)

        self.pool_dataset = None
        self.projection_head: Optional[nn.Module] = None
        self.predictor_head: Optional[nn.Module] = None

        # repo-style normalizers (victim mean/std)
        self.normalize: Optional[nn.Module] = None
        self.normalize_pair: Optional[nn.Module] = None

        self._ssl_transforms = None

    # -------------------------
    # Dataset/Normalizer helpers
    # -------------------------

    def _ensure_pool_dataset(self, state: BenchmarkState) -> None:
        if self.pool_dataset is not None:
            return

        dataset_config = state.metadata.get("dataset_config", {})
        if "data_mode" not in dataset_config:
            dataset_config = {"data_mode": "seed", **dataset_config}
        if "name" not in dataset_config:
            dataset_config = {"name": "CIFAR10", **dataset_config}

        self.pool_dataset = create_dataloader(dataset_config, batch_size=1, shuffle=False).dataset

        # IMPORTANT: unlabeled_indices must match real pool length (your old code guessed pool_size)
        N = len(self.pool_dataset)
        st = state.attack_state
        if not st.get("unlabeled_indices"):
            st["unlabeled_indices"] = list(range(N))
        else:
            st["unlabeled_indices"] = [i for i in st["unlabeled_indices"] if 0 <= i < N]

    def _ensure_normalizers(self, state: BenchmarkState, device: torch.device) -> None:
        if self.normalize is not None and getattr(self.normalize, "mean", None) is not None:
            if self.normalize.mean.device == device:
                return

        victim_config = state.metadata.get("victim_config", {})
        normalization = victim_config.get("normalization") or {"mean": [0.0], "std": [1.0]}
        mean = torch.tensor(normalization["mean"], dtype=torch.float32, device=device)
        std = torch.tensor(normalization["std"], dtype=torch.float32, device=device)

        self.normalize = NormalizeByChannelMeanStd(mean, std).to(device)
        self.normalize_pair = NormalizeByChannelMeanStdSimSiam(mean, std).to(device)

    # -------------------------
    # SSL transforms (raw -> aug -> raw) then normalize via normalize_pair
    # -------------------------

    def _build_ssl_transforms(self, state: BenchmarkState) -> transforms.Compose:
        # mimic repo pipeline except Normalize is handled by normalize_pair (differentiable)
        return transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),  # raw [0,1]
            ]
        )

    def _apply_two_crops(self, x_batch: torch.Tensor, device: torch.device, state: BenchmarkState) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._ssl_transforms is None:
            self._ssl_transforms = self._build_ssl_transforms(state)

        to_pil = transforms.ToPILImage()
        view1_list, view2_list = [], []

        # x_batch is raw [0,1] from mebench loaders
        for x in x_batch.detach().cpu():
            img = to_pil(x.clamp(0, 1))
            if img.mode != "RGB":
                img = img.convert("RGB")
            view1_list.append(self._ssl_transforms(img))
            view2_list.append(self._ssl_transforms(img))

        v1_raw = torch.stack(view1_list).to(device)
        v2_raw = torch.stack(view2_list).to(device)
        return v1_raw, v2_raw

    # -------------------------
    # State init
    # -------------------------

    def _initialize_state(self, state: BenchmarkState) -> None:
        # DO NOT guess pool_size; fill indices after loading pool_dataset
        state.attack_state["labeled_indices"] = []
        state.attack_state["unlabeled_indices"] = []  # filled after _ensure_pool_dataset()
        state.attack_state["query_data_x"] = []
        state.attack_state["query_data_y"] = []
        state.attack_state["class_counts"] = {}
        state.attack_state["victim_outputs"] = {}
        state.attack_state["substitute"] = None
        state.attack_state["round"] = 0
        state.attack_state["sampling_mode"] = "entropy"

    # -------------------------
    # Propose / sampling
    # -------------------------

    def propose(self, k: int, state: BenchmarkState) -> QueryBatch:
        self._ensure_pool_dataset(state)

        labeled_indices = state.attack_state["labeled_indices"]
        unlabeled_indices = state.attack_state["unlabeled_indices"]

        if len(unlabeled_indices) == 0:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            x = torch.randn(k, *input_shape)
            return QueryBatch(x=x, meta={"indices": [], "pool_exhausted": True})

        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))
        initial_seed_size = int(self.initial_seed_ratio * total_budget)

        if len(labeled_indices) < initial_seed_size:
            n_seed = min(k, initial_seed_size - len(labeled_indices), len(unlabeled_indices))
            selected = np.random.choice(unlabeled_indices, n_seed, replace=False).tolist() if n_seed > 0 else []
            if len(selected) < k:
                remaining = [idx for idx in unlabeled_indices if idx not in selected]
                n_extra = min(k - len(selected), len(remaining))
                if n_extra > 0:
                    selected.extend(np.random.choice(remaining, n_extra, replace=False).tolist())
        else:
            selected = self._select_samples(k, state)

        for idx in selected:
            if idx in state.attack_state["unlabeled_indices"]:
                state.attack_state["unlabeled_indices"].remove(idx)
                state.attack_state["labeled_indices"].append(idx)

        x_list, indices = [], []
        for idx in selected:
            img, _ = self.pool_dataset[idx]
            x_list.append(img)     # raw
            indices.append(int(idx))

        if len(x_list) < k:
            input_shape = state.metadata.get("input_shape", (3, 32, 32))
            for _ in range(k - len(x_list)):
                x_list.append(torch.randn(*input_shape))
                indices.append(-1)

        x = torch.stack(x_list[:k])
        return QueryBatch(x=x, meta={"indices": indices[:k], "sampling_mode": state.attack_state["sampling_mode"]})

    def _select_samples(self, k: int, state: BenchmarkState) -> List[int]:
        self._update_sampling_mode(state)
        if state.attack_state["substitute"] is None:
            unlabeled = state.attack_state["unlabeled_indices"]
            return np.random.choice(unlabeled, min(k, len(unlabeled)), replace=False).tolist()

        if state.attack_state["sampling_mode"] == "rare_class":
            return self._select_rare_class(k, state)
        return self._select_entropy(k, state)

    def _update_sampling_mode(self, state: BenchmarkState) -> None:
        labeled_indices = state.attack_state["labeled_indices"]
        class_counts = state.attack_state["class_counts"]

        if len(labeled_indices) == 0:
            return

        num_classes = int(state.metadata.get("num_classes") or self.config.get("num_classes") or 10)
        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))

        total_q = len(labeled_indices)
        mean_per_class = total_q / num_classes

        rare_classes = [c for c in range(num_classes) if class_counts.get(c, 0) < mean_per_class]
        if not rare_classes:
            state.attack_state["sampling_mode"] = "entropy"
            return

        rare_sum = sum(class_counts.get(c, 0) for c in rare_classes)
        mean_rare = rare_sum / len(rare_classes)
        remaining_budget = total_budget - total_q
        threshold = len(rare_classes) * (mean_per_class - mean_rare)

        state.attack_state["sampling_mode"] = "rare_class" if remaining_budget <= threshold else "entropy"

    def _select_entropy(self, k: int, state: BenchmarkState) -> List[int]:
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        substitute = state.attack_state["substitute"]
        if substitute is None:
            return np.random.choice(unlabeled_indices, min(k, len(unlabeled_indices)), replace=False).tolist()

        substitute.eval()
        device = next(substitute.parameters()).device
        self._ensure_normalizers(state, device)

        # candidate cap for speed
        cand_n = min(len(unlabeled_indices), self.max_pool_eval)
        candidates = np.random.choice(unlabeled_indices, cand_n, replace=False).tolist() if cand_n > 0 else []
        if not candidates:
            return []

        entropy_scores = []
        bs = 128
        with torch.no_grad():
            for start in range(0, len(candidates), bs):
                chunk = candidates[start:start + bs]
                x_raw = torch.stack([self.pool_dataset[i][0] for i in chunk]).to(device)
                x_norm = self.normalize(x_raw)
                probs = F.softmax(substitute(x_norm), dim=1)
                ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                entropy_scores.extend([(chunk[i], float(ent[i].item())) for i in range(len(chunk))])

        entropy_scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in entropy_scores[:min(k, len(entropy_scores))]]

    def _extract_features_for_indices(self, indices: List[int], substitute: nn.Module, device: torch.device, state: BenchmarkState, batch_size: int = 128) -> torch.Tensor:
        self._ensure_normalizers(state, device)
        feats = []
        substitute.eval()
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start:start + batch_size]
                x_raw = torch.stack([self.pool_dataset[i][0] for i in chunk]).to(device)
                x_norm = self.normalize(x_raw)
                feats.append(substitute.features(x_norm).detach())
        return torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)

    def _select_rare_class(self, k: int, state: BenchmarkState) -> List[int]:
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        labeled_indices = state.attack_state["labeled_indices"]
        class_counts = state.attack_state["class_counts"]
        victim_outputs = state.attack_state.get("victim_outputs", {})

        substitute = state.attack_state.get("substitute")
        if substitute is None or not hasattr(substitute, "features"):
            return self._select_entropy(k, state)

        num_classes = int(state.metadata.get("num_classes") or self.config.get("num_classes") or 10)
        if num_classes <= 0:
            return self._select_entropy(k, state)

        total_q = sum(class_counts.values())
        mean_per_class = total_q / num_classes if num_classes > 0 else 0
        rare_classes = [c for c in range(num_classes) if class_counts.get(c, 0) < mean_per_class]
        if not rare_classes:
            return self._select_entropy(k, state)

        y_n = min(rare_classes, key=lambda c: class_counts.get(c, 0))
        q_y = [idx for idx in labeled_indices if idx in victim_outputs and int(victim_outputs[idx].argmax().item()) == y_n]
        if len(q_y) == 0:
            return self._select_entropy(k, state)

        cand_n = min(len(unlabeled_indices), self.max_pool_eval)
        candidates = np.random.choice(unlabeled_indices, cand_n, replace=False).tolist() if cand_n > 0 else []
        if len(candidates) == 0:
            return []

        device = next(substitute.parameters()).device
        rare_feats = self._extract_features_for_indices(q_y, substitute, device, state)
        pool_feats = self._extract_features_for_indices(candidates, substitute, device, state)
        if rare_feats.numel() == 0 or pool_feats.numel() == 0:
            return self._select_entropy(k, state)

        dists_sq = torch.cdist(pool_feats, rare_feats).pow(2)
        kernel = torch.exp(-dists_sq / (2.0 * (self.kde_sigma ** 2)))
        scores = kernel.sum(dim=1)

        topk = min(int(k), scores.numel())
        _, top_idx = torch.topk(scores, k=topk, largest=True)
        selected = [candidates[i] for i in top_idx.tolist()]
        return selected

    # -------------------------
    # Observe
    # -------------------------

    def observe(self, query_batch: QueryBatch, oracle_output: OracleOutput, state: BenchmarkState) -> None:
        x_batch = query_batch.x           # raw
        y_batch = oracle_output.y         # soft prob [B,K] or hard [B]

        state.attack_state["query_data_x"].append(x_batch.detach().cpu())
        state.attack_state["query_data_y"].append(y_batch.detach().cpu())

        indices = query_batch.meta.get("indices", [])
        if oracle_output.kind == "soft_prob":
            for i, idx in enumerate(indices):
                if int(idx) >= 0:
                    state.attack_state["victim_outputs"][int(idx)] = y_batch[i].detach().cpu()
            labels = [int(y.argmax().item()) for y in y_batch]
        else:
            num_classes = int(state.metadata.get("num_classes") or 10)
            for i, idx in enumerate(indices):
                if int(idx) >= 0:
                    lab = int(y_batch[i].item()) if y_batch[i].ndim == 0 else int(y_batch[i].argmax().item())
                    one_hot = torch.zeros(num_classes)
                    one_hot[lab] = 1.0
                    state.attack_state["victim_outputs"][int(idx)] = one_hot
            labels = [int(y.item()) if y.ndim == 0 else int(y.argmax().item()) for y in y_batch]

        for lab in labels:
            state.attack_state["class_counts"][lab] = state.attack_state["class_counts"].get(lab, 0) + 1

        labeled_count = len(state.attack_state["labeled_indices"])
        total_budget = int(state.metadata.get("max_budget") or self.config.get("max_budget", 10000))
        round_size = max(1, total_budget // max(self.I, 1))
        if labeled_count % round_size == 0 and labeled_count > 0:
            self.train_substitute(state)

    # -------------------------
    # Train: repo main_sup_soft 핵심 로직 적용
    # loss = loss1(unlabeled self) + loss2(soft sup on Q) + lambda2*loss3(cost-sensitive sharpness)
    # -------------------------

    def train_substitute(self, state: BenchmarkState) -> None:
        self._ensure_pool_dataset(state)

        qx = state.attack_state["query_data_x"]
        qy = state.attack_state["query_data_y"]
        if len(qx) == 0:
            return

        x_all = torch.cat(qx, dim=0)  # raw
        y_all = torch.cat(qy, dim=0)

        # dataset Q
        class QueryDataset(torch.utils.data.Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def __len__(self): return len(self.x)
            def __getitem__(self, idx): return self.x[idx], self.y[idx]

        dataset_q = QueryDataset(x_all, y_all)

        total_size = len(dataset_q)
        val_size = max(1, int(0.2 * total_size))
        train_size = total_size - val_size
        if train_size < 2:
            return

        train_q, val_q = torch.utils.data.random_split(
            dataset_q, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        labeled_loader = torch.utils.data.DataLoader(train_q, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_q, batch_size=256, shuffle=False, num_workers=0)

        # dataset U (unqueried pool) for self-supervised term
        unlabeled_indices = state.attack_state["unlabeled_indices"]
        if len(unlabeled_indices) == 0:
            # fallback: no U, use Q for self term (not paper-faithful, but safe)
            unlabeled_loader = None
        else:
            class PoolU(torch.utils.data.Dataset):
                def __init__(self, indices, pool):
                    self.indices = indices
                    self.pool = pool
                def __len__(self): return len(self.indices)
                def __getitem__(self, i):
                    idx = self.indices[i]
                    img, _ = self.pool[idx]
                    return img

            # cap U size for speed
            cap = min(len(unlabeled_indices), self.max_pool_eval)
            u_indices = np.random.choice(unlabeled_indices, cap, replace=False).tolist() if cap > 0 else []
            dataset_u = PoolU(u_indices, self.pool_dataset)
            unlabeled_loader = torch.utils.data.DataLoader(dataset_u, batch_size=min(self.batch_size, 128), shuffle=True, num_workers=0, drop_last=True)

        device = torch.device(state.metadata.get("device", "cpu"))
        num_classes = int(state.metadata.get("num_classes") or 10)

        # init / warm-start substitute
        substitute = state.attack_state.get("substitute")
        if not isinstance(substitute, SwiftThiefSubstitute):
            base = create_substitute(
                arch="resnet18",
                num_classes=num_classes,
                input_channels=state.metadata.get("input_shape", (3, 32, 32))[0],
            ).to(device)
            if not hasattr(base, "fc"):
                raise ValueError("SwiftThief expects a ResNet-style substitute with .fc")
            classifier = base.fc
            base.fc = nn.Identity()
            substitute = SwiftThiefSubstitute(base, classifier).to(device)
        else:
            substitute = substitute.to(device)

        # heads
        feat_dim = substitute.classifier.in_features
        if self.projection_head is None:
            self.projection_head = SimSiamProjectionHead(feat_dim, self.projection_dim).to(device)
        if self.predictor_head is None:
            self.predictor_head = SimSiamPredictorHead(self.projection_dim).to(device)

        self._ensure_normalizers(state, device)

        # costs (repo: effective-number based from current class histogram)
        # use counts from state.attack_state["class_counts"]
        cnt = torch.zeros(num_classes, device=device)
        for c in range(num_classes):
            cnt[c] = float(state.attack_state["class_counts"].get(c, 0))
        beta = 0.99
        costs = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta, device=device), cnt + 1.0))
        costs = costs / costs.sum().clamp_min(1e-12)

        # criteria (repo)
        criterion = SimSiamLoss('simplified').to(device)
        soft_criterion = SoftSupSimSiamLossV17(device, num_classes).to(device)
        cost_sensitive_criterion = SimSiamLoss_cost_sensitive(costs).to(device)

        # FGSM adversary (repo)
        fgsm_model = _SimSiamWrapper(substitute, self.projection_head, self.predictor_head).to(device)
        reg_adversary = CL_FGSM(fgsm_model, self.fgsm_epsilon, device)

        # optimizer (repo uses SGD for CL)
        optimizer = torch.optim.SGD(
            list(substitute.parameters()) + list(self.projection_head.parameters()) + list(self.predictor_head.parameters()),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        best_f1 = 0.0
        patience_counter = 0
        best_state = None

        # iterators
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None

        for epoch in range(self.cl_epochs):
            substitute.train()
            self.projection_head.train()
            self.predictor_head.train()

            # repo: unlabeled loader drives loop; we mimic
            steps = len(unlabeled_loader) if unlabeled_loader is not None else len(labeled_loader)

            for _ in range(steps):
                # -------- U batch --------
                if unlabeled_loader is not None:
                    try:
                        u_raw = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        u_raw = next(unlabeled_iter)
                    u_raw = u_raw.to(device)
                    u1_raw, u2_raw = self._apply_two_crops(u_raw, device, state)
                    u1, u2 = self.normalize_pair(u1_raw, u2_raw)
                    outs_u = fgsm_model(im_aug1=u1, im_aug2=u2)
                    loss1 = criterion(outs_u['z1'], outs_u['z2'], outs_u['p1'], outs_u['p2'])
                else:
                    loss1 = torch.zeros((), device=device)

                # -------- Q batch --------
                try:
                    x_raw, y = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    x_raw, y = next(labeled_iter)

                x_raw = x_raw.to(device)
                y = y.to(device)

                x1_raw, x2_raw = self._apply_two_crops(x_raw, device, state)
                x1, x2 = self.normalize_pair(x1_raw, x2_raw)

                # adversarial x1 (repo)
                adv_x1 = reg_adversary(x1, x2)

                # labeled forward (clean) for soft supervised
                outs_l = fgsm_model(im_aug1=x1, im_aug2=x2)

                # loss2: SoftSupSimSiamLossV17 expects probabilities
                # mebench soft_prob already probs; if hard labels, skip
                if y.ndim > 1 and y.shape[1] > 1:
                    targets = y.clamp_min(1e-8)
                    targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
                    loss2 = soft_criterion(
                        p=torch.cat([outs_l['p1'], outs_l['p2']], dim=0),
                        z=torch.cat([outs_l['z1'], outs_l['z2']], dim=0),
                        targets=torch.cat([targets, targets], dim=0)
                    )
                    y_idx = targets.argmax(dim=1)
                else:
                    loss2 = torch.zeros((), device=device)
                    y_idx = y.long()

                # loss3: sharpness minimization (cost-sensitive) on adversarial input
                outs_adv = fgsm_model(im_aug1=adv_x1, im_aug2=x2)
                loss3 = cost_sensitive_criterion(
                    outs_adv['z1'], outs_adv['z2'],
                    outs_adv['p1'], outs_adv['p2'],
                    y_idx
                )

                loss = loss1 + loss2 + self.lambda2 * loss3

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            # validation: use classifier head (requires normalized input)
            val_f1 = self._compute_f1(substitute, val_loader, device, state)
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_state = {
                    "sub": {k: v.detach().cpu().clone() for k, v in substitute.state_dict().items()},
                    "proj": {k: v.detach().cpu().clone() for k, v in self.projection_head.state_dict().items()},
                    "pred": {k: v.detach().cpu().clone() for k, v in self.predictor_head.state_dict().items()},
                }
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"[SwiftThief-CL] epoch={epoch} val_f1={val_f1:.4f}")

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            substitute.load_state_dict(best_state["sub"])
            self.projection_head.load_state_dict(best_state["proj"])
            self.predictor_head.load_state_dict(best_state["pred"])

        state.attack_state["substitute"] = substitute
        print(f"SwiftThief substitute trained (CL stage). Best F1: {best_f1:.4f}")

    def _compute_f1(self, model: nn.Module, val_loader: DataLoader, device: torch.device, state: BenchmarkState) -> float:
        model.eval()
        self._ensure_normalizers(state, device)

        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_raw, y in val_loader:
                x_raw = x_raw.to(device)
                x = self.normalize(x_raw)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                if y.ndim > 1:
                    targets = torch.argmax(y, dim=1).cpu().numpy()
                else:
                    targets = y.cpu().numpy()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")

