"""Query data storage for Track A training caching."""

from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
import pickle
import shutil


class QueryStorage(Dataset):
    """Storage for queried data (x, y) tensors for Track A training.

    This provides:
    1. Batch-level append during oracle queries
    2. Random access for DataLoader during training
    3. Automatic cleanup on completion
    4. Run-scoped directory structure

    Storage format:
    - queries.pt: Concatenated tensor [N, C, H, W] of all images
    - labels.pt: Tensor [N] of oracle outputs (labels or probs)

    For soft_prob mode: labels stores probability distributions [N, K]
    For hard_top1 mode: labels stores class indices [N]
    """

    def __init__(self, cache_dir: Path, output_mode: str = "soft_prob"):
        """Initialize query storage.

        Args:
            cache_dir: Directory for storage files
            output_mode: Oracle output mode (soft_prob or hard_top1)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_mode = output_mode

        # Storage tensors
        self.queries: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.count = 0

    def add_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor) -> None:
        """Append a batch of queries to storage.

        Args:
            x_batch: Query images [B, C, H, W]
            y_batch: Oracle outputs [B] or [B, K]
        """
        # Move to CPU for storage
        x_cpu = x_batch.detach().cpu()
        y_cpu = y_batch.detach().cpu()

        # Initialize storage tensors
        if self.queries is None:
            self.queries = x_cpu
            self.labels = y_cpu
        else:
            # Concatenate with existing storage
            self.queries = torch.cat([self.queries, x_cpu], dim=0)
            self.labels = torch.cat([self.labels, y_cpu], dim=0)

        self.count += x_batch.shape[0]

    def save(self) -> None:
        """Save current storage to disk."""
        if self.queries is None or self.labels is None:
            return

        queries_path = self.cache_dir / "queries.pt"
        labels_path = self.cache_dir / "labels.pt"
        meta_path = self.cache_dir / "meta.pkl"

        # Save tensors
        torch.save(self.queries, queries_path)
        torch.save(self.labels, labels_path)

        # Save metadata
        meta = {
            "count": self.count,
            "output_mode": self.output_mode,
            "queries_shape": self.queries.shape,
            "labels_shape": self.labels.shape,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)

        print(f"Saved {self.count} queries to {self.cache_dir}")

    def load(self) -> None:
        """Load storage from disk."""
        queries_path = self.cache_dir / "queries.pt"
        labels_path = self.cache_dir / "labels.pt"
        meta_path = self.cache_dir / "meta.pkl"

        if not queries_path.exists() or not labels_path.exists():
            return

        # Load tensors
        self.queries = torch.load(queries_path, weights_only=True)
        self.labels = torch.load(labels_path, weights_only=True)

        # Load metadata
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.count = meta["count"]
        self.output_mode = meta.get("output_mode", self.output_mode)

        print(f"Loaded {self.count} queries from {self.cache_dir}")

    def cleanup(self) -> None:
        """Remove all storage files."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"Cleaned up cache at {self.cache_dir}")

    def __len__(self) -> int:
        """Return number of stored queries."""
        return self.count

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single query by index.

        Args:
            idx: Query index

        Returns:
            (x, y) tuple
        """
        if self.queries is None or self.labels is None:
            raise RuntimeError("QueryStorage not loaded. Call load() first.")

        return self.queries[idx], self.labels[idx]


def create_query_storage(
    run_dir: Path,
    output_mode: str = "soft_prob",
) -> QueryStorage:
    """Create query storage in run directory.

    Args:
        run_dir: Run directory path
        output_mode: Oracle output mode

    Returns:
        QueryStorage instance
    """
    cache_dir = run_dir / "query_cache"
    return QueryStorage(cache_dir, output_mode)
