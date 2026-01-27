"""Oracle wrapper for victim model inference."""

import torch
import torch.nn as nn
from mebench.core.types import OracleOutput


class Oracle:
    """Wrapper for victim model that enforces budget and output constraints."""

    def __init__(self, model: nn.Module, config: dict):
        """Initialize oracle.

        Args:
            model: The victim model (pre-trained)
            config: Oracle/Victim configuration
        """
        self.model = model
        self.model.eval()
        self.config = config

        self.output_mode = config.get("output_mode", "soft_prob")
        self.temperature = float(config.get("temperature", 1.0))
        self.input_shape = tuple(config.get("input_size", (3, 32, 32)))
        if len(self.input_shape) == 2:
             # Add channel if missing
             self.input_shape = (config.get("channels", 1), *self.input_shape)

        self.budget_consumed = 0

    @torch.no_grad()
    def query(self, x_batch: torch.Tensor) -> OracleOutput:
        """Query victim model with a batch of images.

        Args:
            x_batch: Input tensor of shape (N, C, H, W) or (N, D).
                Assumed to be in [0, 1] scale.

        Returns:
            OracleOutput container
        """
        # Ensure model is on the same device
        device = next(self.model.parameters()).device
        x_batch = x_batch.to(device)

        # Increment budget by number of images
        self.budget_consumed += x_batch.size(0)

        # Normalize inputs to match victim's channels/size: reshape if needed
        # Contract: Assume x_batch is in [0, 1]. No additional normalization.
        x_reshaped = x_batch.view(x_batch.size(0), *self.input_shape)

        # Forward pass
        logits = self.model(x_reshaped)

        # Apply temperature
        logits = logits / self.temperature

        if self.output_mode == "soft_prob":
            # Soft softmax probabilities
            probs = torch.softmax(logits, dim=1)
            return OracleOutput(kind="soft_prob", y=probs.cpu())
        elif self.output_mode == "hard_top1":
            # Hard class label (top-1)
            labels = torch.argmax(logits, dim=1)
            return OracleOutput(kind="hard_top1", y=labels.cpu())
        else:
            raise ValueError(f"Unsupported output mode: {self.output_mode}")
