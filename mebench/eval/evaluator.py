"""Track A evaluator: trains substitute from scratch at each checkpoint."""

from typing import Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mebench.models.substitute_factory import create_substitute
from mebench.eval.metrics import evaluate_substitute


class Evaluator:
    """Evaluates benchmark state using Track A protocol."""

    def __init__(self, config: dict, state: Any, query_storage: Any):
        """Initialize evaluator.

        Args:
            config: Full experiment configuration
            state: BenchmarkState object
            query_storage: QueryStorage object
        """
        self.config = config
        self.state = state
        self.query_storage = query_storage
        self.device = state.metadata["device"]

    def evaluate(
        self,
        victim: nn.Module,
        test_loader: DataLoader,
        checkpoint_budget: int,
    ) -> Dict[str, Dict[str, float]]:
        """Perform Track A evaluation.

        Args:
            victim: The victim model
            test_loader: DataLoader for test set
            checkpoint_budget: Current query budget

        Returns:
            Dictionary with results (track_a)
        """
        # Load query data
        query_loader = self.query_storage.get_dataloader(batch_size=128)

        # 1. Setup substitute
        sub_config = self.config["substitute"]
        num_classes = int(self.config["victim"]["num_classes"])
        input_channels = int(self.config["victim"]["channels"])

        # Reset RNG for reproducibility
        torch.manual_seed(sub_config.get("init_seed", 42))

        substitute = create_substitute(
            arch=sub_config["arch"],
            num_classes=num_classes,
            input_channels=input_channels,
        ).to(self.device)

        # 2. Train from scratch
        self._train_track_a(substitute, query_loader, checkpoint_budget)

        # 3. Evaluate
        output_mode = self.config["victim"]["output_mode"]
        temperature = self.config["victim"]["temperature"]

        metrics = evaluate_substitute(
            substitute=substitute,
            victim=victim,
            test_loader=test_loader,
            device=self.device,
            output_mode=output_mode,
            temperature=temperature,
        )

        return {"track_a": metrics}

    def _train_track_a(
        self, model: nn.Module, train_loader: DataLoader, checkpoint_budget: int
    ) -> None:
        """Standard training protocol for Track A.

        Args:
            model: Substitute model to train
            train_loader: Collected (x, y_oracle) data
            checkpoint_budget: Budget at this checkpoint
        """
        # 1. Optimizer and Scheduler
        opt_config = self.config["substitute"]["optimizer"]
        patience = self.config["substitute"].get("patience", 100)

        import torch.optim as optim
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(opt_config.get("lr", 0.01)),
            momentum=float(opt_config.get("momentum", 0.9)),
            weight_decay=float(opt_config.get("weight_decay", 5e-4)),
        )

        # Get training steps: S(B) = ceil(0.2 × B)
        steps_coeff = self.config["substitute"]["trackA"]["steps_coeff_c"]
        num_steps = int(steps_coeff * checkpoint_budget + 0.9999)

        # 2. Loss function
        output_mode = self.config["victim"]["output_mode"]

        if output_mode == "soft_prob":
            criterion = nn.KLDivLoss(reduction="batchmean")
        else:
            criterion = nn.CrossEntropyLoss()

        # 3. Validation set (20% of collected data)
        # For small budgets, use full data for training
        val_loader = train_loader

        # 4. Training loop
        model.train()
        current_step = 0
        best_f1 = -1.0
        best_model_state = None
        patience_counter = 0
        done = False

        while not done and current_step < num_steps:
            for x_batch, y_batch in train_loader:
                if current_step >= num_steps:
                    break

                x_batch = x_batch.to(self.device)
                # Contract: Inputs are already in [0, 1].

                # Handle output modes
                if output_mode == "soft_prob":
                    # Soft labels: convert to log probs for KL loss
                    y_batch = y_batch.to(self.device)
                    # Clip probabilities to avoid log(0) and normalize
                    y_batch = torch.clamp(y_batch, min=1e-10)
                    y_batch = y_batch / y_batch.sum(dim=1, keepdim=True)
                    
                    outputs = model(x_batch)
                    # Ensure outputs are log probabilities
                    log_outputs = torch.log_softmax(outputs, dim=1)
                    loss = criterion(log_outputs, y_batch)
                else:
                    # Hard labels: standard cross entropy
                    y_batch = y_batch.long().to(self.device)
                    outputs = model(x_batch)
                    loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent NaNs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                current_step += 1

                # Validate every 100 steps or at end
                if current_step % 100 == 0 or current_step == num_steps:
                    val_f1 = self._compute_f1(model, val_loader, output_mode)

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        patience_counter = 0
                        best_model_state = {
                            k: v.cpu().clone() for k, v in model.state_dict().items()
                        }
                    else:
                        patience_counter += 1

                    if patience_counter >= patience:
                        print(f"Early stopping at step {current_step}")
                        done = True
                        break

        # Load best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Track A training complete. Best F1: {best_f1:.4f}")
        else:
            print(f"Track A training complete. No validation improvement.")

    def _compute_f1(self, model: nn.Module, val_loader: DataLoader, output_mode: str) -> float:
        """Compute F1 score on validation set."""
        from sklearn.metrics import f1_score
        import numpy as np

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                # Contract: Inputs in [0, 1].
                outputs = model(x_batch)

                # Get predictions
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)

                # Get true labels
                if output_mode == "soft_prob":
                    targets = torch.argmax(y_batch, dim=1).cpu().numpy()
                else:
                    targets = y_batch.cpu().numpy()
                all_targets.extend(targets)

        return f1_score(all_targets, all_preds, average="macro")
