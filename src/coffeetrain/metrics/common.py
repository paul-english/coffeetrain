"""Common metrics for pre-training and general use."""

from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy


class SafeMulticlassAccuracy(MulticlassAccuracy):
    """MulticlassAccuracy that handles compute() before any update() calls.

    Torchmetrics warns when compute() is called before update(). This wrapper
    tracks whether any updates have occurred and returns 0.0 if not, avoiding
    the warning on the first epoch when metrics may be computed before training
    batches complete.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("_safe_update_count", default=torch.tensor(0), dist_reduce_fx="sum")
        # Set torchmetrics internal counter to bypass no-update warning
        # We handle the no-update case ourselves in compute()
        object.__setattr__(self, "_update_count", torch.tensor(1))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state with predictions and targets."""
        self._safe_update_count += 1
        super().update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute accuracy, returning 0.0 if no updates have occurred."""
        if self._safe_update_count == 0:
            return torch.tensor(0.0, device=self._safe_update_count.device)
        return super().compute()


class MeanLoss(Metric):
    """Track mean loss across batches.

    Simple metric that accumulates loss values and computes the mean.
    Useful for tracking training loss in train_metrics.
    """

    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor) -> None:
        """Update with a batch loss value.

        Args:
            loss: Loss tensor (scalar)
        """
        self.total_loss += loss.detach()
        self.count += 1

    def compute(self) -> torch.Tensor:
        """Compute mean loss."""
        if self.count == 0:
            return torch.tensor(0.0, device=self.total_loss.device)
        return self.total_loss / self.count


class Perplexity(Metric):
    """Perplexity metric for language modeling.

    Computes perplexity as exp(avg_loss) from accumulated loss values.
    """

    full_state_update: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor) -> None:
        """Update with a batch loss value.

        Args:
            loss: Loss tensor (scalar or per-sample losses)
        """
        if loss.dim() == 0:
            # Scalar loss
            self.total_loss += loss.detach()
            self.count += 1
        else:
            # Per-sample losses
            self.total_loss += loss.detach().sum()
            self.count += loss.numel()

    def compute(self) -> torch.Tensor:
        """Compute perplexity as exp(avg_loss)."""
        if self.count == 0:
            return torch.tensor(float("inf"), device=self.total_loss.device)
        avg_loss = self.total_loss / self.count
        return torch.exp(avg_loss)
