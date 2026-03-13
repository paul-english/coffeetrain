"""TrainerModel protocol and mixin."""

from typing import Any, Dict, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class TrainerModel(Protocol):
    """Protocol for models compatible with Trainer.

    Models must implement:
    - forward(batch) -> Dict: Run forward pass, return outputs dict
    - loss(outputs, batch) -> Tensor: Compute loss from outputs

    The Trainer calls these methods and manages the training loop.

    Example:
        class MyModel(nn.Module, TrainerModel):
            def forward(self, batch):
                x = batch['input_ids']
                logits = self.linear(x)
                return {'logits': logits}

            def loss(self, outputs, batch):
                return F.cross_entropy(outputs['logits'], batch['labels'])
    """

    def forward(self, batch: Any) -> Dict[str, Any]:
        """Run forward pass on a batch.

        Args:
            batch: Batch data from dataloader

        Returns:
            Dictionary of outputs (must include data needed for loss)
        """
        ...

    def loss(self, outputs: Dict[str, Any], batch: Any) -> Tensor:
        """Compute loss from forward outputs.

        Args:
            outputs: Dictionary from forward()
            batch: Original batch data

        Returns:
            Loss tensor for backpropagation
        """
        ...


class TrainerModelMixin:
    """Mixin providing common model wrapper functionality.

    Optional methods that enhance integration with Trainer.
    Subclass this along with your model for extra features.

    Example:
        class MyModel(nn.Module, TrainerModelMixin):
            def forward(self, batch): ...
            def loss(self, outputs, batch): ...

            def get_metrics(self, is_train):
                return {'accuracy': Accuracy()}
    """

    def get_metrics(self, is_train: bool = False) -> Dict[str, Any]:
        """Get metrics for training or evaluation.

        Args:
            is_train: True for training metrics, False for eval

        Returns:
            Dictionary of metric name to metric object
        """
        return {}

    def update_metric(self, batch: Any, outputs: Dict[str, Any], metric: Any) -> None:
        """Update a metric with batch outputs.

        Args:
            batch: Input batch
            outputs: Model outputs
            metric: Metric object to update
        """
        pass

    def eval_forward(self, batch: Any, outputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Forward pass for evaluation.

        Can be overridden for eval-specific behavior (e.g., no dropout).

        Args:
            batch: Input batch
            outputs: Optional outputs from previous forward (reused if provided)

        Returns:
            Model outputs
        """
        if outputs is not None:
            return outputs
        return self.forward(batch)


class ModelWrapper(torch.nn.Module, TrainerModelMixin):
    """Base class for model wrappers.

    Wraps an existing model to add TrainerModel interface.

    Example:
        class MyWrapper(ModelWrapper):
            def __init__(self, model):
                super().__init__(model)

            def forward(self, batch):
                return self.model(batch['input_ids'])

            def loss(self, outputs, batch):
                return F.mse_loss(outputs, batch['target'])
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize wrapper with underlying model.

        Args:
            model: The model to wrap
        """
        super().__init__()
        self.model = model

    def forward(self, batch: Any) -> Dict[str, Any]:
        """Must be implemented by subclass."""
        raise NotImplementedError

    def loss(self, outputs: Dict[str, Any], batch: Any) -> Tensor:
        """Must be implemented by subclass."""
        raise NotImplementedError
