"""TorchMetrics integration callback."""

from typing import Any, Dict, Union

import torch
from torchmetrics import Metric

from coffeetrain.callback import Callback
from coffeetrain.state import State


class TorchMetricsCallback(Callback):
    """Callback for integrating torchmetrics with the trainer.

    Handles metric lifecycle:
    - Initialization at fit_start (gets metrics from model.get_metrics())
    - Updates after each batch (calls model.update_metric())
    - Computation at epoch_end/eval_end (stores in state metrics)
    - Reset at epoch_start/eval_start

    The callback checks if the model implements get_metrics() and update_metric()
    from TrainerModelMixin. If not, it gracefully no-ops.

    Supports both scalar metrics and dict-returning metrics (like NERMetrics).

    Args:
        prefix: Prefix for metric names (default: "" for no prefix)
    """

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.train_metrics: Dict[str, Metric] = {}
        self.eval_metrics: Dict[str, Metric] = {}

    def _store_metric_result(
        self,
        result: Union[torch.Tensor, Dict[str, torch.Tensor]],
        name: str,
        target_dict: Dict[str, float],
    ) -> None:
        """Store metric result, handling both scalar and dict returns.

        Args:
            result: Metric compute() result (scalar tensor or dict of tensors)
            name: Base metric name
            target_dict: Dict to store results in (train_metrics or eval_metrics)
        """
        key_prefix = f"{self.prefix}{name}" if self.prefix else name

        if isinstance(result, dict):
            # Dict of metrics (e.g., from NERMetrics)
            for metric_name, value in result.items():
                full_key = f"{key_prefix}/{metric_name}"
                if isinstance(value, torch.Tensor):
                    target_dict[full_key] = value.item()
                else:
                    target_dict[full_key] = float(value)
        elif isinstance(result, torch.Tensor):
            # Scalar tensor
            target_dict[key_prefix] = result.item()
        else:
            # Fallback for other types
            target_dict[key_prefix] = float(result)

    def fit_start(self, state: State) -> None:
        """Initialize metrics from model."""
        model = state.model

        if hasattr(model, "get_metrics"):
            self.train_metrics = model.get_metrics(is_train=True)
            self.eval_metrics = model.get_metrics(is_train=False)

            for metric in self.train_metrics.values():
                metric.to(state.device)
            for metric in self.eval_metrics.values():
                metric.to(state.device)

    def epoch_start(self, state: State) -> None:
        """Reset training metrics at epoch start."""
        for metric in self.train_metrics.values():
            metric.reset()

    def after_forward(self, state: State) -> None:
        """Update training metrics after forward pass."""
        if not self.train_metrics:
            return

        model = state.model
        if hasattr(model, "update_metric"):
            for name, metric in self.train_metrics.items():
                model.update_metric(state.batch, state.outputs, metric)

    def epoch_end(self, state: State) -> None:
        """Compute and store training metrics."""
        for name, metric in self.train_metrics.items():
            result = metric.compute()
            self._store_metric_result(result, name, state.train_metrics)

    def eval_start(self, state: State) -> None:
        """Reset eval metrics at evaluation start."""
        for metric in self.eval_metrics.values():
            metric.reset()

    def eval_after_forward(self, state: State) -> None:
        """Update eval metrics after forward pass."""
        if not self.eval_metrics:
            return

        model = state.model

        # Use eval_forward if available for richer outputs (e.g., entity extraction)
        if hasattr(model, "eval_forward"):
            outputs = model.eval_forward(state.batch, state.outputs)
            state.outputs = outputs

        if hasattr(model, "update_metric"):
            for name, metric in self.eval_metrics.items():
                model.update_metric(state.batch, state.outputs, metric)

    def eval_end(self, state: State) -> None:
        """Compute and store eval metrics."""
        for name, metric in self.eval_metrics.items():
            result = metric.compute()
            self._store_metric_result(result, name, state.eval_metrics)

    def state_dict(self) -> Dict[str, Any]:
        """Save metric states for checkpointing."""
        return {
            "train_metrics": {
                k: v.state_dict() for k, v in self.train_metrics.items()
            },
            "eval_metrics": {k: v.state_dict() for k, v in self.eval_metrics.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore metric states from checkpoint."""
        if "train_metrics" in state_dict:
            for k, v in state_dict["train_metrics"].items():
                if k in self.train_metrics:
                    self.train_metrics[k].load_state_dict(v)
        if "eval_metrics" in state_dict:
            for k, v in state_dict["eval_metrics"].items():
                if k in self.eval_metrics:
                    self.eval_metrics[k].load_state_dict(v)
