"""Early stopping callback for fine-tuning."""

from typing import Any, Dict

from coffeetrain.callback import Callback
from coffeetrain.state import State


class EarlyStoppingCallback(Callback):
    """Early stopping for fine-tuning with EMA-smoothed validation metrics.

    Uses exponential moving average to smooth noisy validation metrics,
    preventing premature stopping due to random fluctuations.

    Args:
        metric: Name of metric in state.eval_metrics to monitor
        mode: 'min' or 'max' - whether lower or higher is better
        patience: Number of epochs without improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        smoothing: EMA alpha (0-1). Higher = more responsive, lower = more stable
    """

    def __init__(
        self,
        metric: str = "entity_f1",
        mode: str = "max",
        patience: int = 3,
        min_delta: float = 0.0,
        smoothing: float = 0.3,
    ):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing = smoothing

        self.smoothed_value: float | None = None
        self.best_value: float | None = None
        self.epochs_without_improvement: int = 0
        self.stopped_epoch: int | None = None

    def _is_improvement(self, smoothed: float) -> bool:
        """Check if smoothed value is an improvement over best."""
        if self.best_value is None:
            return True
        if self.mode == "min":
            return smoothed < self.best_value - self.min_delta
        return smoothed > self.best_value + self.min_delta

    def eval_end(self, state: State) -> None:
        """Check for early stopping after evaluation."""
        current = state.eval_metrics.get(self.metric)
        if current is None:
            return

        # EMA smoothing
        if self.smoothed_value is None:
            self.smoothed_value = current
        else:
            self.smoothed_value = (
                self.smoothing * current
                + (1 - self.smoothing) * self.smoothed_value
            )

        if self._is_improvement(self.smoothed_value):
            self.best_value = self.smoothed_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        if self.epochs_without_improvement >= self.patience:
            self.stopped_epoch = state.epoch
            state.stop_training = True
            print(f"Early stopping at epoch {state.epoch + 1}")
            print(f"Best smoothed {self.metric}: {self.best_value:.6f}")

    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing."""
        return {
            "smoothed_value": self.smoothed_value,
            "best_value": self.best_value,
            "epochs_without_improvement": self.epochs_without_improvement,
            "stopped_epoch": self.stopped_epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore callback state from checkpoint."""
        self.smoothed_value = state_dict["smoothed_value"]
        self.best_value = state_dict["best_value"]
        self.epochs_without_improvement = state_dict["epochs_without_improvement"]
        self.stopped_epoch = state_dict["stopped_epoch"]
