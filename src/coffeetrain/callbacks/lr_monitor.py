"""Learning rate monitoring callback."""

from typing import Any, Dict

from coffeetrain.callback import Callback
from coffeetrain.state import State


class LRMonitor(Callback):
    """Monitor and log learning rates.

    Logs learning rates for all parameter groups at configurable intervals.

    Attributes:
        log_interval: Print every N batches (0 = epoch end only)
    """

    def __init__(self, log_interval: int = 0):
        """Initialize LR monitor.

        Args:
            log_interval: Print every N batches (0 = epoch end only)
        """
        self.log_interval = log_interval
        self._batch_count = 0

    def batch_end(self, state: State) -> None:
        """Log LR at interval."""
        self._batch_count += 1

        if self.log_interval > 0 and self._batch_count % self.log_interval == 0:
            lrs = state.get_all_lrs()
            lr_str = ", ".join(f"{k}={v:.2e}" for k, v in lrs.items())
            print(f"  LR: {lr_str}")

    def epoch_end(self, state: State) -> None:
        """Log LR at epoch end."""
        lrs = state.get_all_lrs()
        state.train_metrics['learning_rate'] = state.get_lr()
        state.train_metrics['learning_rates'] = lrs

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {'batch_count': self._batch_count}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._batch_count = state_dict['batch_count']
