"""Training speed monitoring callback."""

import time
from typing import Any, Dict

from coffeetrain.callback import Callback
from coffeetrain.state import State


class SpeedMonitor(Callback):
    """Monitor training speed (samples per second).

    Tracks throughput and timing for performance analysis.

    Attributes:
        window_size: Number of batches to average over
    """

    def __init__(self, window_size: int = 50, log_interval: int = 0):
        """Initialize speed monitor.

        Args:
            window_size: Batches to average for speed calculation
            log_interval: Print every N batches (0 = epoch end only)
        """
        self.window_size = window_size
        self.log_interval = log_interval

        self._batch_times: list = []
        self._batch_sizes: list = []
        self._epoch_start_time: float = 0.0
        self._batch_start_time: float = 0.0

    def epoch_start(self, state: State) -> None:
        """Record epoch start time."""
        self._epoch_start_time = time.time()
        self._batch_times = []
        self._batch_sizes = []

    def batch_start(self, state: State) -> None:
        """Record batch start time."""
        self._batch_start_time = time.time()

    def batch_end(self, state: State) -> None:
        """Record batch timing."""
        batch_time = time.time() - self._batch_start_time

        # Get batch size
        batch = state.batch
        if isinstance(batch, dict) and 'input_ids' in batch:
            batch_size = batch['input_ids'].size(0)
        elif isinstance(batch, (list, tuple)):
            batch_size = len(batch)
        else:
            batch_size = 1

        self._batch_times.append(batch_time)
        self._batch_sizes.append(batch_size)

        # Keep window
        if len(self._batch_times) > self.window_size:
            self._batch_times = self._batch_times[-self.window_size:]
            self._batch_sizes = self._batch_sizes[-self.window_size:]

        # Log if interval
        if self.log_interval > 0 and len(self._batch_times) % self.log_interval == 0:
            samples_per_sec = self._compute_samples_per_sec()
            print(f"  Speed: {samples_per_sec:.1f} samples/sec")

    def epoch_end(self, state: State) -> None:
        """Log epoch timing."""
        epoch_time = time.time() - self._epoch_start_time
        samples_per_sec = self._compute_samples_per_sec()

        state.train_metrics['epoch_time'] = epoch_time
        state.train_metrics['samples_per_sec'] = samples_per_sec

        print(f"  Epoch time: {epoch_time:.1f}s, {samples_per_sec:.1f} samples/sec")

    def _compute_samples_per_sec(self) -> float:
        """Compute samples per second over window."""
        if not self._batch_times:
            return 0.0
        total_time = sum(self._batch_times)
        total_samples = sum(self._batch_sizes)
        if total_time == 0:
            return 0.0
        return total_samples / total_time

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        pass
