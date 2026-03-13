"""Progress display callback."""

from typing import Any, Dict

from coffeetrain.callback import Callback
from coffeetrain.state import State


class ProgressCallback(Callback):
    """Display training progress.

    Prints epoch summaries and key metrics. Use with Trainer(progress_bar=False)
    for a cleaner log-style output.

    Attributes:
        print_batch_loss: Print loss for each batch
        batch_log_interval: Print every N batches
    """

    def __init__(
        self,
        print_batch_loss: bool = False,
        batch_log_interval: int = 100,
    ):
        """Initialize progress callback.

        Args:
            print_batch_loss: Print individual batch losses
            batch_log_interval: Batch print interval
        """
        self.print_batch_loss = print_batch_loss
        self.batch_log_interval = batch_log_interval

    def fit_start(self, state: State) -> None:
        """Print training start."""
        print("\n" + "=" * 60)
        print(f"Starting training for {state.max_epochs} epochs")
        print(f"Training batches: {state.num_batches}")
        print(f"Device: {state.device}")
        print("=" * 60)

    def epoch_start(self, state: State) -> None:
        """Print epoch start."""
        print(f"\nEpoch {state.epoch + 1}/{state.max_epochs}")
        print("-" * 40)

    def batch_end(self, state: State) -> None:
        """Print batch progress."""
        if not self.print_batch_loss:
            return

        if (state.batch_idx + 1) % self.batch_log_interval == 0:
            loss = state.loss.item() if state.loss is not None else 0.0
            lr = state.get_lr()
            print(f"  Batch {state.batch_idx + 1}/{state.num_batches}: loss={loss:.4f}, lr={lr:.2e}")

    def epoch_end(self, state: State) -> None:
        """Print epoch summary."""
        metrics = state.train_metrics
        parts = []

        if 'loss' in metrics or state.loss is not None:
            loss = metrics.get('loss', state.loss.item() if state.loss else 0.0)
            parts.append(f"loss={loss:.4f}")

        if 'learning_rate' in metrics:
            parts.append(f"lr={metrics['learning_rate']:.2e}")

        if parts:
            print(f"  Train: {', '.join(parts)}")

    def eval_end(self, state: State) -> None:
        """Print evaluation summary."""
        metrics = state.eval_metrics
        if not metrics:
            return

        parts = [f"{k}={v:.4f}" for k, v in metrics.items()]
        print(f"  Val: {', '.join(parts)}")

    def fit_end(self, state: State) -> None:
        """Print training complete."""
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Total steps: {state.global_step}")
        print("=" * 60 + "\n")

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        pass
