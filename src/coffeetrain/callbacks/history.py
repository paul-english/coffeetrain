"""Training history logging callback."""

import json
from pathlib import Path
from typing import Any, Dict, List

from coffeetrain.callback import Callback
from coffeetrain.state import State


class HistoryCallback(Callback):
    """Track and save training history to JSON.

    Maintains backward compatibility with existing history.json format.

    Saved metrics:
    - train_loss: Average training loss per epoch
    - val_loss: Validation loss per epoch
    - val_classification_loss: Classification component loss
    - val_structure_loss: Structure prediction loss
    - val_count_loss: Count prediction loss
    - learning_rates: Learning rate per epoch

    Attributes:
        save_dir: Directory to save history.json
        history: Dictionary of metric lists
    """

    def __init__(self, save_dir: Path | str):
        """Initialize history callback.

        Args:
            save_dir: Directory to save history.json
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'val_classification_loss': [],
            'val_structure_loss': [],
            'val_count_loss': [],
            'learning_rates': [],
        }

        # Epoch accumulators
        self._epoch_loss_sum: float = 0.0
        self._epoch_batch_count: int = 0

    def epoch_start(self, state: State) -> None:
        """Reset epoch accumulators."""
        self._epoch_loss_sum = 0.0
        self._epoch_batch_count = 0

    def batch_end(self, state: State) -> None:
        """Accumulate batch loss."""
        if state.loss is not None:
            self._epoch_loss_sum += state.loss.item()
            self._epoch_batch_count += 1

    def epoch_end(self, state: State) -> None:
        """Record training metrics at epoch end."""
        # Average training loss
        if self._epoch_batch_count > 0:
            avg_loss = self._epoch_loss_sum / self._epoch_batch_count
            self.history['train_loss'].append(avg_loss)

        # Learning rate
        lr = state.get_lr()
        self.history['learning_rates'].append(lr)

        # Save after each epoch
        self._save_history()

    def eval_end(self, state: State) -> None:
        """Record validation metrics."""
        eval_metrics = state.eval_metrics

        if 'loss' in eval_metrics:
            self.history['val_loss'].append(eval_metrics['loss'])

        # Component losses (for models that provide them)
        for key in ['classification_loss', 'structure_loss', 'count_loss']:
            if key in eval_metrics:
                self.history[f'val_{key}'].append(eval_metrics[key])

        self._save_history()

    def _save_history(self) -> None:
        """Save history to JSON file."""
        history_path = self.save_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            'history': self.history,
            'epoch_loss_sum': self._epoch_loss_sum,
            'epoch_batch_count': self._epoch_batch_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.history = state_dict['history']
        self._epoch_loss_sum = state_dict['epoch_loss_sum']
        self._epoch_batch_count = state_dict['epoch_batch_count']
