"""Best model checkpointing callback."""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from coffeetrain.callback import Callback
from coffeetrain.state import State


class BestModelCheckpointer(Callback):
    """Save best model based on validation metric.

    Monitors a validation metric and saves checkpoint when it improves.

    Supports two save modes:
    1. PyTorch state_dict (default): Saves model.state_dict() to a .pt file
    2. HuggingFace format: Calls save_pretrained_fn(save_dir) to save config + weights

    Attributes:
        save_dir: Directory to save checkpoints
        metric_name: Name of metric to monitor
        mode: 'min' for loss, 'max' for accuracy
        best_value: Current best metric value
    """

    def __init__(
        self,
        save_dir: Path | str,
        metric_name: str = 'loss',
        mode: str = 'min',
        filename: str = 'best_model.pt',
        save_pretrained_fn: Optional[Callable[[Path], None]] = None,
        save_dirname: str = 'best_model',
        track_train_metrics: bool = False,
        save_steps: Optional[int] = None,
    ):
        """Initialize checkpointer.

        Args:
            save_dir: Directory to save checkpoint
            metric_name: Metric to monitor (from state.eval_metrics or state.train_metrics)
            mode: 'min' if lower is better, 'max' if higher is better
            filename: Checkpoint filename (used when save_pretrained_fn is None)
            save_pretrained_fn: Optional function to save in HuggingFace format.
                If provided, called with save directory path instead of torch.save().
                Signature: fn(save_dir: Path) -> None
            save_dirname: Directory name for HF format saves (used with save_pretrained_fn)
            track_train_metrics: If True, also check train_metrics at epoch_end.
                Useful for pre-training when there's no validation.
            save_steps: If set, check for best model every N global steps using
                running average loss. Useful for long epochs (e.g. pre-training).
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode
        self.filename = filename
        self.save_pretrained_fn = save_pretrained_fn
        self.save_dirname = save_dirname
        self.track_train_metrics = track_train_metrics
        self.save_steps = save_steps

        self.best_value: float = float('inf') if mode == 'min' else float('-inf')
        self._best_epoch: Optional[int] = None
        self._step_loss_sum: float = 0.0
        self._step_loss_count: int = 0

    def batch_end(self, state: State) -> None:
        """Check for best model at step intervals using running average loss."""
        if self.save_steps is None:
            return
        if not state.is_main_process:
            return

        # Accumulate loss
        if state.loss is not None:
            self._step_loss_sum += state.loss.item()
            self._step_loss_count += 1

        # Check every save_steps global steps
        if state.global_step > 0 and state.global_step % self.save_steps == 0 and self._step_loss_count > 0:
            avg_loss = self._step_loss_sum / self._step_loss_count
            self._step_loss_sum = 0.0
            self._step_loss_count = 0

            is_best = self._is_improvement(avg_loss)
            if is_best:
                self.best_value = avg_loss
                self._best_epoch = state.epoch
                self._save_checkpoint(state)
                print(f"  New best train {self.metric_name}: {avg_loss:.4f} (step {state.global_step})")

    def epoch_end(self, state: State) -> None:
        """Check if current model is best based on train metrics."""
        if not self.track_train_metrics:
            return

        # Only save on main process in distributed training
        if not state.is_main_process:
            return

        current = state.train_metrics.get(self.metric_name)
        if current is None:
            return

        is_best = self._is_improvement(current)

        if is_best:
            self.best_value = current
            self._best_epoch = state.epoch
            self._save_checkpoint(state)
            print(f"  New best train {self.metric_name}: {current:.4f} (epoch {state.epoch + 1})")

    def eval_end(self, state: State) -> None:
        """Check if current model is best and save."""
        # Only save on main process in distributed training
        if not state.is_main_process:
            return

        current = state.eval_metrics.get(self.metric_name)
        if current is None:
            return

        is_best = self._is_improvement(current)

        if is_best:
            self.best_value = current
            self._best_epoch = state.epoch
            self._save_checkpoint(state)
            print(f"  New best {self.metric_name}: {current:.4f} (epoch {state.epoch + 1})")

    def _is_improvement(self, current: float) -> bool:
        """Check if current value improves on best."""
        if self.mode == 'min':
            return current < self.best_value
        return current > self.best_value

    def _save_checkpoint(self, state: State) -> None:
        """Save the best model checkpoint."""
        if self.save_pretrained_fn is not None:
            # HuggingFace format: save to directory
            save_path = self.save_dir / self.save_dirname
            self.save_pretrained_fn(save_path)
        else:
            # PyTorch format: save state_dict to file
            checkpoint = {
                'epoch': state.epoch + 1,
                'global_step': state.global_step,
                'model_state_dict': state.model.state_dict(),
                f'best_{self.metric_name}': self.best_value,
                'eval_metrics': dict(state.eval_metrics),
            }

            # Include EMA model if available
            ema_model = state.callback_state.get('ema_model')
            if ema_model is not None:
                checkpoint['ema_model_state_dict'] = ema_model.state_dict()

            # Include SWA model if available
            swa_model = state.callback_state.get('swa_model')
            if swa_model is not None:
                checkpoint['swa_model_state_dict'] = swa_model.state_dict()
                checkpoint['swa_n_averaged'] = state.callback_state.get('swa_n_averaged', 0)

            save_path = self.save_dir / self.filename
            torch.save(checkpoint, save_path)

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            'best_value': self.best_value,
            'best_epoch': self._best_epoch,
            'step_loss_sum': self._step_loss_sum,
            'step_loss_count': self._step_loss_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.best_value = state_dict['best_value']
        self._best_epoch = state_dict['best_epoch']
        self._step_loss_sum = state_dict.get('step_loss_sum', 0.0)
        self._step_loss_count = state_dict.get('step_loss_count', 0)
