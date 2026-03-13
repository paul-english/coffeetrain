"""Exponential Moving Average callback."""

from copy import deepcopy
from typing import Any, Dict, Optional

import torch

from coffeetrain.callback import Callback
from coffeetrain.state import State


class EMACallback(Callback):
    """Exponential Moving Average callback.

    Maintains an EMA copy of the model weights, updated after each batch.
    Can optionally swap to EMA model for evaluation.

    EMA formula: ema = decay * ema + (1 - decay) * model

    Attributes:
        decay: EMA decay rate (0.999 = slow updates, 0.9 = fast updates)
        ema_model: Copy of model with EMA weights
        use_ema_for_eval: Whether to use EMA model during evaluation
    """

    def __init__(
        self,
        decay: float = 0.999,
        update_interval: int = 1,
        use_ema_for_eval: bool = True,
    ):
        """Initialize EMA callback.

        Args:
            decay: EMA decay rate (higher = slower updates)
            update_interval: Update EMA every N batches
            use_ema_for_eval: Use EMA model for validation
        """
        self.decay = decay
        self.update_interval = update_interval
        self.use_ema_for_eval = use_ema_for_eval

        self.ema_model: Optional[torch.nn.Module] = None
        self._original_model: Optional[torch.nn.Module] = None
        self._batch_count = 0

    def init(self, state: State) -> None:
        """Create EMA model copy."""
        self.ema_model = deepcopy(state.model)
        self.ema_model.eval()
        self.ema_model.to(state.device)

        # Store on callback_state for access by other callbacks
        state.callback_state['ema_model'] = self.ema_model

    def batch_end(self, state: State) -> None:
        """Update EMA weights after each training batch."""
        if self.ema_model is None:
            return

        self._batch_count += 1
        if self._batch_count % self.update_interval != 0:
            return

        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                state.model.parameters()
            ):
                ema_param.data.mul_(self.decay).add_(
                    model_param.data, alpha=1.0 - self.decay
                )

    def eval_start(self, state: State) -> None:
        """Swap to EMA model for evaluation."""
        if not self.use_ema_for_eval or self.ema_model is None:
            return

        self._original_model = state.model
        state.model = self.ema_model

    def eval_end(self, state: State) -> None:
        """Restore original model after evaluation."""
        if self._original_model is not None:
            state.model = self._original_model
            self._original_model = None

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            'decay': self.decay,
            'batch_count': self._batch_count,
            'ema_state_dict': self.ema_model.state_dict() if self.ema_model else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.decay = state_dict['decay']
        self._batch_count = state_dict['batch_count']
        if self.ema_model is not None and state_dict.get('ema_state_dict') is not None:
            self.ema_model.load_state_dict(state_dict['ema_state_dict'])
