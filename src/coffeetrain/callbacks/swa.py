"""Stochastic Weight Averaging callback."""

from copy import deepcopy
from typing import Any, Dict, Optional

import torch

from coffeetrain.callback import Callback
from coffeetrain.state import State


class SWACallback(Callback):
    """Stochastic Weight Averaging callback.

    Maintains an SWA copy of the model weights by averaging model snapshots
    taken at regular intervals during the later stages of training.

    Unlike EMA (exponential decay), SWA uses a simple arithmetic mean:
    swa = (swa * n + model) / (n + 1) where n = number of updates

    SWA typically starts after the model has largely converged (e.g., last 25%
    of training) and averages snapshots from this phase to find flatter minima.

    Attributes:
        start_epoch: Epoch to start SWA averaging (0-indexed)
        update_interval: Average model every N epochs
        swa_model: Copy of model with SWA weights
    """

    def __init__(
        self,
        start_epoch: int = 0,
        update_interval: int = 1,
    ):
        """Initialize SWA callback.

        Args:
            start_epoch: Epoch to start SWA (0-indexed, default: 0 = immediately)
            update_interval: Update SWA every N epochs
        """
        self.start_epoch = start_epoch
        self.update_interval = update_interval

        self.swa_model: Optional[torch.nn.Module] = None
        self._n_averaged: int = 0
        self._update_count: int = 0
        self._swa_started: bool = False

    def _maybe_start_swa(self, state: State) -> None:
        """Initialize SWA model on first update after start_epoch."""
        if self._swa_started:
            return

        if state.epoch < self.start_epoch:
            return

        # Initialize SWA model with current weights
        self.swa_model = deepcopy(state.model)
        self.swa_model.eval()
        self.swa_model.to(state.device)
        self._n_averaged = 1
        self._swa_started = True

        # Store on callback_state for access by other callbacks
        state.callback_state['swa_model'] = self.swa_model
        state.callback_state['swa_n_averaged'] = self._n_averaged

        print(f"  SWA: Started averaging at epoch {state.epoch + 1}")

    def _update_swa(self, state: State) -> None:
        """Update SWA weights with current model."""
        if self.swa_model is None:
            return

        with torch.no_grad():
            # Running average: swa = (swa * n + model) / (n + 1)
            for swa_param, model_param in zip(
                self.swa_model.parameters(),
                state.model.parameters()
            ):
                swa_param.data.mul_(self._n_averaged)
                swa_param.data.add_(model_param.data)
                swa_param.data.div_(self._n_averaged + 1)

        self._n_averaged += 1
        state.callback_state['swa_n_averaged'] = self._n_averaged

    def epoch_end(self, state: State) -> None:
        """Update SWA weights at end of each epoch."""
        was_started = self._swa_started
        self._maybe_start_swa(state)

        if not self._swa_started:
            return

        # Skip update on the epoch we started (already captured first snapshot)
        if not was_started:
            return

        self._update_count += 1
        if self._update_count % self.update_interval == 0:
            self._update_swa(state)
            print(f"  SWA: Updated weights ({self._n_averaged} models averaged)")

    def fit_end(self, state: State) -> None:
        """Log final SWA statistics."""
        if self._swa_started and self.swa_model is not None:
            print(f"  SWA: Final model averaged from {self._n_averaged} snapshots")

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            'start_epoch': self.start_epoch,
            'n_averaged': self._n_averaged,
            'update_count': self._update_count,
            'swa_started': self._swa_started,
            'swa_state_dict': self.swa_model.state_dict() if self.swa_model else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._n_averaged = state_dict['n_averaged']
        self._update_count = state_dict['update_count']
        self._swa_started = state_dict['swa_started']
        if self.swa_model is not None and state_dict.get('swa_state_dict') is not None:
            self.swa_model.load_state_dict(state_dict['swa_state_dict'])
