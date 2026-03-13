"""Training state container."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


@dataclass
class State:
    """Centralized training state.

    Contains all state needed during training, passed to callbacks
    for reading and modification.

    Attributes:
        model: The model being trained
        optimizers: List of optimizers
        schedulers: List of LR schedulers
        train_dataloader: Training data loader
        eval_dataloader: Optional validation data loader

        batch: Current batch data
        outputs: Model forward outputs
        loss: Current loss tensor

        epoch: Current epoch (0-indexed)
        batch_idx: Current batch index within epoch
        global_step: Total batches processed
        max_epochs: Total epochs to train
        grad_accum: Gradient accumulation steps (can be modified by callbacks)

        train_metrics: Accumulated training metrics
        eval_metrics: Accumulated evaluation metrics

        device: Training device
    """

    # Core training components
    model: torch.nn.Module
    optimizers: List[Optimizer]
    schedulers: List[LRScheduler]
    train_dataloader: DataLoader
    eval_dataloader: Optional[DataLoader] = None

    # Current batch state
    batch: Any = None
    outputs: Any = None
    loss: Optional[Tensor] = None

    # Progress tracking
    epoch: int = 0
    batch_idx: int = 0
    global_step: int = 0
    max_epochs: int = 1
    grad_accum: int = 1  # Gradient accumulation steps (modifiable by callbacks)
    stop_training: bool = False  # Set by callbacks to stop training early

    # Metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    eval_metrics: Dict[str, float] = field(default_factory=dict)

    # Device
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))

    # Distributed training
    is_main_process: bool = True  # True if rank 0 or single-process training

    # Extensibility: callbacks can store arbitrary data here
    callback_state: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_training(self) -> bool:
        """Check if model is in training mode."""
        return self.model.training

    @property
    def num_batches(self) -> int:
        """Get number of batches in training dataloader."""
        return len(self.train_dataloader)

    @property
    def total_steps(self) -> int:
        """Get total training steps (batches * epochs)."""
        return self.num_batches * self.max_epochs

    def get_lr(self) -> float:
        """Get current learning rate from first optimizer."""
        if self.optimizers:
            return self.optimizers[0].param_groups[0]['lr']
        return 0.0

    def get_all_lrs(self) -> Dict[str, float]:
        """Get all learning rates from all param groups."""
        lrs = {}
        for i, optimizer in enumerate(self.optimizers):
            for j, group in enumerate(optimizer.param_groups):
                name = group.get('name', f'opt{i}_group{j}')
                lrs[name] = group['lr']
        return lrs
