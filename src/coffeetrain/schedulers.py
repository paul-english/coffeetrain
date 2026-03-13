"""Learning rate schedulers."""

import math
from typing import List, Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

DecayType = Literal["cosine", "linear", "1-sqrt"]


class CosineWarmupScheduler(LRScheduler):
    """Cosine annealing with linear warmup.

    Learning rate schedule:
    - Linear warmup from 0 to base_lr over warmup_steps
    - Cosine decay from base_lr to min_lr over remaining steps

    Example:
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=1000,
            total_steps=10000,
            min_lr=0.0,
        )
        for batch in dataloader:
            loss.backward()
            optimizer.step()
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate after decay
            last_epoch: Index of last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch + 1  # 0-indexed

        if step < self.warmup_steps:
            # Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, progress)  # Clamp to [0, 1]
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(LRScheduler):
    """Linear warmup then constant learning rate.

    Learning rate schedule:
    - Linear warmup from 0 to base_lr over warmup_steps
    - Constant base_lr for remaining steps

    Example:
        scheduler = LinearWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=1000,
        )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            last_epoch: Index of last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            return list(self.base_lrs)


class WSDScheduler(LRScheduler):
    """Warmup-Stable-Decay (trapezoidal) learning rate scheduler.

    Three-phase schedule:
    1. Warmup: Linear ramp from 0 to base_lr
    2. Stable: Constant base_lr
    3. Decay: Decay from base_lr to min_lr using specified decay type

    This schedule is also known as "modified trapezoidal" and is used in
    ModernBERT training. The 1-sqrt decay was found to outperform linear
    and cosine decay (Hägele et al., 2024).

    Example:
        scheduler = WSDScheduler(
            optimizer=optimizer,
            warmup_steps=1000,
            stable_steps=5000,
            decay_steps=4000,
            min_lr=0.0,
            decay_type="1-sqrt",
        )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        min_lr: float = 0.0,
        decay_type: DecayType = "cosine",
        last_epoch: int = -1,
    ):
        """Initialize scheduler.

        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps (linear ramp)
            stable_steps: Number of stable steps (constant LR)
            decay_steps: Number of decay steps
            min_lr: Minimum learning rate after decay
            decay_type: Decay function - 'cosine', 'linear', or '1-sqrt'
            last_epoch: Index of last epoch (for resuming)
        """
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.decay_type = decay_type
        super().__init__(optimizer, last_epoch)

    @property
    def total_steps(self) -> int:
        """Total number of steps across all phases."""
        return self.warmup_steps + self.stable_steps + self.decay_steps

    def _compute_decay_factor(self, progress: float) -> float:
        """Compute decay multiplier based on decay type.

        Args:
            progress: Progress through decay phase, in [0, 1]

        Returns:
            Decay factor in [0, 1], where 1 = base_lr and 0 = min_lr
        """
        if self.decay_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.decay_type == "linear":
            return 1.0 - progress
        elif self.decay_type == "1-sqrt":
            return 1.0 - math.sqrt(progress)
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")

    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch + 1

        warmup_end = self.warmup_steps
        stable_end = warmup_end + self.stable_steps
        decay_end = stable_end + self.decay_steps

        if step < warmup_end:
            # Phase 1: Linear warmup
            warmup_factor = step / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        elif step < stable_end:
            # Phase 2: Stable (constant LR)
            return list(self.base_lrs)

        elif step < decay_end:
            # Phase 3: Decay
            progress = (step - stable_end) / max(1, self.decay_steps)
            decay_factor = self._compute_decay_factor(progress)
            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]

        else:
            # Beyond decay phase - hold at min_lr
            return [self.min_lr for _ in self.base_lrs]
