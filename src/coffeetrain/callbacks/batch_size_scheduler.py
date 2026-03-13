"""Batch size scheduling callback for warmup."""

from typing import Any, Dict, Literal

from coffeetrain.callback import Callback
from coffeetrain.state import State


class BatchSizeSchedulerCallback(Callback):
    """Schedule batch size warmup during training.

    Increases effective batch size by adjusting gradient accumulation.
    Batch size increases from start_batch_size to final_batch_size
    over warmup_steps.

    The effective batch size = microbatch_size * grad_accum.
    This callback modifies state.grad_accum to achieve the target batch sizes.

    Two schedule types are available:
    - "linear": Smooth linear interpolation (default)
    - "equal_steps": Equal optimizer update steps at each discrete batch size
      level (ModernBERT's "uneven token schedule")

    From ModernBERT: "Batch size warmup is a common-knowledge trick to speed up
    model training when working with medium to large batch sizes. Instead of
    'wasting' a full batch on updating the suboptimal initial weight distribution,
    we update the model weights on a gradually increasing batch size."

    ModernBERT uses an "uneven token schedule so each batch size has the same
    number of update steps" - this is the "equal_steps" schedule type.

    Example:
        callback = BatchSizeSchedulerCallback(
            microbatch_size=96,       # Physical batch size
            start_batch_size=768,     # Effective batch at start (grad_accum=8)
            final_batch_size=4608,    # Effective batch at end (grad_accum=48)
            warmup_steps=10000,       # Steps to warmup over
            schedule_type="equal_steps",  # Equal steps per batch size level
        )
        trainer = Trainer(..., callbacks=[callback])

    Attributes:
        microbatch_size: Physical batch size per forward pass
        start_batch_size: Initial effective batch size
        final_batch_size: Target effective batch size
        warmup_steps: Number of optimizer steps to warmup over
        schedule_type: "linear" for smooth interpolation, "equal_steps" for
            equal optimizer steps at each discrete batch size level
    """

    def __init__(
        self,
        microbatch_size: int,
        start_batch_size: int,
        final_batch_size: int,
        warmup_steps: int,
        schedule_type: Literal["linear", "equal_steps"] = "linear",
    ):
        """Initialize batch size scheduler.

        Args:
            microbatch_size: Physical batch size per forward pass
            start_batch_size: Initial effective batch size
            final_batch_size: Target effective batch size
            warmup_steps: Number of optimizer steps to warmup over
            schedule_type: Schedule type - "linear" for smooth interpolation,
                "equal_steps" for equal optimizer steps at each batch size level

        Raises:
            ValueError: If batch sizes are not divisible by microbatch_size
        """
        if start_batch_size % microbatch_size != 0:
            raise ValueError(
                f"start_batch_size ({start_batch_size}) must be divisible by "
                f"microbatch_size ({microbatch_size})"
            )
        if final_batch_size % microbatch_size != 0:
            raise ValueError(
                f"final_batch_size ({final_batch_size}) must be divisible by "
                f"microbatch_size ({microbatch_size})"
            )
        if start_batch_size > final_batch_size:
            raise ValueError(
                f"start_batch_size ({start_batch_size}) must be <= "
                f"final_batch_size ({final_batch_size})"
            )

        self.microbatch_size = microbatch_size
        self.start_batch_size = start_batch_size
        self.final_batch_size = final_batch_size
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type

        # Pre-compute grad_accum bounds
        self.start_grad_accum = start_batch_size // microbatch_size
        self.final_grad_accum = final_batch_size // microbatch_size

        # Pre-compute for equal_steps schedule
        if schedule_type == "equal_steps":
            self.num_levels = self.final_grad_accum - self.start_grad_accum + 1
            self.steps_per_level = warmup_steps / self.num_levels if self.num_levels > 0 else 1

    def _compute_grad_accum(self, step: int) -> int:
        """Compute gradient accumulation for a given step.

        Args:
            step: Current optimizer step (global_step)

        Returns:
            Gradient accumulation value for this step
        """
        if step >= self.warmup_steps:
            return self.final_grad_accum

        if self.schedule_type == "equal_steps":
            # Equal optimizer steps at each batch size level
            level = int(step / self.steps_per_level)
            return min(self.start_grad_accum + level, self.final_grad_accum)
        else:
            # Linear interpolation (default)
            progress = step / max(1, self.warmup_steps)
            grad_accum_float = (
                self.start_grad_accum
                + progress * (self.final_grad_accum - self.start_grad_accum)
            )
            return max(1, round(grad_accum_float))

    def init(self, state: State) -> None:
        """Set initial grad_accum at training start."""
        state.grad_accum = self._compute_grad_accum(0)
        print(
            f"[BatchSizeScheduler] initialized: grad_accum={state.grad_accum} "
            f"(effective batch: {self.microbatch_size * state.grad_accum}), "
            f"warmup_steps={self.warmup_steps}, schedule={self.schedule_type}"
        )

    def batch_end(self, state: State) -> None:
        """Update grad_accum after each optimizer step."""
        # Update for next batch based on current step
        new_grad_accum = self._compute_grad_accum(state.global_step)
        if new_grad_accum != state.grad_accum:
            print(
                f"[BatchSizeScheduler] step={state.global_step}: "
                f"grad_accum {state.grad_accum} -> {new_grad_accum} "
                f"(effective batch: {self.microbatch_size * new_grad_accum})"
            )
            state.grad_accum = new_grad_accum

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {
            'microbatch_size': self.microbatch_size,
            'start_batch_size': self.start_batch_size,
            'final_batch_size': self.final_batch_size,
            'warmup_steps': self.warmup_steps,
            'schedule_type': self.schedule_type,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Note: The scheduler parameters are stored for verification,
        but the actual grad_accum is recomputed from global_step.
        """
        if state_dict.get('microbatch_size') != self.microbatch_size:
            print(
                f"Warning: BatchSizeScheduler microbatch_size mismatch: "
                f"checkpoint={state_dict.get('microbatch_size')}, "
                f"current={self.microbatch_size}"
            )
