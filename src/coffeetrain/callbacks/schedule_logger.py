"""Learning rate schedule phase logging callback."""

from typing import Any, Dict, Literal, Optional

from coffeetrain.callback import Callback
from coffeetrain.state import State
from coffeetrain.schedulers import WSDScheduler

Phase = Literal["warmup", "stable", "decay", "done"]


class ScheduleLoggerCallback(Callback):
    """Log learning rate schedule phase transitions.

    Detects when the LR scheduler transitions between phases (warmup, stable,
    decay, done) and prints a notification to stdout.

    Works with WSDScheduler (Warmup-Stable-Decay) by checking the current step
    against the scheduler's phase boundaries.

    Example:
        >>> callback = ScheduleLoggerCallback()
        >>> trainer = Trainer(model, schedulers=[wsd_scheduler], callbacks=[callback])

    Output:
        [LRSchedule] step=1000: warmup -> stable (lr=1.00e-04)
        [LRSchedule] step=6000: stable -> decay (lr=1.00e-04)
    """

    def __init__(self):
        """Initialize the schedule logger."""
        self._current_phase: Optional[Phase] = None
        self._wsd_scheduler: Optional[WSDScheduler] = None

    def init(self, state: State) -> None:
        """Find WSDScheduler in the trainer's schedulers."""
        for scheduler in state.schedulers:
            if isinstance(scheduler, WSDScheduler):
                self._wsd_scheduler = scheduler
                break

        if self._wsd_scheduler is None:
            print("[ScheduleLogger] No WSDScheduler found, phase logging disabled")
            return

        # Determine initial phase
        self._current_phase = self._get_phase(0)
        print(
            f"[LRSchedule] initialized: phase={self._current_phase}, "
            f"warmup={self._wsd_scheduler.warmup_steps}, "
            f"stable={self._wsd_scheduler.stable_steps}, "
            f"decay={self._wsd_scheduler.decay_steps}"
        )

    def _get_phase(self, step: int) -> Phase:
        """Determine the current phase for a given step."""
        if self._wsd_scheduler is None:
            return "done"

        warmup_end = self._wsd_scheduler.warmup_steps
        stable_end = warmup_end + self._wsd_scheduler.stable_steps
        decay_end = stable_end + self._wsd_scheduler.decay_steps

        if step < warmup_end:
            return "warmup"
        elif step < stable_end:
            return "stable"
        elif step < decay_end:
            return "decay"
        else:
            return "done"

    def batch_end(self, state: State) -> None:
        """Check for phase transitions after each batch."""
        if self._wsd_scheduler is None:
            return

        new_phase = self._get_phase(state.global_step)
        if new_phase != self._current_phase:
            lr = state.get_lr()
            print(
                f"[LRSchedule] step={state.global_step}: "
                f"{self._current_phase} -> {new_phase} (lr={lr:.2e})"
            )
            self._current_phase = new_phase

    def state_dict(self) -> Dict[str, Any]:
        """Return state for checkpointing."""
        return {"current_phase": self._current_phase}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self._current_phase = state_dict.get("current_phase")
