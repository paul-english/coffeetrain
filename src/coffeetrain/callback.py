"""Callback base class for training hooks."""

from typing import Any, Dict

from coffeetrain.state import State


class Callback:
    """Base class for training callbacks.

    Callbacks receive State at each event and can read/modify it.
    Override the methods corresponding to events you want to handle.

    All methods are no-ops by default, so subclasses only need to
    implement the hooks they care about.

    Example:
        class PrintLossCallback(Callback):
            def batch_end(self, state: State) -> None:
                print(f"Loss: {state.loss.item():.4f}")
    """

    # Initialization
    def init(self, state: State) -> None:
        """Called after Trainer construction, before fit().

        Use for callback initialization that needs access to state.
        """
        pass

    # Fit lifecycle
    def fit_start(self, state: State) -> None:
        """Called at the start of fit(), before any training."""
        pass

    def fit_end(self, state: State) -> None:
        """Called at the end of fit(), after all training."""
        pass

    # Epoch lifecycle
    def epoch_start(self, state: State) -> None:
        """Called at the start of each epoch."""
        pass

    def epoch_end(self, state: State) -> None:
        """Called at the end of each epoch, before evaluation."""
        pass

    # Training batch lifecycle
    def batch_start(self, state: State) -> None:
        """Called at the start of each training batch."""
        pass

    def before_forward(self, state: State) -> None:
        """Called before model.forward()."""
        pass

    def after_forward(self, state: State) -> None:
        """Called after model.forward(), state.outputs is set."""
        pass

    def before_loss(self, state: State) -> None:
        """Called before model.loss()."""
        pass

    def after_loss(self, state: State) -> None:
        """Called after model.loss(), state.loss is set."""
        pass

    def before_backward(self, state: State) -> None:
        """Called before loss.backward()."""
        pass

    def after_backward(self, state: State) -> None:
        """Called after loss.backward(), before optimizer step."""
        pass

    def batch_end(self, state: State) -> None:
        """Called after optimizer step and scheduler step."""
        pass

    # Evaluation lifecycle
    def eval_start(self, state: State) -> None:
        """Called at the start of evaluation."""
        pass

    def eval_batch_start(self, state: State) -> None:
        """Called at the start of each eval batch."""
        pass

    def eval_before_forward(self, state: State) -> None:
        """Called before model.forward() during eval."""
        pass

    def eval_after_forward(self, state: State) -> None:
        """Called after model.forward() during eval."""
        pass

    def eval_batch_end(self, state: State) -> None:
        """Called at the end of each eval batch."""
        pass

    def eval_end(self, state: State) -> None:
        """Called at the end of evaluation."""
        pass

    # Checkpointing support
    def state_dict(self) -> Dict[str, Any]:
        """Return callback state for checkpointing.

        Override to save callback-specific state.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load callback state from checkpoint.

        Override to restore callback-specific state.
        """
        pass
