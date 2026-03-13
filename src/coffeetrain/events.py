"""Training lifecycle events."""

from enum import Enum


class Event(Enum):
    """Events fired during training lifecycle.

    Events are dispatched to callbacks at specific points in the training loop.
    Each event has a corresponding method name on the Callback class.

    Lifecycle order for a single batch:
        FIT_START (once)
        └── EPOCH_START
            └── BATCH_START
                ├── BEFORE_FORWARD
                ├── AFTER_FORWARD
                ├── BEFORE_LOSS
                ├── AFTER_LOSS
                ├── BEFORE_BACKWARD
                ├── AFTER_BACKWARD
                └── BATCH_END
            └── EPOCH_END
                └── EVAL_START (if eval_dataloader)
                    └── EVAL_BATCH_START
                        ├── EVAL_BEFORE_FORWARD
                        ├── EVAL_AFTER_FORWARD
                        └── EVAL_BATCH_END
                    └── EVAL_END
        TRAINING_INTERRUPTED (if interrupted by signal, before FIT_END)
        FIT_END (once)
    """

    # Initialization and fit lifecycle
    INIT = "init"
    FIT_START = "fit_start"
    FIT_END = "fit_end"
    TRAINING_INTERRUPTED = "training_interrupted"

    # Epoch lifecycle
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"

    # Training batch lifecycle
    BATCH_START = "batch_start"
    BEFORE_FORWARD = "before_forward"
    AFTER_FORWARD = "after_forward"
    BEFORE_LOSS = "before_loss"
    AFTER_LOSS = "after_loss"
    BEFORE_BACKWARD = "before_backward"
    AFTER_BACKWARD = "after_backward"
    BATCH_END = "batch_end"

    # Evaluation lifecycle
    EVAL_START = "eval_start"
    EVAL_BATCH_START = "eval_batch_start"
    EVAL_BEFORE_FORWARD = "eval_before_forward"
    EVAL_AFTER_FORWARD = "eval_after_forward"
    EVAL_BATCH_END = "eval_batch_end"
    EVAL_END = "eval_end"
