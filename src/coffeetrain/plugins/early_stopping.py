#!/usr/bin/env python3
"""Early stopping plugin: stop training when a monitored metric plateaus.

Works with the default `train` loop — `BATCH_AFTER` raises `StopLoop`, which
the `train` command catches for a graceful stop — or with any loop that checks
the `stop_training` state flag.

Usage:
  from coffeetrain.plugins.early_stopping import early_stopping
  from coffeetrain import Trainer

  trainer = Trainer()
  trainer.register_plugin(early_stopping)

  # Configure via hyperparams / CLI flags:
  #   --early_stopping_metric eval_loss --early_stopping_mode min \\
  #   --early_stopping_patience 5 --early_stopping_min_delta 0.0

Parameters (all optional with sensible defaults):
  - early_stopping_metric: Context key holding the monitored value
    (default: 'eval_loss')
  - early_stopping_mode: 'min' (lower is better) or 'max' (default: 'min')
  - early_stopping_patience: Epochs without improvement before stopping
    (default: 5)
  - early_stopping_min_delta: Minimum change to count as improvement
    (default: 0.0)
"""

from coffeetrain.plugin import Plugin
from coffeetrain.plugins.interruptable_train import StopLoop

early_stopping = Plugin(
    name='early_stopping',
    description='Stops training early if the monitored metric plateaus. '
                'The training loop checks `stop_training` / catches StopLoop on BATCH_AFTER.',
)


@early_stopping.system('FIT_BEFORE')
def init_early_stopping(
    early_stopping_metric: str = 'eval_loss',
    early_stopping_mode: str = 'min',
    early_stopping_patience: int = 5,
    early_stopping_min_delta: float = 0.0,
):
    """Initialize early stopping state.

    Raises:
        ValueError: If mode is not 'min' or 'max', or patience is negative.
    """
    if early_stopping_mode not in ('min', 'max'):
        raise ValueError(
            f"early_stopping_mode must be 'min' or 'max', got {early_stopping_mode!r}"
        )
    if early_stopping_patience < 0:
        raise ValueError(
            f"early_stopping_patience must be >= 0, got {early_stopping_patience}"
        )
    return {
        'early_stopping_metric': early_stopping_metric,
        'early_stopping_mode': early_stopping_mode,
        'early_stopping_patience': early_stopping_patience,
        'early_stopping_min_delta': early_stopping_min_delta,
        'early_stopping_best': None,
        'early_stopping_bad_epochs': 0,
    }


def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if mode == 'min':
        return current < best - min_delta
    return current > best + min_delta


def _check_metric(get_state, set_state) -> None:
    """Shared plateau check: update best/bad-epoch counters, maybe stop."""
    metric = get_state('early_stopping_metric')
    current = get_state(metric)
    if current is None:
        return
    try:
        current_value = float(current.item() if hasattr(current, 'item') else current)
    except (TypeError, ValueError):
        return

    mode = get_state('early_stopping_mode')
    min_delta = get_state('early_stopping_min_delta')
    patience = get_state('early_stopping_patience')
    best = get_state('early_stopping_best')
    bad_epochs = get_state('early_stopping_bad_epochs') or 0

    if best is None or _is_improvement(current_value, best, mode, min_delta):
        set_state({
            'early_stopping_best': current_value,
            'early_stopping_bad_epochs': 0,
        })
        return

    bad_epochs += 1
    set_state({'early_stopping_bad_epochs': bad_epochs})
    if bad_epochs >= patience:
        set_state({'stop_training': True})


@early_stopping.system('EVAL_AFTER')
def early_stopping_eval_end(get_state, set_state):
    """Record the monitored metric right after eval produces it."""
    _check_metric(get_state, set_state)


@early_stopping.system('BATCH_AFTER')
def should_we_stop(get_state):
    """Raise StopLoop once the plateau has been flagged (graceful stop)."""
    if get_state('stop_training'):
        metric = get_state('early_stopping_metric')
        best = get_state('early_stopping_best')
        raise StopLoop(
            f'Early stopping training, {metric} has plateaued at {best}'
        )


@early_stopping.system('FIT_AFTER')
def cleanup_early_stopping(set_state):
    """Reset per-run early stopping counters."""
    set_state({
        'early_stopping_bad_epochs': 0,
    })
