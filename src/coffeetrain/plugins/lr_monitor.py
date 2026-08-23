#!/usr/bin/env python3
"""Learning rate monitoring plugin for Trainer.

This plugin monitors and logs learning rates during training.

Features:
  - Print learning rates at configurable batch intervals
  - Track learning rate metrics per epoch
  - Support for multiple parameter groups

Usage:
  from coffeetrain.plugins.lr_monitor import lr_monitor_plugin
  from coffeetrain import Trainer

  trainer = Trainer()
  trainer.register_plugin(lr_monitor_plugin)

  # Configure via trainer hyperparams or command-line args:
  # trainer --lr_monitor_log_interval 100

Parameters (all optional with sensible defaults):
  - lr_monitor_log_interval: Print learning rate every N batches (default: 0, which means epoch-end only)
"""

from typing import Any, Dict, Optional

from coffeetrain.plugin import Plugin

lr_monitor_plugin = Plugin(
    name='lr_monitor',
    description='Learning rate monitoring with periodic logging and metric tracking',
)


def _get_learning_rates(optimizer) -> Dict[str, float]:
    """Extract learning rates from all parameter groups.

    Args:
        optimizer: PyTorch optimizer instance

    Returns:
        Dictionary mapping parameter group names to learning rates
    """
    lrs = {}
    for i, param_group in enumerate(optimizer.param_groups):
        group_name = param_group.get('name', f'param_group_{i}')
        lrs[group_name] = param_group['lr']
    return lrs


@lr_monitor_plugin.system('FIT_BEFORE')
def init_lr_monitor(lr_monitor_log_interval: int = 0):
    """Initialize learning rate monitor state.

    Args:
        lr_monitor_log_interval: Print LR every N batches (0 = epoch end only)
    """
    return {
        'lr_monitor_batch_count': 0,
        'lr_monitor_log_interval': lr_monitor_log_interval,
    }


@lr_monitor_plugin.system('BATCH_AFTER')
def lr_monitor_batch_end(
    optimizer,
    global_step: int,
    get_state,
    set_state,
    lr_monitor_log_interval: int = 0,
):
    """Log learning rate at specified batch interval.

    Args:
        optimizer: PyTorch optimizer instance
        global_step: Current global training step
        get_state: Function to get plugin state
        set_state: Function to set plugin state
        lr_monitor_log_interval: Log interval in batches (0 = never)
    """
    batch_count = get_state('lr_monitor_batch_count')
    batch_count += 1

    if lr_monitor_log_interval > 0 and batch_count % lr_monitor_log_interval == 0:
        lrs = _get_learning_rates(optimizer)
        lr_str = ", ".join(f"{k}={v:.2e}" for k, v in lrs.items())
        print(f"  LR: {lr_str}")

    set_state({'lr_monitor_batch_count': batch_count})


@lr_monitor_plugin.system('EPOCH_AFTER')
def lr_monitor_epoch_end(
    optimizer,
    get_state,
    set_state,
    train_metrics: Optional[Dict[str, Any]] = None,
):
    """Log learning rate metrics at epoch end.

    Args:
        optimizer: PyTorch optimizer instance
        get_state: Function to get plugin state
        set_state: Function to set plugin state
        train_metrics: Dictionary to store training metrics
    """
    lrs = _get_learning_rates(optimizer)

    # Get the first (or only) learning rate as the primary learning_rate metric
    if lrs:
        primary_lr = next(iter(lrs.values()))
    else:
        primary_lr = None

    # Update train metrics if provided
    if train_metrics is not None:
        if primary_lr is not None:
            train_metrics['learning_rate'] = primary_lr
        train_metrics['learning_rates'] = lrs


@lr_monitor_plugin.system('FIT_AFTER')
def cleanup_lr_monitor(get_state, set_state):
    """Clean up learning rate monitor state."""
    set_state({
        'lr_monitor_batch_count': 0,
    })
