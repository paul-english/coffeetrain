#!/usr/bin/env python3
"""Exponential Moving Average plugin for Trainer.

Maintains an EMA copy of the model weights, updated at configurable batch intervals.
Can optionally swap to EMA model for evaluation.

EMA formula: ema = decay * ema + (1 - decay) * model

Usage:
  from coffeetrain.plugins.ema import ema_plugin
  from coffeetrain import Trainer

  trainer = Trainer()
  trainer.register_plugin(ema_plugin)

  # Configure via trainer hyperparams:
  # trainer --ema_decay 0.999 --ema_update_interval 1 --ema_use_for_eval True

Parameters (all optional with sensible defaults):
  - ema_decay: EMA decay rate (default: 0.999, higher = slower updates)
  - ema_update_interval: Update EMA every N batches (default: 1)
  - ema_use_for_eval: Use EMA model for evaluation (default: True)
"""

from copy import deepcopy
import torch

from coffeetrain.plugin import Plugin

ema_plugin = Plugin(
    name='ema',
    description='Exponential Moving Average plugin with shadow model, batch-interval updates, and optional eval swap',
)


@ema_plugin.system('FIT_BEFORE')
def init_ema_model(model, device, ema_decay: float = 0.999, ema_update_interval: int = 1, ema_use_for_eval: bool = True):
    """Create EMA model copy and initialize state.

    Args:
        model: The training model
        device: Device to place EMA model on
        ema_decay: EMA decay rate (higher = slower updates)
        ema_update_interval: Update EMA every N batches
        ema_use_for_eval: Use EMA model for validation
    """
    # Create EMA model copy
    ema_model = deepcopy(model)
    ema_model.eval()
    ema_model.to(device)

    return {
        'ema_model': ema_model,
        'ema_batch_count': 0,
        'ema_decay': ema_decay,
        'ema_update_interval': ema_update_interval,
        'ema_use_for_eval': ema_use_for_eval,
        'ema_original_model': None,
    }


@ema_plugin.system('BATCH_AFTER')
def update_ema_weights(
    model,
    get_state,
    set_state,
    ema_decay: float = 0.999,
    ema_update_interval: int = 1,
):
    """Update EMA weights after each training batch.

    Args:
        model: The training model
        get_state: Function to get plugin state
        set_state: Function to set plugin state
        ema_decay: EMA decay rate
        ema_update_interval: Update EMA every N batches
    """
    ema_model = get_state('ema_model')
    if ema_model is None:
        return

    batch_count = get_state('ema_batch_count')
    batch_count += 1

    # Only update at specified intervals
    if batch_count % ema_update_interval != 0:
        set_state({'ema_batch_count': batch_count})
        return

    # Update EMA weights
    with torch.no_grad():
        for ema_param, model_param in zip(
            ema_model.parameters(),
            model.parameters()
        ):
            ema_param.data.mul_(ema_decay).add_(
                model_param.data, alpha=1.0 - ema_decay
            )

    set_state({'ema_batch_count': batch_count})


@ema_plugin.system('EVAL_BEFORE')
def swap_to_ema_for_eval(
    get_state,
    set_state,
    ema_use_for_eval: bool = True,
):
    """Swap to EMA model for evaluation if enabled.

    Args:
        get_state: Function to get plugin state
        set_state: Function to set plugin state
        ema_use_for_eval: Whether to use EMA model for evaluation
    """
    if not ema_use_for_eval:
        return

    ema_model = get_state('ema_model')
    if ema_model is None:
        return

    # Note: The actual model swap happens in the execution context
    # We just store that we should be using EMA for this eval phase
    return {'ema_eval_swap': True}


@ema_plugin.system('EVAL_BEFORE')
def store_original_model_for_eval_swap(
    model,
    get_state,
    set_state,
    ema_use_for_eval: bool = True,
):
    """Store original model reference before swapping to EMA.

    Args:
        model: The training model
        get_state: Function to get plugin state
        set_state: Function to set plugin state
        ema_use_for_eval: Whether to use EMA model for evaluation
    """
    if not ema_use_for_eval:
        return

    ema_model = get_state('ema_model')
    if ema_model is None:
        return

    # Store reference to original model and swap in context
    # This swaps the model reference in the execution context
    return {
        'ema_original_model': model,
        'model': ema_model,
    }


@ema_plugin.system('EVAL_AFTER')
def restore_original_model_after_eval(
    get_state,
    set_state,
    ema_use_for_eval: bool = True,
):
    """Restore original model after evaluation.

    Args:
        get_state: Function to get plugin state
        set_state: Function to set plugin state
        ema_use_for_eval: Whether to use EMA model for evaluation
    """
    if not ema_use_for_eval:
        return

    ema_original_model = get_state('ema_original_model')
    if ema_original_model is None:
        return

    # Restore the original model
    return {
        'model': ema_original_model,
        'ema_original_model': None,
    }


@ema_plugin.system('FIT_AFTER')
def cleanup_ema(get_state, set_state):
    """Clean up EMA state after training.

    Args:
        get_state: Function to get plugin state
        set_state: Function to set plugin state
    """
    set_state({
        'ema_model': None,
        'ema_batch_count': 0,
        'ema_original_model': None,
    })
