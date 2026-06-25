#!/usr/bin/env python3
"""Stochastic Weight Averaging plugin for TrainerV2.

Maintains an SWA copy of the model weights by averaging model snapshots
taken at regular intervals during the later stages of training.

Unlike EMA (exponential decay), SWA uses a simple arithmetic mean:
swa = (swa * n + model) / (n + 1) where n = number of updates

SWA typically starts after the model has largely converged (e.g., last 25%
of training) and averages snapshots from this phase to find flatter minima.

Usage:
  from coffeetrain.plugins.swa import swa_plugin
  from coffeetrain import TrainerV2

  trainer = TrainerV2()
  trainer.register_plugin(swa_plugin)

  # Configure via trainer hyperparams:
  # trainer --swa_start_epoch 75 --swa_update_interval 1

Parameters (all optional with sensible defaults):
  - swa_start_epoch: Epoch to start SWA averaging (default: 0, which means immediately)
  - swa_update_interval: Update SWA every N epochs (default: 1)
"""

from copy import deepcopy
from typing import Optional

import torch

from coffeetrain.plugin import Plugin

swa_plugin = Plugin(
    name='swa',
    description='Stochastic Weight Averaging with arithmetic mean, epoch-interval updates, stored in context for checkpointer',
)


@swa_plugin.system('FIT_BEFORE')
def init_swa_model(swa_start_epoch: int = 0, swa_update_interval: int = 1):
    """Initialize SWA plugin state.

    Args:
        swa_start_epoch: Epoch to start SWA averaging (0-indexed)
        swa_update_interval: Update SWA every N epochs
    """
    return {
        'swa_model': None,
        'swa_n_averaged': 0,
        'swa_start_epoch': swa_start_epoch,
        'swa_update_interval': swa_update_interval,
        'swa_started': False,
        'swa_update_count': 0,
    }


@swa_plugin.system('EPOCH_AFTER')
def update_swa_weights(
    model,
    epoch: int,
    device,
    get_state,
    set_state,
    swa_start_epoch: int = 0,
    swa_update_interval: int = 1,
):
    """Update SWA weights at epoch end.

    Args:
        model: The training model
        epoch: Current epoch (0-indexed)
        device: Device to place SWA model on
        get_state: Function to get plugin state
        set_state: Function to set plugin state
        swa_start_epoch: Epoch to start SWA averaging
        swa_update_interval: Update SWA every N epochs
    """
    swa_model = get_state('swa_model')
    swa_started = get_state('swa_started')
    swa_n_averaged = get_state('swa_n_averaged')
    swa_update_count = get_state('swa_update_count')

    # Initialize SWA model on first update after start_epoch
    if not swa_started and epoch >= swa_start_epoch:
        swa_model = deepcopy(model)
        swa_model.eval()
        swa_model.to(device)
        swa_n_averaged = 1
        swa_started = True

        set_state({
            'swa_model': swa_model,
            'swa_n_averaged': swa_n_averaged,
            'swa_started': swa_started,
        })
        print(f"  SWA: Started averaging at epoch {epoch + 1}")
        return

    # Update SWA only if started
    if not swa_started:
        return

    # Check update interval
    swa_update_count += 1
    if swa_update_count % swa_update_interval != 0:
        set_state({'swa_update_count': swa_update_count})
        return

    # Update SWA weights with arithmetic mean
    with torch.no_grad():
        for swa_param, model_param in zip(
            swa_model.parameters(),
            model.parameters()
        ):
            swa_param.data.mul_(swa_n_averaged)
            swa_param.data.add_(model_param.data)
            swa_param.data.div_(swa_n_averaged + 1)

    swa_n_averaged += 1
    set_state({
        'swa_n_averaged': swa_n_averaged,
        'swa_update_count': swa_update_count,
    })
    print(f"  SWA: Updated weights ({swa_n_averaged} models averaged)")


@swa_plugin.system('FIT_AFTER')
def cleanup_swa(get_state, set_state):
    """Finalize SWA state after training.

    Args:
        get_state: Function to get plugin state
        set_state: Function to set plugin state
    """
    swa_model = get_state('swa_model')
    swa_n_averaged = get_state('swa_n_averaged')

    if swa_model is not None:
        print(f"  SWA: Final model averaged from {swa_n_averaged} snapshots")
