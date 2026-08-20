#!/usr/bin/env python3
"""Save best model plugin for Trainer.

This plugin saves the model when a monitored validation metric improves.

Features:
  - Monitor validation metrics for improvement
  - Support for 'min' (loss) and 'max' (accuracy) modes
  - PyTorch state_dict/.pt saves with optional EMA/SWA inclusion
  - HuggingFace save_pretrained format support
  - Step-interval saves for monitoring long epochs (pre-training)
  - Distributed training safe: only main process saves
  - Configurable save directory, metric name, and modes

Usage:
  from coffeetrain.plugins.save_best_model import save_best_model_plugin
  from coffeetrain import Trainer

  trainer = Trainer()
  trainer.register_plugin(save_best_model_plugin)

  # Configure via trainer hyperparams or command-line args:
  # trainer --save_best_model_dir ./checkpoints \\
  #         --save_best_model_metric loss \\
  #         --save_best_model_mode min \\
  #         --save_best_model_save_steps 0

Parameters (all optional with sensible defaults):
  - save_best_model_dir: Directory to save checkpoints (default: './checkpoints')
  - save_best_model_metric: Metric to monitor (default: 'loss')
  - save_best_model_mode: 'min' for loss/error, 'max' for accuracy (default: 'min')
  - save_best_model_filename: Checkpoint filename (default: 'best_model.pt')
  - save_best_model_save_pretrained_fn: Optional callable for HF format saves
  - save_best_model_dirname: Directory name for HF saves (default: 'best_model')
  - save_best_model_track_train_metrics: Use train metrics instead of eval (default: False)
  - save_best_model_save_steps: Step interval for saving during batches, 0 to disable (default: 0)
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch

from coffeetrain.plugin import Plugin

save_best_model_plugin = Plugin(
    name='save_best_model',
    description='Save model checkpoint when monitored metric improves, with EMA/SWA support and step-interval saves',
)


@save_best_model_plugin.system('FIT_BEFORE')
def init_save_best_model(
    save_best_model_dir: str = './checkpoints',
    save_best_model_metric: str = 'loss',
    save_best_model_mode: str = 'min',
    save_best_model_filename: str = 'best_model.pt',
    save_best_model_save_pretrained_fn: Optional[Callable[[Path], None]] = None,
    save_best_model_dirname: str = 'best_model',
    save_best_model_track_train_metrics: bool = False,
    save_best_model_save_steps: int = 0,
):
    """Initialize best model saver state.

    Args:
        save_best_model_dir: Directory to save checkpoints
        save_best_model_metric: Metric to monitor
        save_best_model_mode: 'min' or 'max'
        save_best_model_filename: Checkpoint filename
        save_best_model_save_pretrained_fn: Optional HF save function
        save_best_model_dirname: Directory name for HF saves
        save_best_model_track_train_metrics: Monitor train metrics instead of eval
        save_best_model_save_steps: Step interval for batch-level saves (0 = disabled)
    """
    save_dir = Path(save_best_model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return {
        'save_best_model_dir': save_dir,
        'save_best_model_metric': save_best_model_metric,
        'save_best_model_mode': save_best_model_mode,
        'save_best_model_filename': save_best_model_filename,
        'save_best_model_save_pretrained_fn': save_best_model_save_pretrained_fn,
        'save_best_model_dirname': save_best_model_dirname,
        'save_best_model_track_train_metrics': save_best_model_track_train_metrics,
        'save_best_model_save_steps': save_best_model_save_steps,
        'save_best_model_best_value': float('inf') if save_best_model_mode == 'min' else float('-inf'),
        'save_best_model_best_epoch': None,
        'save_best_model_step_loss_sum': 0.0,
        'save_best_model_step_loss_count': 0,
    }


def _is_improvement(current: float, best: float, mode: str) -> bool:
    """Check if current value improves on best.

    Args:
        current: Current metric value
        best: Best metric value seen so far
        mode: 'min' or 'max'

    Returns:
        True if current is an improvement
    """
    if mode == 'min':
        return current < best
    return current > best


def _save_checkpoint(
    state_dict: Dict[str, Any],
    save_dir: Path,
    save_filename: str,
    save_pretrained_fn: Optional[Callable[[Path], None]],
    save_dirname: str,
) -> None:
    """Save checkpoint to disk.

    Args:
        state_dict: Dictionary with model state and metadata
        save_dir: Base directory for saving
        save_filename: Filename for .pt save
        save_pretrained_fn: Optional function for HF format save
        save_dirname: Directory name for HF saves
    """
    if save_pretrained_fn is not None:
        # HuggingFace format: save to directory
        save_path = save_dir / save_dirname
        save_pretrained_fn(save_path)
    else:
        # PyTorch format: save state_dict to file
        save_path = save_dir / save_filename
        torch.save(state_dict, save_path)


@save_best_model_plugin.system('BATCH_AFTER')
def save_best_model_batch_end(
    model,
    loss,
    global_step: int,
    epoch: int,
    is_main_process: bool,
    get_state,
    set_state,
):
    """Check for best model at step intervals during training.

    Args:
        model: Current model
        loss: Current batch loss
        global_step: Current global step
        epoch: Current epoch
        is_main_process: Whether this is the main process (for distributed training)
        get_state: Function to get plugin state
        set_state: Function to set plugin state
    """
    if not is_main_process:
        return

    save_steps = get_state('save_best_model_save_steps')
    if save_steps == 0 or loss is None:
        return

    # Accumulate loss
    step_loss_sum = get_state('save_best_model_step_loss_sum')
    step_loss_count = get_state('save_best_model_step_loss_count')
    step_loss_sum += loss.item()
    step_loss_count += 1

    # Check every save_steps global steps
    if global_step > 0 and global_step % save_steps == 0 and step_loss_count > 0:
        avg_loss = step_loss_sum / step_loss_count
        metric_name = get_state('save_best_model_metric')
        best_value = get_state('save_best_model_best_value')
        mode = get_state('save_best_model_mode')

        is_best = _is_improvement(avg_loss, best_value, mode)
        if is_best:
            best_value = avg_loss
            save_dir = get_state('save_best_model_dir')
            filename = get_state('save_best_model_filename')
            save_fn = get_state('save_best_model_save_pretrained_fn')
            dirname = get_state('save_best_model_dirname')

            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                f'best_{metric_name}': best_value,
            }

            _save_checkpoint(checkpoint, save_dir, filename, save_fn, dirname)
            print(f"  New best train {metric_name}: {avg_loss:.4f} (step {global_step})")

        # Reset for next interval
        set_state({
            'save_best_model_best_value': best_value,
            'save_best_model_step_loss_sum': 0.0,
            'save_best_model_step_loss_count': 0,
        })
    else:
        # Just update counters
        set_state({
            'save_best_model_step_loss_sum': step_loss_sum,
            'save_best_model_step_loss_count': step_loss_count,
        })


@save_best_model_plugin.system('EPOCH_AFTER')
def save_best_model_epoch_end(
    model,
    epoch: int,
    is_main_process: bool,
    get_state,
    set_state,
    train_metrics: Optional[Dict[str, Any]] = None,
):
    """Check if current model is best based on train metrics.

    Args:
        model: Current model
        epoch: Current epoch (0-indexed)
        is_main_process: Whether this is the main process
        train_metrics: Training metrics dictionary
        get_state: Function to get plugin state
        set_state: Function to set plugin state
    """
    if not is_main_process:
        return

    track_train_metrics = get_state('save_best_model_track_train_metrics')
    if not track_train_metrics or train_metrics is None:
        return

    metric_name = get_state('save_best_model_metric')
    current = train_metrics.get(metric_name)
    if current is None:
        return

    best_value = get_state('save_best_model_best_value')
    mode = get_state('save_best_model_mode')

    is_best = _is_improvement(current, best_value, mode)

    if is_best:
        best_value = current
        save_dir = get_state('save_best_model_dir')
        filename = get_state('save_best_model_filename')
        save_fn = get_state('save_best_model_save_pretrained_fn')
        dirname = get_state('save_best_model_dirname')

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            f'best_{metric_name}': best_value,
            'train_metrics': dict(train_metrics),
        }

        # Include EMA model if available
        ema_model = get_state('ema_model')
        if ema_model is not None:
            checkpoint['ema_model_state_dict'] = ema_model.state_dict()

        # Include SWA model if available
        swa_model = get_state('swa_model')
        if swa_model is not None:
            checkpoint['swa_model_state_dict'] = swa_model.state_dict()
            checkpoint['swa_n_averaged'] = get_state('swa_n_averaged')

        _save_checkpoint(checkpoint, save_dir, filename, save_fn, dirname)
        print(f"  New best train {metric_name}: {current:.4f} (epoch {epoch + 1})")

        set_state({'save_best_model_best_value': best_value, 'save_best_model_best_epoch': epoch})


@save_best_model_plugin.system('EVAL_AFTER')
def save_best_model_eval_end(
    model,
    epoch: int,
    global_step: int,
    is_main_process: bool,
    get_state,
    set_state,
    eval_metrics: Optional[Dict[str, Any]] = None,
):
    """Check if current model is best and save after evaluation.

    Args:
        model: Current model
        epoch: Current epoch (0-indexed)
        global_step: Current global step
        is_main_process: Whether this is the main process
        eval_metrics: Evaluation metrics dictionary
        get_state: Function to get plugin state
        set_state: Function to set plugin state
    """
    if not is_main_process:
        return

    track_train = get_state('save_best_model_track_train_metrics')
    if track_train or eval_metrics is None:
        return

    metric_name = get_state('save_best_model_metric')
    current = eval_metrics.get(metric_name)
    if current is None:
        return

    best_value = get_state('save_best_model_best_value')
    mode = get_state('save_best_model_mode')

    is_best = _is_improvement(current, best_value, mode)

    if is_best:
        best_value = current
        save_dir = get_state('save_best_model_dir')
        filename = get_state('save_best_model_filename')
        save_fn = get_state('save_best_model_save_pretrained_fn')
        dirname = get_state('save_best_model_dirname')

        checkpoint = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            f'best_{metric_name}': best_value,
            'eval_metrics': dict(eval_metrics),
        }

        # Include EMA model if available
        ema_model = get_state('ema_model')
        if ema_model is not None:
            checkpoint['ema_model_state_dict'] = ema_model.state_dict()

        # Include SWA model if available
        swa_model = get_state('swa_model')
        if swa_model is not None:
            checkpoint['swa_model_state_dict'] = swa_model.state_dict()
            checkpoint['swa_n_averaged'] = get_state('swa_n_averaged')

        _save_checkpoint(checkpoint, save_dir, filename, save_fn, dirname)
        print(f"  New best {metric_name}: {current:.4f} (epoch {epoch + 1})")

        set_state({'save_best_model_best_value': best_value, 'save_best_model_best_epoch': epoch})


@save_best_model_plugin.system('FIT_AFTER')
def cleanup_save_best_model(get_state, set_state):
    """Clean up best model saver state after training.

    Args:
        get_state: Function to get plugin state
        set_state: Function to set plugin state
    """
    set_state({
        'save_best_model_step_loss_sum': 0.0,
        'save_best_model_step_loss_count': 0,
    })
