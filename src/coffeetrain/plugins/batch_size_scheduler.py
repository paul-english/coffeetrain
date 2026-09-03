#!/usr/bin/env python3
"""Batch size scheduling plugin for Trainer.

This plugin schedules batch size warmup during training, gradually increasing
the effective batch size from start_batch_size to final_batch_size over
warmup_steps.

The plugin works in conjunction with the gradient_accumulator plugin:
- batch_size_scheduler manages the batch_size parameter over a schedule
- gradient_accumulator computes grad_accum = batch_size / real_batch_size

This design keeps them independent: the scheduler doesn't need to know about
gradient accumulation details, it just manages batch_size.

Two schedule types are available:
  - "linear": Smooth linear interpolation (default)
  - "equal_steps": Equal optimizer update steps at each discrete batch size
    level (from ModernBERT's "uneven token schedule")

Example:
  from coffeetrain.plugins.batch_size_scheduler import batch_size_scheduler
  from coffeetrain import Trainer

  trainer = Trainer()
  trainer.register_plugin(batch_size_scheduler)

  # Configure via trainer hyperparams or command-line args:
  # trainer --start_batch_size 768 \\
  #         --final_batch_size 4608 \\
  #         --batch_size_warmup_steps 10000 \\
  #         --batch_size_schedule_type equal_steps

Parameters (all optional with sensible defaults):
  - start_batch_size: Initial effective batch size (default: same as batch_size)
  - final_batch_size: Target effective batch size (default: same as batch_size)
  - batch_size_warmup_steps: Number of optimizer steps to warmup over (default: 0)
  - batch_size_schedule_type: 'linear' or 'equal_steps' (default: 'linear')
  - real_batch_size: Physical batch size per forward pass (default: batch_size)
"""

from typing import Literal, Optional

from coffeetrain.plugin import Plugin

batch_size_scheduler = Plugin(
    name='batch_size_scheduler',
    description='Schedule batch size warmup with linear or equal_steps schedules',
)


@batch_size_scheduler.system('FIT_BEFORE')
def init_batch_size_scheduler(
    batch_size: int,
    real_batch_size: int,
    start_batch_size: Optional[int] = None,
    final_batch_size: Optional[int] = None,
    batch_size_warmup_steps: int = 0,
    batch_size_schedule_type: Literal['linear', 'equal_steps'] = 'linear',
):
    """Initialize batch size scheduler.

    Args:
        batch_size: Current effective batch size (used as default for start/final)
        real_batch_size: Physical batch size per forward pass
        start_batch_size: Initial effective batch size (default: batch_size)
        final_batch_size: Target effective batch size (default: batch_size)
        batch_size_warmup_steps: Number of optimizer steps to warmup over (default: 0)
        batch_size_schedule_type: 'linear' for smooth interpolation,
            'equal_steps' for equal optimizer steps at each batch size level

    Returns:
        State dict with scheduler parameters

    Raises:
        ValueError: If batch sizes are invalid or not divisible by real_batch_size
    """
    # Use defaults if not specified
    start = start_batch_size if start_batch_size is not None else batch_size
    final = final_batch_size if final_batch_size is not None else batch_size

    # Validation
    if start % real_batch_size != 0:
        raise ValueError(
            f"start_batch_size ({start}) must be divisible by "
            f"real_batch_size ({real_batch_size})"
        )
    if final % real_batch_size != 0:
        raise ValueError(
            f"final_batch_size ({final}) must be divisible by "
            f"real_batch_size ({real_batch_size})"
        )
    if start > final:
        raise ValueError(
            f"start_batch_size ({start}) must be <= final_batch_size ({final})"
        )

    start_grad_accum = start // real_batch_size
    final_grad_accum = final // real_batch_size

    # Pre-compute for equal_steps schedule
    num_levels = final_grad_accum - start_grad_accum + 1
    steps_per_level = (
        batch_size_warmup_steps / num_levels if num_levels > 0 and batch_size_warmup_steps > 0 else 1
    )

    state = {
        'batch_size_start': start,
        'batch_size_final': final,
        'batch_size_warmup_steps': batch_size_warmup_steps,
        'batch_size_schedule_type': batch_size_schedule_type,
        'batch_size_real_batch_size': real_batch_size,
        'batch_size_start_grad_accum': start_grad_accum,
        'batch_size_final_grad_accum': final_grad_accum,
        'batch_size_num_levels': num_levels,
        'batch_size_steps_per_level': steps_per_level,
        'batch_size_last_batch_size': start,
        'batch_size_last_grad_accum': start_grad_accum,
    }

    # Log initialization
    if batch_size_warmup_steps > 0:
        print(
            f"[BatchSizeScheduler] initialized: "
            f"batch_size={start} (grad_accum={start_grad_accum}), "
            f"warmup_steps={batch_size_warmup_steps}, "
            f"final_batch_size={final} (grad_accum={final_grad_accum}), "
            f"schedule={batch_size_schedule_type}"
        )

    return state


def _compute_batch_size(
    global_step: int,
    start_batch_size: int,
    final_batch_size: int,
    warmup_steps: int,
    schedule_type: str,
    start_grad_accum: int,
    final_grad_accum: int,
    num_levels: int,
    steps_per_level: float,
    real_batch_size: int,
) -> tuple[int, int]:
    """Compute batch_size and grad_accum for a given step.

    Args:
        global_step: Current optimizer step
        start_batch_size: Starting effective batch size
        final_batch_size: Final effective batch size
        warmup_steps: Total warmup steps
        schedule_type: 'linear' or 'equal_steps'
        start_grad_accum: Starting gradient accumulation
        final_grad_accum: Final gradient accumulation
        num_levels: Number of discrete batch size levels (for equal_steps)
        steps_per_level: Steps per level (for equal_steps)
        real_batch_size: Physical batch size for validation

    Returns:
        Tuple of (batch_size, grad_accum)
    """
    if global_step >= warmup_steps:
        return final_batch_size, final_grad_accum

    if warmup_steps == 0:
        return start_batch_size, start_grad_accum

    if schedule_type == 'equal_steps':
        # Equal optimizer steps at each batch size level
        level = int(global_step / steps_per_level)
        level = min(level, final_grad_accum - start_grad_accum)
        grad_accum = start_grad_accum + level
        batch_size = grad_accum * real_batch_size
        return batch_size, grad_accum
    else:
        # Linear interpolation (default)
        progress = global_step / max(1, warmup_steps)
        grad_accum_float = (
            start_grad_accum
            + progress * (final_grad_accum - start_grad_accum)
        )
        grad_accum = max(1, round(grad_accum_float))
        batch_size = grad_accum * real_batch_size
        return batch_size, grad_accum


@batch_size_scheduler.system('BATCH_AFTER')
def update_batch_size(
    global_step: int,
    get_state,
    set_state,
):
    """Update batch size after each optimizer step.

    Args:
        global_step: Current optimizer step
        get_state: Function to get plugin state
        set_state: Function to set plugin state

    Returns:
        Dict with updated batch_size and grad_accum if they changed, else None
    """
    start = get_state('batch_size_start')
    final = get_state('batch_size_final')
    warmup_steps = get_state('batch_size_warmup_steps')
    schedule_type = get_state('batch_size_schedule_type')
    start_grad_accum = get_state('batch_size_start_grad_accum')
    final_grad_accum = get_state('batch_size_final_grad_accum')
    num_levels = get_state('batch_size_num_levels')
    steps_per_level = get_state('batch_size_steps_per_level')
    real_batch_size = get_state('batch_size_real_batch_size')

    if warmup_steps == 0:
        return

    new_batch_size, new_grad_accum = _compute_batch_size(
        global_step=global_step,
        start_batch_size=start,
        final_batch_size=final,
        warmup_steps=warmup_steps,
        schedule_type=schedule_type,
        start_grad_accum=start_grad_accum,
        final_grad_accum=final_grad_accum,
        num_levels=num_levels,
        steps_per_level=steps_per_level,
        real_batch_size=real_batch_size,
    )

    last_batch_size = get_state('batch_size_last_batch_size')
    last_grad_accum = get_state('batch_size_last_grad_accum')

    # Only return and log if values changed
    if new_batch_size != last_batch_size or new_grad_accum != last_grad_accum:
        print(
            f"[BatchSizeScheduler] step={global_step}: "
            f"batch_size {last_batch_size} -> {new_batch_size}, "
            f"grad_accum {last_grad_accum} -> {new_grad_accum}"
        )
        set_state({
            'batch_size_last_batch_size': new_batch_size,
            'batch_size_last_grad_accum': new_grad_accum,
        })
        return {
            'batch_size': new_batch_size,
            'grad_accum': new_grad_accum,
        }

    # Update state even if values didn't change (for tracking)
    set_state({
        'batch_size_last_batch_size': new_batch_size,
        'batch_size_last_grad_accum': new_grad_accum,
    })


@batch_size_scheduler.system('FIT_AFTER')
def cleanup_batch_size_scheduler(get_state, set_state):
    """Clean up batch size scheduler state.

    Args:
        get_state: Function to get plugin state
        set_state: Function to set plugin state
    """
    set_state({
        'batch_size_start': 0,
        'batch_size_final': 0,
        'batch_size_warmup_steps': 0,
        'batch_size_schedule_type': 'linear',
        'batch_size_real_batch_size': 0,
        'batch_size_start_grad_accum': 0,
        'batch_size_final_grad_accum': 0,
        'batch_size_num_levels': 0,
        'batch_size_steps_per_level': 0,
        'batch_size_last_batch_size': 0,
        'batch_size_last_grad_accum': 0,
    })
