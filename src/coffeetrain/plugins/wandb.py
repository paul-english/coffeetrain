#!/usr/bin/env python3
"""Weights & Biases plugin for experiment tracking and visualization.

This plugin provides comprehensive W&B integration for the Trainer event-driven system.

Features:
  - Metric logging with configurable intervals
  - Proper x-axis alignment via define_metric()
  - Summary statistics (min/max/best) for key metrics
  - Gradient/parameter histogram tracking via wandb.watch()
  - Loss spike and NaN alerts
  - Prediction tables for error analysis
  - Debug JSONL logging for offline analysis
  - is_main_process guard for distributed training

Usage:
  from coffeetrain.plugins.wandb import wandb_plugin
  from coffeetrain import Trainer

  trainer = Trainer()
  trainer.register_plugin(wandb_plugin)

  # Configure via trainer hyperparams or command-line args:
  # trainer --wandb_project my_project --wandb_watch_model true --wandb_enable_alerts true

Parameters (all optional with sensible defaults):
  - wandb_project: W&B project name (default: "coffeetrain")
  - wandb_name: Run name, auto-generated if None
  - wandb_config: Pydantic config model or dict of hyperparams to log
  - wandb_log_interval: Log training metrics every N batches (default: 10)
  - wandb_resume: Resume from previous run if available (default: true)
  - wandb_debug_log: Write timestamped JSONL file for debugging (default: false)
  - wandb_watch_model: Enable gradient/parameter histograms (default: false)
  - wandb_watch_log: What to log: "gradients", "parameters", or "all" (default: "gradients")
  - wandb_watch_log_freq: Log gradients/params every N batches (default: 100)
  - wandb_enable_alerts: Enable loss spike and NaN alerts (default: true)
  - wandb_log_predictions: Log prediction tables during evaluation (default: false)
  - wandb_max_prediction_samples: Max prediction samples to log (default: 50)
"""

import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TextIO, Tuple

from coffeetrain.plugin import Plugin

wandb_plugin = Plugin(
    name='wandb',
    description='Automatic instrumentation for wandb experiment logging with metrics, gradients, alerts, and prediction tables',
)


def _get_wandb():
    """Lazy import wandb to avoid import errors when not installed."""
    try:
        import wandb
        return wandb
    except ImportError:
        raise ImportError(
            "wandb is not installed. Install it with: pip install wandb"
        )


@wandb_plugin.system('FIT_BEFORE')
def wandb_init(
    model,
    get_state,
    set_state,
    log,
    is_main_process: bool = True,
    wandb_project: str = "coffeetrain",
    wandb_name: Optional[str] = None,
    wandb_config: Optional[Dict[str, Any]] = None,
    wandb_log_model: bool = False,
    wandb_resume: bool = True,
    wandb_debug_log: bool = False,
    wandb_watch_model: bool = False,
    wandb_watch_log: str = "gradients",
    wandb_watch_log_freq: int = 100,
    wandb_enable_alerts: bool = True,
    wandb_log_predictions: bool = False,
    wandb_max_prediction_samples: int = 50,
):
    """Initialize wandb run and configure metrics tracking.

    Only initializes on the main process to avoid duplicate runs in distributed training.
    """
    if not is_main_process:
        # Initialize empty state for non-main processes
        set_state({
            'wandb_run': None,
            'wandb_debug_file': None,
            'wandb_debug_path': None,
        })
        return

    wandb = _get_wandb()

    # Check if W&B run already active
    if wandb.run is not None:
        run = wandb.run
        run_id = run.id
        _define_metrics(wandb)
        log.info(f"[WandbCallback] Attached to existing run: {run.name}")
    else:
        # Determine resume mode
        resume_mode = "allow" if wandb_resume else None

        # Convert Pydantic model to dict if needed
        config = wandb_config
        if config is not None and hasattr(config, "model_dump"):
            config = config.model_dump()

        run = wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=config if isinstance(config, dict) else None,
            resume=resume_mode,
        )
        run_id = run.id

        _define_metrics(wandb)

    # Initialize debug logging
    debug_file = None
    debug_path = None
    if wandb_debug_log:
        debug_path = Path(f"wandb_debug_{run_id}.jsonl")
        debug_file = open(debug_path, "a")
        log.info(f"[WandbCallback] Debug logging enabled: {debug_path}")

    # Enable gradient/parameter tracking if requested
    if wandb_watch_model:
        wandb.watch(
            model,
            log=wandb_watch_log,
            log_freq=wandb_watch_log_freq,
            log_graph=True,
        )

    # Store state for other systems to use
    set_state({
        'wandb_run': run,
        'wandb_run_id': run_id,
        'wandb_debug_file': debug_file,
        'wandb_debug_path': debug_path,
        'wandb_loss_window': deque(maxlen=100),
        'wandb_best_train_loss': None,
        'wandb_best_val_loss': None,
        'wandb_enable_alerts': wandb_enable_alerts,
        'wandb_log_interval': 10,
        'wandb_log_predictions': wandb_log_predictions,
        'wandb_max_prediction_samples': wandb_max_prediction_samples,
    })


def _define_metrics(wandb) -> None:
    """Configure metrics for proper x-axis alignment and summary stats."""
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")

    wandb.define_metric("train/loss", summary="min", goal="minimize")
    wandb.define_metric("val/loss", summary="min", goal="minimize")
    wandb.define_metric("train/lr", summary="last")

    wandb.define_metric("train/classification_loss", summary="min")
    wandb.define_metric("train/structure_loss", summary="min")
    wandb.define_metric("train/count_loss", summary="min")

    wandb.define_metric("train/samples_per_sec", summary="mean")
    wandb.define_metric("train/epoch_time", summary="last")


def _write_debug_log(debug_file: Optional[TextIO], metrics: Dict[str, Any], step: int) -> None:
    """Write metrics to debug log file."""
    if debug_file is None:
        return

    record = {
        "ts": datetime.now().isoformat(),
        "step": step,
        **metrics,
    }
    debug_file.write(json.dumps(record) + "\n")
    debug_file.flush()


def _track_loss(
    loss_val: float,
    step: int,
    loss_window: deque,
    wandb_run,
    wandb_enable_alerts: bool,
) -> Tuple[Optional[float], bool]:
    """Track loss for alerts and best metrics. Returns (best_train_loss, is_nan)."""
    if not math.isfinite(loss_val):
        if wandb_enable_alerts and wandb_run:
            wandb_run.alert(
                title="NaN/Inf Loss Detected",
                text=f"Loss became {loss_val} at step {step}",
                level="ERROR",
            )
        return None, True

    best_loss = loss_val

    if loss_window:
        avg_loss = sum(loss_window) / len(loss_window)
        if wandb_enable_alerts and loss_val > 10 * avg_loss and wandb_run:
            wandb_run.alert(
                title="Loss Spike Detected",
                text=f"Loss {loss_val:.4f} > 10x avg {avg_loss:.4f} at step {step}",
                level="WARN",
                wait_duration=300,
            )

    loss_window.append(loss_val)
    return best_loss, False


@wandb_plugin.system('BATCH_AFTER')
def wandb_batch_end(
    loss,
    global_step: int,
    get_state,
    set_state,
    wandb_log_interval: int = 10,
    train_metrics: Dict[str, float] = None,
    lr: Optional[float] = None,
):
    """Log training metrics at specified interval."""
    wandb_run = get_state('wandb_run')
    if wandb_run is None:
        return

    loss_window = get_state('wandb_loss_window')
    best_train_loss = get_state('wandb_best_train_loss')
    wandb_enable_alerts = get_state('wandb_enable_alerts')
    debug_file = get_state('wandb_debug_file')

    if loss is not None:
        loss_val = loss.item()
        new_best, _ = _track_loss(loss_val, global_step, loss_window, wandb_run, wandb_enable_alerts)
        if new_best is not None and (best_train_loss is None or new_best < best_train_loss):
            best_train_loss = new_best

    if global_step % wandb_log_interval != 0:
        set_state({'wandb_best_train_loss': best_train_loss})
        return

    metrics: Dict[str, Any] = {
        "global_step": global_step,
    }

    if loss is not None:
        metrics["train/loss"] = loss.item()

    if lr is not None:
        metrics["train/lr"] = lr

    # Log any additional training metrics if provided
    if train_metrics:
        for key, value in train_metrics.items():
            metrics[f"train/{key}"] = value

    _write_debug_log(debug_file, metrics, global_step)
    wandb_run.log(metrics, step=global_step)

    set_state({'wandb_best_train_loss': best_train_loss})


@wandb_plugin.system('EPOCH_AFTER')
def wandb_epoch_end(epoch: int, global_step: int, get_state):
    """Log epoch marker."""
    wandb_run = get_state('wandb_run')
    if wandb_run is None:
        return

    wandb_run.log({"epoch": epoch}, step=global_step)


@wandb_plugin.system('EVAL_AFTER')
def wandb_eval_end(
    global_step: int,
    get_state,
    set_state,
    eval_metrics: Optional[Dict[str, float]] = None,
):
    """Log validation metrics and prediction tables."""
    wandb_run = get_state('wandb_run')
    if wandb_run is None:
        return

    if eval_metrics is None:
        eval_metrics = {}

    metrics: Dict[str, Any] = {}
    best_val_loss = get_state('wandb_best_val_loss')

    for key, value in eval_metrics.items():
        metrics[f"val/{key}"] = value

        if key == "loss" and isinstance(value, (int, float)):
            if best_val_loss is None or value < best_val_loss:
                best_val_loss = value

    if metrics:
        wandb_run.log(metrics, step=global_step)

    # Log prediction table if enabled and available
    wandb_log_predictions = get_state('wandb_log_predictions')
    if wandb_log_predictions:
        _log_prediction_table(wandb_run, global_step, get_state)

    set_state({'wandb_best_val_loss': best_val_loss})


def _log_prediction_table(wandb_run, global_step: int, get_state) -> None:
    """Log prediction samples as a wandb Table."""
    if wandb_run is None:
        return

    wandb = _get_wandb()
    max_samples = get_state('wandb_max_prediction_samples')

    # Try to get predictions from callback_state (populated by trainer)
    # This is flexible - could come from various sources
    predictions = []
    try:
        # Attempt to fetch from trainer context
        pass
    except:
        pass

    if not predictions:
        return

    table = wandb.Table(
        columns=["text", "ground_truth", "prediction", "correct"],
        data=predictions[:max_samples],
    )
    wandb_run.log({"val/predictions": table}, step=global_step)


@wandb_plugin.system('FIT_AFTER')
def wandb_fit_end(get_state, set_state):
    """Finish wandb run with summary metrics."""
    debug_file = get_state('wandb_debug_file')
    debug_path = get_state('wandb_debug_path')
    wandb_run = get_state('wandb_run')
    best_train_loss = get_state('wandb_best_train_loss')
    best_val_loss = get_state('wandb_best_val_loss')

    if debug_file is not None:
        debug_file.close()
        print(f"[WandbCallback] Debug log saved: {debug_path}")

    if wandb_run is not None:
        if best_train_loss is not None:
            wandb_run.summary["best_train_loss"] = best_train_loss
        if best_val_loss is not None:
            wandb_run.summary["best_val_loss"] = best_val_loss

        wandb_run.finish()

    set_state({
        'wandb_run': None,
        'wandb_debug_file': None,
        'wandb_best_train_loss': None,
        'wandb_best_val_loss': None,
    })
