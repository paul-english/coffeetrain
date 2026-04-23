#!/usr/bin/env python3
"""Comet.ml plugin for experiment tracking and visualization.

This plugin provides Comet.ml integration for the TrainerV2 event-driven system.

Features:
  - Metric logging with configurable intervals
  - Experiment tracking and resumption via experiment key
  - Configuration logging
  - Best loss tracking for summary metrics

Usage:
  from coffeetrain.plugins.comet import comet_plugin
  from coffeetrain import TrainerV2

  trainer = TrainerV2()
  trainer.register_plugin(comet_plugin)

  # Configure via trainer hyperparams or command-line args:
  # trainer --comet_project my_project --comet_resume true

Parameters (all optional with sensible defaults):
  - comet_project: Comet project name (default: "coffeetrain")
  - comet_workspace: Comet workspace (default: None, uses default workspace)
  - comet_name: Experiment name, auto-generated if None (default: None)
  - comet_config: Pydantic config model or dict of hyperparams to log (default: None)
  - comet_log_interval: Log training metrics every N batches (default: 10)
  - comet_resume: Resume from previous experiment if available (default: true)
"""

from typing import Any, Dict, Optional

from coffeetrain.plugin import Plugin

comet_plugin = Plugin(
    name='comet',
    description='Automatic instrumentation for Comet.ml experiment logging with metrics and experiment tracking',
)


def _get_comet():
    """Lazy import comet_ml to avoid import errors when not installed."""
    try:
        import comet_ml
        return comet_ml
    except ImportError:
        raise ImportError(
            "comet_ml is not installed. Install it with: pip install comet-ml"
        )


@comet_plugin.system('FIT_BEFORE')
def comet_init(
    get_state,
    set_state,
    log,
    is_main_process: bool = True,
    comet_project: str = "coffeetrain",
    comet_workspace: Optional[str] = None,
    comet_name: Optional[str] = None,
    comet_config: Optional[Dict[str, Any]] = None,
    comet_log_interval: int = 10,
    comet_resume: bool = True,
):
    """Initialize Comet experiment and configure metrics tracking.

    Only initializes on the main process to avoid duplicate experiments in distributed training.
    """
    if not is_main_process:
        set_state({
            'comet_experiment': None,
            'comet_experiment_key': None,
        })
        return

    comet = _get_comet()

    # Convert Pydantic model to dict if needed
    config = comet_config
    if config is not None and hasattr(config, "model_dump"):
        config = config.model_dump()
    elif not isinstance(config, dict):
        config = None

    # Create or resume experiment
    experiment = None
    if comet_resume:
        # Try to get existing experiment key from state
        experiment_key = get_state('comet_experiment_key')
        if experiment_key:
            try:
                experiment = comet.ExistingExperiment(
                    previous_experiment=experiment_key,
                    project_name=comet_project,
                    workspace=comet_workspace,
                )
                log.info(f"[CometPlugin] Resumed experiment: {experiment_key}")
            except Exception as e:
                log.warning(f"[CometPlugin] Failed to resume experiment {experiment_key}: {e}")
                experiment = None

    if experiment is None:
        experiment = comet.Experiment(
            project_name=comet_project,
            workspace=comet_workspace,
        )
        if comet_name:
            experiment.set_name(comet_name)

    experiment_key = experiment.get_key()

    # Log hyperparameters
    if config:
        experiment.log_parameters(config)

    log.info(f"[CometPlugin] Initialized experiment: {experiment_key}")

    set_state({
        'comet_experiment': experiment,
        'comet_experiment_key': experiment_key,
        'comet_log_interval': comet_log_interval,
        'comet_best_train_loss': None,
        'comet_best_val_loss': None,
    })


@comet_plugin.system('BATCH_AFTER')
def comet_batch_end(
    loss,
    global_step: int,
    get_state,
    set_state,
    comet_log_interval: int = 10,
    train_metrics: Optional[Dict[str, float]] = None,
    lr: Optional[float] = None,
):
    """Log training metrics at specified interval."""
    experiment = get_state('comet_experiment')
    if experiment is None:
        return

    # Track best train loss
    best_train_loss = get_state('comet_best_train_loss')
    if loss is not None:
        loss_val = loss.item()
        if best_train_loss is None or loss_val < best_train_loss:
            best_train_loss = loss_val
        set_state({'comet_best_train_loss': best_train_loss})

    # Only log at specified interval
    if global_step % comet_log_interval != 0:
        return

    # Log training loss
    if loss is not None:
        experiment.log_metric(
            "train/loss", loss.item(), step=global_step
        )

    # Log learning rate
    if lr is not None:
        experiment.log_metric("train/lr", lr, step=global_step)

    # Log any additional training metrics
    if train_metrics:
        for key, value in train_metrics.items():
            experiment.log_metric(f"train/{key}", value, step=global_step)


@comet_plugin.system('EPOCH_AFTER')
def comet_epoch_end(epoch: int, global_step: int, get_state):
    """Log epoch marker."""
    experiment = get_state('comet_experiment')
    if experiment is None:
        return

    experiment.log_metric("epoch", epoch, step=global_step)


@comet_plugin.system('EVAL_AFTER')
def comet_eval_end(
    global_step: int,
    get_state,
    set_state,
    eval_metrics: Optional[Dict[str, float]] = None,
):
    """Log validation metrics."""
    experiment = get_state('comet_experiment')
    if experiment is None:
        return

    if eval_metrics is None:
        eval_metrics = {}

    best_val_loss = get_state('comet_best_val_loss')

    for key, value in eval_metrics.items():
        experiment.log_metric(f"val/{key}", value, step=global_step)

        # Track best val loss
        if key == "loss" and isinstance(value, (int, float)):
            if best_val_loss is None or value < best_val_loss:
                best_val_loss = value

    if best_val_loss is not None:
        set_state({'comet_best_val_loss': best_val_loss})


@comet_plugin.system('FIT_AFTER')
def comet_fit_end(get_state, set_state):
    """Finish Comet experiment with summary metrics."""
    experiment = get_state('comet_experiment')
    if experiment is not None:
        best_train_loss = get_state('comet_best_train_loss')
        best_val_loss = get_state('comet_best_val_loss')

        if best_train_loss is not None:
            experiment.log_metric("best_train_loss", best_train_loss)
        if best_val_loss is not None:
            experiment.log_metric("best_val_loss", best_val_loss)

        experiment.end()

    set_state({
        'comet_experiment': None,
        'comet_experiment_key': None,
        'comet_best_train_loss': None,
        'comet_best_val_loss': None,
    })
