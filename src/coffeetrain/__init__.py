"""Lightweight event-driven PyTorch trainer with composable plugins."""

from coffeetrain.plugin import Plugin
from coffeetrain.trainer import Trainer
from coffeetrain.plugins import (
    batch_size_scheduler,
    comet_plugin,
    cuda_accelerate,
    default_plugins,
    ema_plugin,
    gradient_accumulator,
    lr_monitor_plugin,
    parameter_counter_plugin,
    save_best_model_plugin,
    swa_plugin,
    text_progress,
    tqdm_progress,
    train_plugin,
    wandb_plugin,
)

__all__ = [
    "Trainer",
    "Plugin",
    "batch_size_scheduler",
    "comet_plugin",
    "cuda_accelerate",
    "default_plugins",
    "ema_plugin",
    "gradient_accumulator",
    "lr_monitor_plugin",
    "parameter_counter_plugin",
    "save_best_model_plugin",
    "swa_plugin",
    "text_progress",
    "tqdm_progress",
    "train_plugin",
    "wandb_plugin",
]
