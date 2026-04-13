"""Lightweight event-driven PyTorch trainer with composable callbacks."""

from coffeetrain.events import Event
from coffeetrain.state import State
from coffeetrain.callback import Callback
from coffeetrain.model import TrainerModel, TrainerModelMixin, ModelWrapper
from coffeetrain.trainer import Trainer
from coffeetrain.schedulers import CosineWarmupScheduler, LinearWarmupScheduler, WSDScheduler
from coffeetrain.optimizers import (
    OptimizerConfig,
    create_optimizer,
    create_optimizer_with_param_groups,
)

from coffeetrain.trainer_v2 import TrainerV2
from coffeetrain.plugins import * # FIXME explicit callbacks

# Callbacks
from coffeetrain.callbacks import (
    CometCallback,
    EMACallback,
    HistoryCallback,
    BestModelCheckpointer,
    LRMonitor,
    ParameterCounter,
    SpeedMonitor,
    ProgressCallback,
    TorchMetricsCallback,
    WandbCallback,
    BatchSizeSchedulerCallback,
    SWACallback,
    ScheduleLoggerCallback,
    EarlyStoppingCallback,
)

# Metrics
from coffeetrain.metrics import MeanLoss, Perplexity, SafeMulticlassAccuracy

__all__ = [
    # Core
    "Event",
    "State",
    "Callback",
    "TrainerModel",
    "TrainerModelMixin",
    "ModelWrapper",
    "Trainer",
    "CosineWarmupScheduler",
    "LinearWarmupScheduler",
    "WSDScheduler",
    # Optimizers
    "OptimizerConfig",
    "create_optimizer",
    "create_optimizer_with_param_groups",
    # Callbacks
    "CometCallback",
    "EMACallback",
    "HistoryCallback",
    "BestModelCheckpointer",
    "LRMonitor",
    "ParameterCounter",
    "SpeedMonitor",
    "ProgressCallback",
    "TorchMetricsCallback",
    "WandbCallback",
    "BatchSizeSchedulerCallback",
    "SWACallback",
    "ScheduleLoggerCallback",
    "EarlyStoppingCallback",
    # Metrics
    "MeanLoss",
    "Perplexity",
    "SafeMulticlassAccuracy",
]
