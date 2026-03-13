"""Generic callback implementations for the trainer."""

from coffeetrain.callbacks.batch_size_scheduler import BatchSizeSchedulerCallback
from coffeetrain.callbacks.checkpointing import BestModelCheckpointer
from coffeetrain.callbacks.comet import CometCallback
from coffeetrain.callbacks.early_stopping import EarlyStoppingCallback
from coffeetrain.callbacks.ema import EMACallback
from coffeetrain.callbacks.swa import SWACallback
from coffeetrain.callbacks.history import HistoryCallback
from coffeetrain.callbacks.lr_monitor import LRMonitor
from coffeetrain.callbacks.parameter_counter import ParameterCounter
from coffeetrain.callbacks.progress import ProgressCallback
from coffeetrain.callbacks.speed_monitor import SpeedMonitor
from coffeetrain.callbacks.torchmetrics_callback import TorchMetricsCallback
from coffeetrain.callbacks.wandb import WandbCallback
from coffeetrain.callbacks.schedule_logger import ScheduleLoggerCallback

__all__ = [
    "BatchSizeSchedulerCallback",
    "BestModelCheckpointer",
    "CometCallback",
    "EarlyStoppingCallback",
    "EMACallback",
    "SWACallback",
    "HistoryCallback",
    "LRMonitor",
    "ParameterCounter",
    "ProgressCallback",
    "ScheduleLoggerCallback",
    "SpeedMonitor",
    "TorchMetricsCallback",
    "WandbCallback",
]
