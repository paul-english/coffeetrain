from coffeetrain.plugins.interruptable_train import train_plugin
from coffeetrain.plugins.tqdm_progress import tqdm_progress
from coffeetrain.plugins.text_progress import text_progress
from coffeetrain.plugins.early_stopping import early_stopping
from coffeetrain.plugins.cuda_accelerate import cuda_accelerate
from coffeetrain.plugins.torch_compile import torch_compile
from coffeetrain.plugins.comet import comet_plugin
from coffeetrain.plugins.lr_monitor import lr_monitor_plugin
from coffeetrain.plugins.parameter_counter import parameter_counter_plugin
from coffeetrain.plugins.ema import ema_plugin
from coffeetrain.plugins.swa import swa_plugin
from coffeetrain.plugins.gradient_accumulator import gradient_accumulator
from coffeetrain.plugins.batch_size_scheduler import batch_size_scheduler
from coffeetrain.plugins.save_best_model import save_best_model_plugin
from coffeetrain.plugins.wandb import wandb_plugin
from coffeetrain.plugins.hf_accelerator import hf_accelerator

__all__ = [
    'batch_size_scheduler', 'comet_plugin', 'cuda_accelerate',
    'default_plugins', 'early_stopping', 'ema_plugin',
    'gradient_accumulator', 'hf_accelerator', 'lr_monitor_plugin', 'parameter_counter_plugin',
    'save_best_model_plugin', 'swa_plugin', 'text_progress', 'torch_compile',
    'tqdm_progress', 'train_plugin', 'wandb_plugin',
]

default_plugins = [
    train_plugin,
    tqdm_progress,
    early_stopping,
    cuda_accelerate,
    torch_compile,
]
