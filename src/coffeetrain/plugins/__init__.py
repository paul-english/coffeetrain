from coffeetrain.plugins.interruptable_train import train_plugin
from coffeetrain.plugins.tqdm_progress import tqdm_progress
from coffeetrain.plugins.early_stopping import early_stopping
from coffeetrain.plugins.cuda_accelerate import cuda_accelerate
from coffeetrain.plugins.torch_compile import torch_compile
from coffeetrain.plugins.comet import comet_plugin
from coffeetrain.plugins.lr_monitor import lr_monitor_plugin

default_plugins = [
    train_plugin,
    tqdm_progress,
    early_stopping,
    cuda_accelerate,
    torch_compile,
]
