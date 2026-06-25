"""Progress bar plugin using tqdm for training and evaluation visualization.

Features:
  - tqdm progress bars for training and evaluation phases
  - is_main_process guard for distributed training
  - Configurable metrics in postfix
  - Automatic cleanup at epoch end and eval end

Usage:
  from coffeetrain.plugins.tqdm_progress import tqdm_progress
  from coffeetrain import TrainerV2

  trainer = TrainerV2()
  trainer.register_plugin(tqdm_progress)

  # Configure metrics to display (default: loss, lr, step)
  # trainer --tqdm_metrics "loss,lr,step"
"""

from typing import Optional
from tqdm import tqdm

from coffeetrain.plugin import Plugin

tqdm_progress = Plugin(
    name='tqdm_progress',
    description='Prints tqdm progress bars to track training and evaluation progress with configurable metrics. Includes is_main_process gating for distributed training.',
)


@tqdm_progress.system('EPOCH_BEFORE')
def create_train_progress_bar(
    train_dataloader,
    epoch,
    max_epochs,
    set_state,
    is_main_process: bool = True,
):
    """Create and store tqdm progress bar for training epoch."""
    if not is_main_process:
        return

    dataloader = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{max_epochs}",
        leave=True,
    )
    return {'train_dataloader': dataloader}


@tqdm_progress.system('BATCH_AFTER')
def update_train_progress(
    loss,
    lr: Optional[float],
    global_step,
    train_dataloader,
    is_main_process: bool = True,
    tqdm_metrics: str = "loss,lr,step",
):
    """Update tqdm postfix with current metrics after each batch."""
    if not is_main_process:
        return

    # Skip if train_dataloader is not wrapped with tqdm
    if not hasattr(train_dataloader, 'set_postfix'):
        return

    # Parse which metrics to display
    metrics_list = [m.strip() for m in tqdm_metrics.split(',')]
    postfix = {}

    if 'loss' in metrics_list and loss is not None:
        postfix['loss'] = f'{loss.item():.4f}'

    if 'lr' in metrics_list and lr is not None:
        postfix['lr'] = f'{lr:.2e}'

    if 'step' in metrics_list:
        postfix['step'] = global_step

    if postfix:
        train_dataloader.set_postfix(postfix)


@tqdm_progress.system('EPOCH_AFTER')
def close_train_progress_bar(
    train_dataloader,
    is_main_process: bool = True,
    set_state=None,
):
    """Close the training progress bar at end of epoch."""
    if not is_main_process:
        return

    # If train_dataloader is wrapped with tqdm, close it
    if hasattr(train_dataloader, 'close'):
        train_dataloader.close()

    # Reset train_dataloader reference to original dataloader if stored
    # The next epoch will rewrap it
    if set_state is not None and hasattr(train_dataloader, 'iterable'):
        set_state({'train_dataloader': train_dataloader.iterable})


@tqdm_progress.system('EVAL_BEFORE')
def create_eval_progress_bar(
    eval_dataloader,
    set_state,
    get_state,
    is_main_process: bool = True,
):
    """Create and store tqdm progress bar for evaluation phase."""
    if not is_main_process or eval_dataloader is None:
        return

    eval_pbar = tqdm(
        eval_dataloader,
        desc="Evaluating",
        leave=True,
    )
    return {'eval_progress_bar': eval_pbar}


@tqdm_progress.system('EVAL_BATCH_AFTER')
def update_eval_progress(
    eval_progress_bar,
    is_main_process: bool = True,
):
    """Update eval progress bar after each eval batch."""
    if not is_main_process or eval_progress_bar is None:
        return

    # tqdm automatically updates, but we could add postfix here if needed
    pass


@tqdm_progress.system('EVAL_AFTER')
def close_eval_progress_bar(
    get_state,
    set_state,
    is_main_process: bool = True,
):
    """Close the evaluation progress bar at end of eval phase."""
    if not is_main_process:
        return

    eval_pbar = get_state('eval_progress_bar')
    if eval_pbar is not None and hasattr(eval_pbar, 'close'):
        eval_pbar.close()
        set_state({'eval_progress_bar': None})
