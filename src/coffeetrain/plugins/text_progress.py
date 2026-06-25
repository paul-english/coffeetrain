"""Text-log style progress plugin for clean, readable training output.

Features:
  - Clean formatted log output without progress bars
  - Epoch summaries with key metrics
  - Optional batch-level loss reporting
  - is_main_process guard for distributed training
  - Similar to V1 ProgressCallback

Usage:
  from coffeetrain.plugins.text_progress import text_progress
  from coffeetrain import TrainerV2

  trainer = TrainerV2()
  trainer.register_plugin(text_progress)

  # Configure:
  # trainer --text_print_batch_loss true --text_batch_log_interval 100
"""

from typing import Optional, Dict, Any

from coffeetrain.plugin import Plugin

text_progress = Plugin(
    name='text_progress',
    description='Clean text-log style progress output with epoch summaries and optional batch loss reporting. Includes is_main_process gating.',
)


@text_progress.system('FIT_BEFORE')
def print_training_start(
    max_epochs: int,
    is_main_process: bool = True,
):
    """Print training start message."""
    if not is_main_process:
        return

    print("\n" + "=" * 60)
    print(f"Starting training for {max_epochs} epochs")
    print("=" * 60)


@text_progress.system('EPOCH_BEFORE')
def print_epoch_start(
    epoch: int,
    max_epochs: int,
    is_main_process: bool = True,
):
    """Print epoch start message."""
    if not is_main_process:
        return

    print(f"\nEpoch {epoch + 1}/{max_epochs}")
    print("-" * 40)


@text_progress.system('BATCH_AFTER')
def print_batch_progress(
    loss,
    lr: Optional[float],
    batch_idx: int,
    global_step: int,
    is_main_process: bool = True,
    text_print_batch_loss: bool = False,
    text_batch_log_interval: int = 100,
):
    """Print batch progress at specified intervals."""
    if not is_main_process or not text_print_batch_loss:
        return

    if (batch_idx + 1) % text_batch_log_interval == 0:
        loss_val = loss.item() if loss is not None else 0.0
        lr_str = f", lr={lr:.2e}" if lr is not None else ""
        print(f"  Batch {batch_idx + 1}: loss={loss_val:.4f}{lr_str}, step={global_step}")


@text_progress.system('EPOCH_AFTER')
def print_epoch_summary(
    epoch: int,
    loss: Optional[Any],
    lr: Optional[float],
    train_metrics: Optional[Dict[str, float]] = None,
    is_main_process: bool = True,
):
    """Print epoch summary with final metrics."""
    if not is_main_process:
        return

    parts = []

    # Add loss from train_metrics or current loss
    if train_metrics and 'loss' in train_metrics:
        parts.append(f"loss={train_metrics['loss']:.4f}")
    elif loss is not None:
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        parts.append(f"loss={loss_val:.4f}")

    # Add learning rate
    if train_metrics and 'learning_rate' in train_metrics:
        parts.append(f"lr={train_metrics['learning_rate']:.2e}")
    elif lr is not None:
        parts.append(f"lr={lr:.2e}")

    if parts:
        print(f"  Train: {', '.join(parts)}")


@text_progress.system('EVAL_AFTER')
def print_eval_summary(
    eval_metrics: Optional[Dict[str, float]] = None,
    is_main_process: bool = True,
):
    """Print evaluation summary with val metrics."""
    if not is_main_process or not eval_metrics:
        return

    parts = [f"{k}={v:.4f}" for k, v in eval_metrics.items()]
    print(f"  Val: {', '.join(parts)}")


@text_progress.system('FIT_AFTER')
def print_training_complete(
    global_step: int,
    is_main_process: bool = True,
):
    """Print training completion message."""
    if not is_main_process:
        return

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total steps: {global_step}")
    print("=" * 60 + "\n")
