"""Gradient accumulation plugin for efficient training with large effective batch sizes."""

from coffeetrain.plugin import Plugin

gradient_accumulator = Plugin(
    name='gradient_accumulator',
    description='Enables gradient accumulation with flexible batch size control.'
)


@gradient_accumulator.system('FIT_BEFORE')
def init_grad_accum(batch_size: int, real_batch_size: int):
    """Initialize gradient accumulation steps based on batch size ratio.

    Args:
        batch_size: Effective batch size (desired training batch size)
        real_batch_size: Physical batch size per forward pass (microbatch size)

    Returns:
        State dict with computed grad_accum value

    Raises:
        ValueError: If batch_size is not divisible by real_batch_size
    """
    if batch_size % real_batch_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by "
            f"real_batch_size ({real_batch_size})"
        )

    grad_accum = batch_size // real_batch_size
    return {'grad_accum': grad_accum}


@gradient_accumulator.override_block('BACKWARD')
def run_backward(loss, batch_idx: int, grad_accum: int, optimizer):
    """Backward pass with gradient accumulation.

    Accumulates gradients over multiple batches, then performs optimizer step
    and zero_grad when accumulated enough batches.

    Args:
        loss: Scalar loss tensor
        batch_idx: Current batch index within epoch
        grad_accum: Number of batches to accumulate before optimizer step
        optimizer: PyTorch optimizer instance
    """
    loss.backward()

    # Perform optimizer step every grad_accum batches
    if (batch_idx + 1) % grad_accum == 0:
        optimizer.step()
        optimizer.zero_grad()
