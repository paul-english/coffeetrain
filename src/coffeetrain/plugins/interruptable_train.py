from enum import Enum
import signal
import asyncio

import torch

from coffeetrain.plugin import Plugin

train_plugin = Plugin(
    name='train',
    description="A safely interruptable training loop. Includes standard training loop events, and an event to clean up on interrupt."
)

# FIXME rename StopTrainer, move into trainerv2 and have a good
# catch for it
class StopLoop(Exception):
    """Exception used to stop the execution of the training loop."""

@train_plugin.event()
class TrainEvents(str, Enum):
    DATA = 'DATA'
    MODEL = 'MODEL'
    OPTIMIZER = 'OPTIMIZER'
    FIT = 'FIT'
    TRAINING_INTERRUPTED = "TRAINING_INTERRUPTED"
    EPOCH = 'EPOCH'
    BATCH = 'BATCH'
    FORWARD = 'FORWARD'
    LOSS = 'LOSS'
    BACKWARD = 'BACKWARD'
    EVAL = 'EVAL'
    EVAL_BATCH = 'EVAL_BATCH'
    EVAL_FORWARD = 'EVAL_FORWARD'
    EVAL_LOSS = 'EVAL_LOSS'

@train_plugin.system('FIT_BEFORE')
async def setup_signal_handlers(get_state, set_state, log):
    def _handle_interrupt():
        """Handle interrupt signal (SIGINT/SIGTERM)."""
        interrupted = get_state('interrupted')
        if interrupted:
            # Second interrupt - force exit
            log.info("\nForced exit requested.")
            raise KeyboardInterrupt
        set_state({
            'interrupted': True,
            'stop_training': True,
        })
        log.info("\nGraceful stop requested. Finishing current batch...")

    loop = asyncio.get_event_loop()
    interrupted = False
    original_sigint = loop.add_signal_handler(signal.SIGINT, _handle_interrupt)
    original_sigterm = loop.add_signal_handler(signal.SIGTERM, _handle_interrupt)
    return {
        'interrupted': interrupted,
        'original_sigint': original_sigint,
        'original_sigterm': original_sigterm,
    }

@train_plugin.system('FIT_BEFORE')
def setup_early_state_vars():
    return {
        'should_run_eval': False,
        'global_step': 0,
        'loss': None,
    }

@train_plugin.system('FIT_AFTER')
def restore_signal_handlers(original_sigint, original_sigterm):
    """Restore original signal handlers."""
    if original_sigint is not None:
        signal.signal(signal.SIGINT, original_sigint)
    if original_sigterm is not None:
        signal.signal(signal.SIGTERM, original_sigterm)

@train_plugin.system(['LOSS', 'EVAL_LOSS'])
def compute_loss(loss_fn, outputs, labels):
    loss = loss_fn(outputs, labels)
    return {'loss': loss}

@train_plugin.system('BACKWARD')
def compute_gradient(loss):
    loss.backward()

@train_plugin.system('BATCH_AFTER')
def update_global_step(global_step):
    return {'global_step': global_step + 1}

@train_plugin.system('BATCH_AFTER')
async def check_if_interrupted(interrupted: bool, run_event):
    if interrupted:
        await run_event(TrainEvents.TRAINING_INTERRUPTED)
        raise StopLoop('Training has been interrupted by signal.')

@train_plugin.system('EVAL_BEFORE')
def zero_eval_loss_accumulators():
    return {
        'total_eval_loss': 0.0,
        'num_eval_batches': 0,
    }

@train_plugin.system('EVAL_FORWARD_AFTER')
def update_num_eval_batches(num_eval_batches):
    return {'num_eval_batches': num_eval_batches + 1}

@train_plugin.system('EVAL_LOSS_AFTER')
def update_eval_total_loss(total_eval_loss, loss):
    total_eval_loss += loss.item()
    return {'total_eval_loss': total_eval_loss}

@train_plugin.system('EVAL_AFTER')
def set_eval_loss(total_eval_loss, num_eval_batches):
    return {'eval_loss': total_eval_loss / num_eval_batches}

@train_plugin.system('EPOCH_AFTER')
def check_if_eval_step(epoch, eval_dataloader, eval_interval=1):
    if eval_dataloader is None:
        return {'should_run_eval': False}
    elif (epoch + 1) % eval_interval == 0:
        return {'should_run_eval': True}
    return {'should_run_eval': False}

@train_plugin.command()
async def train(log, run_event, get_state, execution_block, lr=5e-5, max_epochs=10):
    """Runs the training loop."""
    await execution_block(TrainEvents.DATA, call=True)
    await execution_block(TrainEvents.MODEL, call=True)
    await execution_block(TrainEvents.OPTIMIZER, call=True)
    await run_event(TrainEvents.FIT, before=True)
    try:
        await run_event(TrainEvents.FIT)
        for epoch in range(max_epochs):
            async with execution_block(TrainEvents.EPOCH, {'epoch': epoch}):
                train_dataloader = get_state('train_dataloader')
                for batch_idx, batch in enumerate(train_dataloader):
                    new_batch = {
                        'batch_idx': batch_idx,
                        'batch': batch,
                    }
                    async with execution_block(TrainEvents.BATCH, new_batch) as context:
                        await execution_block(TrainEvents.FORWARD, requires='outputs', call=True)
                        await execution_block(TrainEvents.LOSS, requires='loss', call=True)
                        await execution_block(TrainEvents.BACKWARD, call=True)
            if not get_state('should_run_eval'):
                continue
            async with execution_block(TrainEvents.EVAL):
                eval_dataloader = get_state('eval_dataloader')
                with torch.no_grad():
                    for batch_idx, batch in enumerate(eval_dataloader):
                        new_batch = {
                            'batch_idx': batch_idx,
                            'batch': batch
                        }
                        async with execution_block(TrainEvents.EVAL_BATCH, new_batch):
                            await execution_block(TrainEvents.EVAL_FORWARD, requires='outputs', call=True)
                            await execution_block(TrainEvents.EVAL_LOSS, requires='loss', call=True)
    finally:
        await run_event(TrainEvents.FIT, after=True)
