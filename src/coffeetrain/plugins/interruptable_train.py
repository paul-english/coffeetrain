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
    """Capture original SIGINT/SIGTERM handlers so FIT_AFTER can restore them.

    Async so it runs on the event-loop thread (pytest runs sync systems in an
    executor thread, where `signal.signal` raises `ValueError`).
    """
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

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        try:
            loop.add_signal_handler(signal.SIGINT, _handle_interrupt)
            loop.add_signal_handler(signal.SIGTERM, _handle_interrupt)
        except NotImplementedError:
            # e.g. Windows: fall back to synchronous handlers
            signal.signal(signal.SIGINT, lambda *_: _handle_interrupt())
            signal.signal(signal.SIGTERM, lambda *_: _handle_interrupt())
    else:
        signal.signal(signal.SIGINT, lambda *_: _handle_interrupt())
        signal.signal(signal.SIGTERM, lambda *_: _handle_interrupt())
    return {
        'interrupted': False,
        'stop_training': False,
        'original_sigint': original_sigint,
        'original_sigterm': original_sigterm,
    }

@train_plugin.system('FIT_BEFORE')
def setup_early_state_vars():
    return {
        'should_run_eval': False,
        'global_step': 0,
        'loss': None,
        'stop_training': False,
    }

@train_plugin.system('FIT_AFTER')
async def restore_signal_handlers(get_state, log):
    """Restore original signal handlers captured in FIT_BEFORE."""
    for sig, key in ((signal.SIGINT, 'original_sigint'),
                      (signal.SIGTERM, 'original_sigterm')):
        original = get_state(key)
        if original is None:
            continue
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            try:
                loop.remove_signal_handler(sig)
            except Exception:
                pass
        try:
            signal.signal(sig, original)
        except (OSError, ValueError, RuntimeError) as e:
            log.debug(f'Could not restore signal handler for {sig}: {e}')

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


@train_plugin.system('BATCH_AFTER')
def check_if_should_stop(stop_training: bool = False):
    if stop_training:
        raise StopLoop('Training stopped via `stop_training` state.')

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
async def train(log, run_event, get_state, set_state, execution_block, lr=5e-5, max_epochs=10):
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
                    if get_state('stop_training'):
                        raise StopLoop('Training stopped via `stop_training` state.')
                    new_batch = {
                        'batch_idx': batch_idx,
                        'batch': batch,
                    }
                    batch_block = execution_block(TrainEvents.BATCH, new_batch)
                    if batch_block.override is not None:
                        # An override replaces the whole batch body. The
                        # caller still owns the BATCH_AFTER lifecycle hooks.
                        await batch_block.execute()
                        await run_event(TrainEvents.BATCH, after=True)
                    else:
                        async with batch_block:
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
    except StopLoop as e:
        log.info(f'Training stopped early: {e}')
    finally:
        await run_event(TrainEvents.FIT, after=True)
