"""HuggingFace Accelerate integration for Trainer.

Hooks ``accelerate.Accelerator`` into training via a ``BATCH`` override: the
override runs forward/loss through the trainer's own events, then calls
``accelerator.backward`` and (on sync steps) the optimizer/scheduler steps.

Requires the optional ``accelerate`` package::

    pip install accelerate

Usage:
  from coffeetrain.plugins.hf_accelerator import hf_accelerator
  from coffeetrain import Trainer

  trainer = Trainer()
  trainer.register_plugin(hf_accelerator)

Parameters (all optional with sensible defaults):
  - hf_accelerator_gradient_accumulation_steps: microbatches per optimizer step
    (default: 1)
  - hf_accelerator_grad_clip_norm: max grad norm for clipping, None disables
    (default: None)
"""

from coffeetrain.plugin import Plugin

hf_accelerator = Plugin(
    name='hf_accelerator',
    description="Hooks HuggingFace's `accelerator` library into your training.",
)


@hf_accelerator.system('FIT_BEFORE')
def prepare_hf_accelerator(
    get_state,
    set_state,
    hf_accelerator_gradient_accumulation_steps: int = 1,
    hf_accelerator_grad_clip_norm=None,
):
    """Create the Accelerator and prepare model/optimizer(s) with it."""
    try:
        from accelerate import Accelerator
    except ImportError as e:
        raise ImportError(
            "hf_accelerator requires the `accelerate` package "
            "(pip install accelerate)"
        ) from e

    accelerator = Accelerator(
        gradient_accumulation_steps=hf_accelerator_gradient_accumulation_steps,
    )
    set_state({
        'hf_accelerator': accelerator,
        'hf_accelerator_gradient_accumulation_steps':
            hf_accelerator_gradient_accumulation_steps,
        'hf_accelerator_grad_clip_norm': hf_accelerator_grad_clip_norm,
    })

    model = get_state('model')
    optimizer = get_state('optimizer')
    optimizers = get_state('optimizers')
    if optimizers is None and optimizer is not None:
        optimizers = [optimizer]

    train_dataloader = get_state('train_dataloader')
    eval_dataloader = get_state('eval_dataloader')
    prepared = {'model': model}
    if optimizers:
        if len(optimizers) == 1:
            prepared['model'], prepared['optimizer'] = accelerator.prepare(
                model, optimizers[0]
            )
            prepared['optimizers'] = [prepared['optimizer']]
        else:
            prepared['model'], *prepared_opts = accelerator.prepare(
                model, *optimizers
            )
            prepared['optimizers'] = prepared_opts
            prepared['optimizer'] = prepared_opts[0] if prepared_opts else None
    else:
        prepared['model'] = accelerator.prepare(model)
    dataloaders = [
        loader for loader in (train_dataloader, eval_dataloader)
        if loader is not None
    ]
    if dataloaders:
        prepared_loaders = accelerator.prepare(*dataloaders)
        if len(dataloaders) == 1:
            prepared_loaders = (prepared_loaders,)
        prepared['train_dataloader'] = prepared_loaders[0]
        if eval_dataloader is not None:
            prepared['eval_dataloader'] = prepared_loaders[-1]
    set_state(prepared)
    return {
        'hf_accelerator': accelerator,
        'hf_accelerator_gradient_accumulation_steps':
            hf_accelerator_gradient_accumulation_steps,
        'hf_accelerator_grad_clip_norm': hf_accelerator_grad_clip_norm,
    }


@hf_accelerator.override_block('BATCH')
async def run_batch_with_accelerator(
    get_state,
    run_event,
    execution_block,
):
    """Run one batch through forward/loss events with accelerator.backward."""
    from coffeetrain.plugins.interruptable_train import TrainEvents

    accelerator = get_state('hf_accelerator')
    model = get_state('model')
    grad_clip_norm = get_state('hf_accelerator_grad_clip_norm')

    def _as_list(value):
        if value is None:
            return []
        return value if isinstance(value, list) else [value]

    with accelerator.accumulate(model):
        await execution_block(TrainEvents.FORWARD, requires='outputs', call=True)
        await execution_block(TrainEvents.LOSS, requires='loss', call=True)
        loss = get_state('loss')
        await run_event(TrainEvents.BACKWARD, before=True)
        # Do not run the default BACKWARD system: it calls loss.backward(),
        # which would double-backward this graph. Accelerate owns backward
        # scaling and accumulation inside `accumulate`.
        accelerator.backward(loss)

        if accelerator.sync_gradients:
            if grad_clip_norm is not None:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )
            # BACKWARD_AFTER contains the user's optimizer/scheduler systems.
            # Run it only on synchronized accumulation steps.
            await run_event(TrainEvents.BACKWARD, after=True)

    # The enclosing BATCH override caller runs BATCH_AFTER after this returns.
