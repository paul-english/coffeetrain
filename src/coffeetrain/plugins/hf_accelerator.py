hf_accelerator = Plugin(
    name='hf_accelerator',
    description="Hooks HuggingFace's `accelerator` library into your training.",
)

# TODO need the param for grad_accum
@hf_accelerator.override_block('BATCH')
async def run_batch(acceleartor, model, grad_accum, global_step):
    context_manager = accelerator.accumulate(model)
    with context_manager:
        async with execution_block('FORWARD', requires='outputs'):
            ...
        async with execution_block('LOSS', requires='loss'):
            batch = get_state('batch')
            outputs = get_state('outputs')
            loss = loss_fn(outputs, batch)
            scaled_loss = loss / grad_accum
            set_state('loss', scaled_loss)
        async with execution_black('BACKWARD'):
            loss = get_state('loss')
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                if grad_clip_norm is not None:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        grad_clip_norm
                    )
                for optimizer in optimizers:
                    optimizer.step()
                    optimizers.zero_grad()
                # TODO these handle themselves in AFTER_BACKWARD
                # Unless there's a reason this should only happen during sync grads?
                for scheduler in schedulers:
                    scheduler.step()
                # TODO new event for sync gradients?
    set_state('global_step', global_step + 1)
