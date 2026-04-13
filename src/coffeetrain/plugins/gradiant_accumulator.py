from coffeetrain.plugin.Plugin

gradient_accumulation = Plugin(
    name='gradient_accumulation',
    description='Enables gradient accumulation in your training loop.'
)

@gradient_accumulation.override_block('BACKWARD')
def run_backward(batch_idx, grad_accum):
    if (batch_idx + 1) % grad_accum == 0:
        # Could almost move this into it's own plugin, the only reason not to is
        # because accelerator does this here too
        if grad_clip_norm is not None:
            for optimizer in optimizers:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )
        for optmizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
        #FIXME just like with hf one, this is done by the scheduler plugin itself
        # AFTER_BACKWARD
        for scheduler in schedulers:
            scheduler.step()
