import torch

from coffeetrain.plugin import Plugin

torch_compile = Plugin(
    name='torch_compile',
    description='Compiles the model and manages cuda graph step boundaries',
)

# TODO tie into model prepare plugin to compile the model


@torch_compile.system('MODEL_AFTER')
def compile_the_model(model, log):
    log.info('Compiling torch model')
    model = torch.compile(model)
    return {'model': model}

@torch_compile.system(['BATCH_BEFORE', 'EVAL_BATCH_BEFORE'])
async def mark_step_begin():
    """
    Signal CUDAGraphs step boundary at start of each iteration
    Must be called before any computation for torch.compile with max-autotune
    """
    torch.compiler.cudagraph_mark_step_begin()
