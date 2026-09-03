from coffeetrain.plugin import Plugin

cuda_accelerate = Plugin(
    name='cuda_accelerate',
    description='Attempts to automatically move your model and data to the right cuda device.'
)

# TODO model hook to move to device?
@cuda_accelerate.system('FIT_BEFORE')
def initialize_device(device=None):
    return {'device': device}

@cuda_accelerate.system(['BATCH_BEFORE', 'EVAL_BATCH_BEFORE'])
def move_to_gpu(batch, device):
    if device is not None:
        batch = batch.to(device, non_blocking=True)
        return {'batch': batch}
