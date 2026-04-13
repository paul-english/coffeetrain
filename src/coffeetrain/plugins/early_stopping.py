#!/usr/bin/env python3
from coffeetrain.plugin import Plugin

early_stopping = Plugin(
    name='early_stopping',
    description='Stops the network early if the chosen metric has plateaued. The training loop must check for `stop_training` on BATCH_END'
)

@early_stopping.system('FIT_BEFORE')
def not_implemented_yet():
    # TODO FIXME
    #raise NotImplementedError('Needs migration from v1')
    ...

@early_stopping.system('BATCH_AFTER')
def should_we_stop():
    # TODO
    should_stop = False # TODO
    if should_stop:
        raise StopLoop('Early stopping training, {metric} has plateaued at {value}')
