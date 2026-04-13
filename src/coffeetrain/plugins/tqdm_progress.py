from tqdm import tqdm

from coffeetrain.plugin import Plugin

tqdm_progress = Plugin(
    name='tqdm_progress',
    description='Prints a tqdm progress bar to track training progress. Assumes standard training loop events.',
)

# TODO params: which metrics we're logging in postfix: e.g. loss,lr,step

@tqdm_progress.system('EPOCH_BEFORE')
def set_description(train_dataloader, epoch, max_epochs):
    # TODO make sure this doesn't happen each epoch...
    # could move to fit start, but we wouldn't have the desc there, might be ok
    # TODO need a way to track main_process
    #if is_main_process:

    # FIXME maybe we don't even wrap data loader, maybe we tie it to hooks.. that would work the same...
    # progress_bar.step() or whatever right here, setting desc
    # set_state('progress_bar')
    dataloader = tqdm(
        train_dataloader,
        desc=f"Epoch {epoch + 1}/{max_epochs}"
    )
    return {'train_dataloader': dataloader}

# TODO maybe we limit events by main process or not, it'd make main_process a more priveleged value
# Maybe instead we set up conditional systems based on state
#@tqdm_progress.system('BATCH_AFTER', main_process=True)
@tqdm_progress.system('BATCH_AFTER')
def update_postfix(loss, lr, global_step, train_dataloader):
    # TODO if is_main_process:
    # TODO need a way to set main proces... or limit to that...
    # TODO postfix is based on param...
    postfix = {
        'loss': f'{loss.item():.4f}' if loss is not None else '',
        'lr': f'{lr:.2e}',
        'step': global_step,
    }
    train_dataloader.set_postfix(postfix)

# TODO on epoch end close the epoch progress bar
# TODO on eval start, new progress bar
# TODO on eval step, update progress bar
# TODO on eval end close the eval progress bar
