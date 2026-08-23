from typing import Literal
from enum import Enum

import torch
import torch.nn as nn
from datasets import load_dataset
from coffeetrain import Trainer
from typer import Typer
from torch.optim import Muon
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from models import BERT, ModernBERT

trainer = Trainer()

ModelSize = Literal['nano', 'base']

class ModelName(str, Enum):
    bert = 'bert'
    modernbert = 'modernbert'

def get_model_params(size: ModelSize):
    match size:
        case 'nano':
            return {
                'num_layers': 2,
                'nhead': 6,
                'd_model': 312,
                'max_len': 512,
                'vocab_size': 50257,
            }
        case 'base':
            return {
                'num_layers': 12,
                'nhead': 12,
                'd_model': 768,
                'head_dim': 64,
                'max_len': 1024,
                'vocab_size': 50257,
            }
    raise NotImplemented



@trainer.system('EPOCH_BEFORE')
def start_training_model(model, lm_head):
    model.train()
    lm_head.train()
    return {'model': model, 'lm_head': lm_head}

@trainer.system('EVAL_BEFORE')
def start_evaluating_model(model, lm_head):
    model.eval()
    lm_head.eval()
    return {'model': model, 'lm_head': lm_head}

# TODO could move into a "standard optimize plugin" that conflicts with the
# grad accum style
# FIXME should be more baked in unless the user wants to override
@trainer.system('BACKWARD_AFTER')
def run_optimizers(optimizers):
    for optimizer in optimizers:
        optimizer.step()
    for optimizer in optimizers:
        optimizer.zero_grad()
    return {'optimizers': optimizers}

@trainer.system('DATA_BEFORE')
def get_encoder():
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    return {'tokenizer': tokenizer}

@trainer.system('DATA_BEFORE')
def run_model_params(size: ModelSize = 'nano'):
    model_params = get_model_params(size)
    return {'model_params': model_params}

@trainer.system('DATA')
def get_data_loader(log, tokenizer, model_params, batch_size=8):
    hgnc_dataset = load_dataset('Ezi/Human_gene_HGNC')

    def run_tokenizer(example):
        return tokenizer(example['text'], truncation=True)

    def add_text(example):
        return {'text': str(example)}

    remove_cols = [
        'HGNC ID', 'Approved symbol', 'Approved name', 'Status', 'Previous symbols',
        'Alias symbols', 'Chromosome', 'Accession numbers', 'RefSeq IDs', 'Locus type',
        'Previous name', 'OMIM ID(supplied by OMIM)', 'NCBI Gene ID(supplied by NCBI)',
        'LNCipedia ID (supplied by LNCipedia)', 'RefSeq(supplied by NCBI)', 'Alias names',
        'Locus group', 'Date approved', 'Enzyme IDs', 'Gene group ID',
        'text',
        #'input_ids',
        #'token_type_ids',
        #'attention_mask'
    ]


    hgnc_dataset = hgnc_dataset.map(add_text)
    hgnc_dataset = hgnc_dataset.map(run_tokenizer, batched=True)
    hgnc_dataset = hgnc_dataset.remove_columns(remove_cols)
    log.debug('HGNC train', extra={'hgnc_dataset': hgnc_dataset})
    #def collate_fn(x):
    #    print('----x', x)
    #    masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)
    #    log.debug('---', extra={'x': x, 'inputs': inputs, 'masked_index': masked_index})
    #    return x
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=model_params['max_len'],
        pad_to_multiple_of=128
    )
    train_dataloader = DataLoader(
        hgnc_dataset['train'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    train_dataloader = DataLoader(
        hgnc_dataset['test'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(
        hgnc_dataset['test'],
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    return {
        'train_dataloader': train_dataloader,
        'eval_dataloader': test_dataloader
    }

@trainer.system('MODEL')
def get_model(device, model_name: ModelName = 'bert', size: ModelSize = 'nano'):
    model_params = get_model_params(size)
    match model_name:
        case 'bert':
            model = BERT(**model_params)
        case 'modernbert':
            model = ModernBERT(**model_params)
        case _:
            raise NotImplemented

    # TODO if we combine bert & lm_head move to device could be automatic
    # also forward might be something we could clean up more
    lm_head = nn.Linear(
        model_params['d_model'],
        model_params['vocab_size']
    )

    loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    lm_head.to(device)

    return {
        'model': model,
        'lm_head': lm_head,
        'loss_fn': loss_fn,
    }

@trainer.system(['FORWARD', 'EVAL_FORWARD'])
def run_forward(model, lm_head, batch, model_params, device):
    input_ids = batch['input_ids']
    labels = batch['input_ids'].clone()

    # Create Mask: 30% of tokens are candidates for masking
    probability_matrix = torch.full(labels.shape, 0.3).to(device)
    #print('--p', probability_matrix)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    #print('--masked_indices', masked_indices)

    # Ensure we don't mask special tokens (assuming 0 is [PAD], 101 is [CLS], 102 is [SEP])
    special_tokens_mask = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)
    masked_indices &= ~special_tokens_mask

    # Set labels to -100 for non-masked tokens (CrossEntropy ignores -100)
    labels[~masked_indices] = -100

    # Apply [MASK] token (ID 103) to 80% of the masked_indices
    # (In a real BERT, 10% are replaced with random and 10% stay same, kept simple here)
    input_ids[masked_indices] = 103
    #print('----input_ids', input_ids)

    hidden_states = model(batch['input_ids'])
    logits = lm_head(hidden_states)

    #print('---hidden', hidden_states)
    #print('---logits', logits)
    #print('-labels', labels)
    return {
        'outputs': logits.view(-1, model_params['vocab_size']),
        'labels': labels.view(-1),
    }

@trainer.system('OPTIMIZER')
def get_optimizer(model, lm_head):
    from torch.optim import Muon, AdamW

    # Separate parameters for Muon vs. AdamW
    # Muon only handles >= 2D matrices (hidden weights).
    # 1D params (biases, layer-norm) and Embeddings use AdamW.
    muon_params = [p for p in list(model.parameters()) + list(lm_head.parameters()) if p.ndim >= 2 and p.requires_grad]
    adamw_params = [p for p in list(model.parameters()) + list(lm_head.parameters()) if p.ndim < 2 and p.requires_grad]

    # Initialize Optimizers
    # Muon typically uses a larger LR than AdamW (e.g., 0.02 vs 3e-4)
    opt1 = Muon(muon_params, lr=0.02, momentum=0.95)
    opt2 = AdamW(adamw_params, lr=3e-4, weight_decay=0.01)

    return {
        'optimizers': [opt1, opt2]
    }


#@trainer.command()
# TODO get all this into v2 format
def main(
        lr:float=5e-5,
        weight_decay:float=0.01,
        momentum:float=0.95,
        nesterov:bool=True,
        ns_steps:int=5,
        adjust_lr_fn: Literal['original', 'match_rms_adamw'] = 'match_rms_adamw',
        # Implement as "trainer"
        **kwargs,
    ):
    opt = Muon(
        params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        adjust_lr_fn=adjust_lr_fn,
    )

    # Systems
    callbacks = [
        # artifacts
        HistoryCallback(save_dir="output"), # What is this...
        CheckpointingCallback(),
        HubAutoPublishCheckpointer(
            repo_id=f'paul-english/flat-{model_name}-base-hgnc',
            metric_name='val_f1',
        ),
        BestModelCheckpointer(
            save_dir='output',
            metric_name='loss',
            mode='min',
            track_train_metrics=True,
            save_steps=500,
        ),

        # logging
        WandBCallback(project='flat-bert'),
        ParameterCounter(),
        LRMonitor(),
        GPUSystemMonitor(),
        SpeedMonitor(),
        ProgressCallback(),
        SchedulerLoggerCallback(),

        # Metrics
        F1Callback(
            mode='validate',
            plots=[
                'roc',
                'precision_recall',
                'confusion_matrix',
            ]
        ),
        UtilityCallback(
            # FIXME what counts towards costs in mlm?
            cost_fp=0.4,
            cost_tp=-0.1,
            cost_fn=1.0,
            cost_tn=-0.1,
            plots=[
                'utility_strata',
            ]
        ),

        # training
        BatchSizeScheduler(
            microbatch_size=4,
            start_batch_size=4,
            final_batch_size=256,
            warmup_steps=5000,
            schedule_type='equal_steps',
        ),
        # adds stop_training component
        EarlyStopping(),
        EMA(),
        MaskedLanguageModeling(
            mask_probability=0.3,
        ),
        WeightStableDecayLRScheduler(
            opt,
            warmup_steps=7000,
            stable_steps=2000,
            decay_steps=total_steps - 7000 - 2000,
            decay_type='1-sqrt',
        ),

        # Adds: grad_accum state
        GradientAccumulation(
            grad_accum=128,
            grad_clip_norm=1.0,
        ),

        # data efficiency
        PackingCallback(
            efficiency_threshold=0.95,
            max_sequences_per_bin=32,
        ),
        UnpaddingCallback(),

        # Model initialization
        MegatronInitialization(
            std=0.02
        ),

        # Enables more seemless to_device()
        # automatically upgrades to cuda (allows device but doesn't require it)
        CUDADeviceAcceleration(),
        # adds is_main_process, depends on CUDADeviceAcceleration
        # doesn't do anything if you're only using 1 gpu
        DistributedDataParallel(

        ),

        TorchCompile(
            compile_mode,
            disable_cudagraphs
        )
    ]

    # FIXME callbacks add a bunch of these, yeah...
    components = {
        'model': model,
        'dataloaders': {
            'train': train_dataloader,
            'val': test_dataloader,
        },
        'optimizers': opt,
    }

    #trainer = Trainer(systems=callbacks,
    #    components=components, # E.g. state
    #    # trainer requires model, opt, train_dataloader
    #    # val_dataloader enables validation phase
    #    # model add loss, which registers as metric in train & val
    #    # trainer adds:
    #    # - current batch,
    #    # - outputs
    #    # - epoch
    #    # - batch_idx
    #    # - global_step
    #    # - train_metrics
    #    # - val_metrics
    #    #
    #    # auto added property components
    #    # - is_training from model.training
    #    # - num_batches from len(self.train_dataloader)
    #    # - total_steps from num_batches * max_epochs....
    #    # - lr, uses optimizers & scheduler
    #)



if __name__ == "__main__":
    trainer()
