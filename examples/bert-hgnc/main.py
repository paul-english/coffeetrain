#!/usr/bin/env python3
"""Train a small BERT/ModernBERT masked-language model on HGNC data.

This example uses the v2 event/plugin API. Run from this directory with::

    uv run python main.py train --size nano --model_name bert --batch_size 8
"""

from enum import Enum
from typing import Literal

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from coffeetrain import Trainer
from models import BERT, ModernBERT

trainer = Trainer()

ModelSize = Literal['nano', 'base']


class ModelName(str, Enum):
    bert = 'bert'
    modernbert = 'modernbert'


def get_model_params(size: ModelSize):
    if size == 'nano':
        return {
            'num_layers': 2,
            'nhead': 6,
            'd_model': 312,
            'max_len': 512,
            'vocab_size': 50257,
        }
    if size == 'base':
        return {
            'num_layers': 12,
            'nhead': 12,
            'd_model': 768,
            'max_len': 1024,
            'vocab_size': 50257,
        }
    raise ValueError(f'Unknown model size: {size!r}')


@trainer.system('DATA_BEFORE')
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        'huawei-noah/TinyBERT_General_4L_312D'
    )
    return {
        'tokenizer': tokenizer,
        'mask_token_id': tokenizer.mask_token_id,
    }


@trainer.system('DATA_BEFORE')
def load_model_params(size: ModelSize = 'nano'):
    return {'model_params': get_model_params(size)}


@trainer.system('DATA')
def load_data(tokenizer, model_params, batch_size: int = 8):
    dataset = load_dataset('Ezi/Human_gene_HGNC')

    def add_text(example):
        return {'text': str(example)}

    def tokenize(example):
        return tokenizer(example['text'], truncation=True)

    dataset = dataset.map(add_text)
    dataset = dataset.map(tokenize, batched=True)

    # Keep only fields consumed by the tokenizer/collator. Filter against the
    # actual dataset schema because it can change between dataset revisions.
    remove_columns = [
        'HGNC ID', 'Approved symbol', 'Approved name', 'Status',
        'Previous symbols', 'Alias symbols', 'Chromosome', 'Accession numbers',
        'RefSeq IDs', 'Locus type', 'Previous name', 'OMIM ID(supplied by OMIM)',
        'NCBI Gene ID(supplied by NCBI)', 'LNCipedia ID (supplied by LNCipedia)',
        'RefSeq(supplied by NCBI)', 'Alias names', 'Locus group', 'Date approved',
        'Enzyme IDs', 'Gene group ID', 'text',
    ]
    for split in dataset:
        columns = [c for c in remove_columns if c in dataset[split].column_names]
        if columns:
            dataset[split] = dataset[split].remove_columns(columns)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='max_length',
        max_length=model_params['max_len'],
        pad_to_multiple_of=128,
    )
    return {
        'train_dataloader': DataLoader(
            dataset['train'],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
        ),
        'eval_dataloader': DataLoader(
            dataset['test'],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=data_collator,
        ),
    }


@trainer.system('MODEL_BEFORE')
def create_model(
    model_params,
    mask_token_id,
    model_name: ModelName = ModelName.bert,
):
    compute_device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    if model_name is ModelName.bert:
        model = BERT(**model_params)
    elif model_name is ModelName.modernbert:
        model = ModernBERT(**model_params)
    else:
        raise ValueError(f'Unknown model: {model_name!r}')

    lm_head = nn.Linear(model_params['d_model'], model_params['vocab_size'])
    model.to(compute_device)
    lm_head.to(compute_device)
    return {
        'model': model,
        'lm_head': lm_head,
        'loss_fn': nn.CrossEntropyLoss(),
        'compute_device': compute_device,
        'device': None,
        'mask_token_id': mask_token_id,
    }


@trainer.system('OPTIMIZER_BEFORE')
def create_optimizer(model, lm_head, lr: float = 3e-4):
    return {
        'optimizer': AdamW(
            list(model.parameters()) + list(lm_head.parameters()),
            lr=lr,
            weight_decay=0.01,
        ),
        'lr': lr,
    }


@trainer.system(['FORWARD_BEFORE', 'EVAL_FORWARD_BEFORE'])
def prepare_batch(batch, compute_device):
    attention_mask = batch.get('attention_mask')
    return {
        'input_ids': batch['input_ids'].to(compute_device),
        'attention_mask': (
            attention_mask.to(compute_device)
            if attention_mask is not None else None
        ),
    }


@trainer.system(['FORWARD', 'EVAL_FORWARD'])
def masked_language_model_forward(
    model,
    lm_head,
    input_ids,
    attention_mask,
    mask_token_id,
    model_params,
):
    input_ids = input_ids.clone()
    labels = input_ids.clone()
    probability = torch.full_like(input_ids, 0.3, dtype=torch.float32)
    masked = torch.bernoulli(probability).bool()
    special_tokens = (input_ids == 0) | (input_ids == 101) | (input_ids == 102)
    if attention_mask is not None:
        special_tokens |= attention_mask == 0
    masked &= ~special_tokens
    labels[~masked] = -100
    input_ids[masked] = mask_token_id

    hidden_states = model(input_ids, attention_mask=attention_mask)
    logits = lm_head(hidden_states)
    return {
        'outputs': logits.view(-1, model_params['vocab_size']),
        'labels': labels.view(-1),
    }


@trainer.system('EPOCH_BEFORE')
def set_training_mode(model, lm_head):
    model.train()
    lm_head.train()


@trainer.system('EVAL_BEFORE')
def set_eval_mode(model, lm_head):
    model.eval()
    lm_head.eval()


@trainer.system('BACKWARD_AFTER')
def optimizer_step(optimizer):
    optimizer.step()
    optimizer.zero_grad()


if __name__ == '__main__':
    trainer()
