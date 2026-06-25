#!/usr/bin/env python3

import torch
import torch.nn as nn

class BERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=1024,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        dropout=0.1,
    ):
        super().__init__()

        # embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.seg_emb = nn.Embedding(2, d_model)

        # encoder (PyTorch does the heavy lifting)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # saves your sanity
            norm_first=True,   # BERT-style pre-LN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        B, T = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = (
            self.token_emb(input_ids)
            + self.pos_emb(pos)
            + self.seg_emb(token_type_ids)
        )

        # PyTorch expects padding mask: True = ignore
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0  # (B, T)

        x = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        x = self.ln(x)

        return x
