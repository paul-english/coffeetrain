#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- utils ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * x / (norm + self.eps)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RoPE(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2).float() / dim)
        )
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)

        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        return self.cos[:T], self.sin[:T]


# --- attention ---

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = dropout

    def forward(self, x, mask=None, cos=None, sin=None):
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        # RoPE
        if cos is not None:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )

        out = attn.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(out)


# --- feedforward (SwiGLU) ---

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# --- transformer block ---

class ModernBERTBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ff = SwiGLU(d_model, dim_ff)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, cos=None, sin=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask, cos, sin))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


# --- full model ---

class ModernBERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len=2048,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_ff=3072,
        dropout=0.1,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.rope = RoPE(d_model // nhead, max_len=max_len)

        self.layers = nn.ModuleList([
            ModernBERTBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(d_model)

    def forward(self, input_ids, attention_mask=None):
        x = self.token_emb(input_ids)

        cos, sin = self.rope(x)

        for layer in self.layers:
            x = layer(x, attention_mask, cos, sin)

        x = self.norm(x)

        return x
