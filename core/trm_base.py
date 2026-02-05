#!/usr/bin/env python3
"""
trm_base.py - Shared Architecture for Dexter's Tiny Recursive Models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TRMConfig:
    vocab_size: int = 1024
    seq_len: int = 32
    hidden_size: int = 64
    num_heads: int = 2
    num_layers: int = 1
    num_classes: int = 0
    H_cycles: int = 2
    L_cycles: int = 2
    dropout: float = 0.1

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float = 4.0):
        super().__init__()
        inner_dim = int(hidden_size * expansion)
        self.fc1 = nn.Linear(hidden_size, inner_dim * 2, bias=False)
        self.fc2 = nn.Linear(inner_dim, hidden_size, bias=False)
    def forward(self, x):
        x, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * x)

class TRMBlock(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(config.hidden_size, config.num_heads, dropout=config.dropout, batch_first=True)
        self.norm2 = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class BaseTRM(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.seq_len, config.hidden_size)
        self.layers = nn.ModuleList([TRMBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.z_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        B, L = inputs.shape
        device = inputs.device
        pos = torch.arange(L, device=device).unsqueeze(0)
        x = self.embed(inputs) + self.pos_embed(pos)
        z = self.z_init.unsqueeze(0).unsqueeze(1).expand(B, L, -1)
        
        for h in range(self.config.H_cycles):
            for l in range(self.config.L_cycles):
                hidden = x + z
                for layer in self.layers:
                    hidden = layer(hidden)
                z = hidden if h == self.config.H_cycles - 1 else hidden.detach()
        
        out = self.norm(z).mean(dim=1)
        return self.classifier(out)
