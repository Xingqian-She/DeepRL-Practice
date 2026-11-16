"""Configurable residual Transformer blocks."""

import torch
from torch import nn
from .config import TransformerConfig
from .components import make_norm
from .attention import Attention
from .feedforward import make_ffn


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = make_norm(config, config.hidden_dim)
        self.ffn_norm = make_norm(config, config.hidden_dim)
        self.attention = Attention(config)
        self.ffn = make_ffn(config)
        self.alpha_attn = nn.Parameter(torch.tensor(float(config.alpha_init)))
        self.alpha_ffn = nn.Parameter(torch.tensor(float(config.alpha_init)))

    def finish(self, x, attention):
        x = x + self.alpha_attn.to(attention.dtype) * attention
        ffn = self.ffn(self.ffn_norm(x))
        return x + self.alpha_ffn.to(ffn.dtype) * ffn

    def forward(self, x, cos, sin, mask, is_causal):
        q, k, v = self.attention.project(self.attn_norm(x), cos, sin)
        return self.finish(x, self.attention.attend(q, k, v, mask, is_causal))


