"""Self attention and detached-history cache updates."""

import torch
from torch import nn
from torch.nn import functional as F
from .config import TransformerConfig
from .components import make_norm
from .position import apply_rotary


def cached_attention(attention, q, k, v, key_store, value_store, indices, allowed, enable_grad):
    """Append current KV once; differentiable calls snapshot mutable history."""
    if enable_grad:
        keys = key_store.detach().scatter(2, indices, k)
        values = value_store.detach().scatter(2, indices, v)
    else:
        keys = key_store.scatter_(2, indices, k)
        values = value_store.scatter_(2, indices, v)
    result = attention.attend(q, keys, values, allowed, False)
    if enable_grad:
        with torch.no_grad():
            key_store.scatter_(2, indices, k.detach())
            value_store.scatter_(2, indices, v.detach())
    return result


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.q_norm = make_norm(config, config.head_dim, qk=True)
        self.k_norm = make_norm(config, config.head_dim, qk=True)
        for weight in self.qkv.weight.chunk(3, dim=0):
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.out.weight)

    def project(self, x, cos, sin):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return apply_rotary(self.q_norm(q), cos, sin), apply_rotary(self.k_norm(k), cos, sin), v

    def attend(self, q, k, v, mask, is_causal):
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, dropout_p=0.0)
        return self.out(y.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1))


