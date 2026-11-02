"""Causal/windowed parallel features and explicit stateful single-step features."""

import torch
from torch import nn
from torch.nn import functional as F

from .cache import KVCache
from .config import TransformerConfig
from .normalization import RMSNorm
from .position import RotaryEmbedding, apply_rotary


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim, bias=False)
        self.out = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_eps, affine=False) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(config.head_dim, config.rms_eps, affine=False) if config.qk_norm else nn.Identity()
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


class SwiGLU(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.gate_up = nn.Linear(config.hidden_dim, 2 * config.ffn_dim, bias=False)
        self.down = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        for weight in self.gate_up.weight.chunk(2, dim=0):
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.rms_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_eps)
        self.attention = Attention(config)
        self.ffn = SwiGLU(config)
        self.alpha_attn = nn.Parameter(torch.tensor(float(config.alpha_init)))
        self.alpha_ffn = nn.Parameter(torch.tensor(float(config.alpha_init)))

    def finish(self, x, attention):
        x = x + self.alpha_attn.to(attention.dtype) * attention
        ffn = self.ffn(self.ffn_norm(x))
        return x + self.alpha_ffn.to(ffn.dtype) * ffn

    def forward(self, x, cos, sin, mask, is_causal):
        q, k, v = self.attention.project(self.attn_norm(x), cos, sin)
        return self.finish(x, self.attention.attend(q, k, v, mask, is_causal))


class TransformerBackbone(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.output_norm = RMSNorm(config.hidden_dim, config.rms_eps) if config.final_norm else nn.Identity()

    def _check_input(self, x):
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("x must be a floating Tensor")
        if x.ndim != 3 or x.shape[-1] != self.config.hidden_dim or x.shape[0] <= 0 or x.shape[1] <= 0:
            raise ValueError("x must have nonempty shape (B, T, hidden_dim)")
        if x.device != self.blocks[0].attention.qkv.weight.device:
            raise ValueError("x must be on the backbone device")

    def _projection_dtype(self, x):
        dtype = self.blocks[0].attention.qkv.weight.dtype
        if torch.is_autocast_enabled(x.device.type) and dtype != torch.float64:
            return torch.get_autocast_dtype(x.device.type)
        return dtype

    @staticmethod
    def _check_sequence_tensor(tensor, x, name, dtypes):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype not in dtypes:
            raise TypeError(f"{name} has an unsupported dtype")
        if tensor.shape != x.shape[:2] or tensor.device != x.device:
            raise ValueError(f"{name} must have shape (B,T) on the input device")

    def _sequence_metadata(self, x, episode_starts, valid_mask, position_ids, attn_mask, is_causal):
        b, t, _ = x.shape
        time = torch.arange(t, device=x.device)
        if episode_starts is not None:
            self._check_sequence_tensor(episode_starts, x, "episode_starts", (torch.bool,))
        if valid_mask is not None:
            self._check_sequence_tensor(valid_mask, x, "valid_mask", (torch.bool,))
        if position_ids is not None:
            self._check_sequence_tensor(position_ids, x, "position_ids", (torch.int32, torch.int64))
        elif episode_starts is None:
            position_ids = time.expand(b, t)
        else:
            last_start = torch.where(episode_starts, time, 0).cummax(dim=-1).values
            position_ids = time - last_start

        # Use the native causal fast path when no extra constraints are needed.
        if episode_starts is None and valid_mask is None and attn_mask is None:
            if not is_causal or t <= self.config.context_length:
                return position_ids, None, is_causal

        allowed = torch.ones((t, t), device=x.device, dtype=torch.bool)
        if is_causal:
            distance = time[:, None] - time[None, :]
            allowed = (distance >= 0) & (distance < self.config.context_length)
        allowed = allowed[None, None, :, :]
        if episode_starts is not None:
            segments = episode_starts.long().cumsum(dim=-1)
            allowed = allowed & (segments[:, None, :, None] == segments[:, None, None, :])
        if valid_mask is not None:
            allowed = allowed & valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
        if attn_mask is not None:
            if not isinstance(attn_mask, torch.Tensor) or (attn_mask.dtype != torch.bool and not attn_mask.is_floating_point()):
                raise TypeError("attn_mask must be a boolean or floating Tensor")
            if attn_mask.device != x.device:
                raise ValueError("attn_mask must be on the input device")
            if attn_mask.ndim == 2:
                correct_shape = attn_mask.shape == (t, t)
            elif attn_mask.ndim == 4:
                correct_shape = (attn_mask.shape[0] in (1, b) and
                                 attn_mask.shape[1] in (1, self.config.num_heads) and
                                 attn_mask.shape[2:] == (t, t))
            else:
                correct_shape = False
            if not correct_shape:
                raise ValueError("attn_mask must have shape (T,T) or (B,1/H,T,T)")
            if attn_mask.dtype == torch.bool:
                allowed = allowed & attn_mask
            else:
                allowed = attn_mask.to(self._projection_dtype(x)).masked_fill(~allowed, float("-inf"))
        return position_ids, allowed, False

    def forward(self, x, *, episode_starts=None, valid_mask=None, position_ids=None,
                attn_mask=None, is_causal: bool = True) -> torch.Tensor:
        """Parallel features; no KV state is read or written. True masks permit attention."""
        self._check_input(x)
        if type(is_causal) is not bool:
            raise TypeError("is_causal must be bool")
        positions, mask, native_causal = self._sequence_metadata(
            x, episode_starts, valid_mask, position_ids, attn_mask, is_causal
        )
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
        cos, sin = self.rope(positions)
        for block in self.blocks:
            x = block(x, cos, sin, mask, native_causal)
        x = self.output_norm(x)
        return x if valid_mask is None else x.masked_fill(~valid_mask.unsqueeze(-1), 0)

    def create_cache(self, batch_size: int, *, device=None, dtype=None) -> KVCache:
        weight = self.blocks[0].attention.qkv.weight
        return KVCache(self.config, batch_size, device=weight.device if device is None else device,
                       dtype=weight.dtype if dtype is None else dtype)

    def step(self, x, cache: KVCache, *, episode_starts=None, enable_grad: bool = False) -> torch.Tensor:
        """Advance the whole batch by one token; history is always detached.

        enable_grad=True differentiates the current step, not through time.
        Backward can be delayed until after subsequent step/reset calls.
        """
        if type(enable_grad) is not bool:
            raise TypeError("enable_grad must be bool")
        if enable_grad and torch.is_inference_mode_enabled():
            raise RuntimeError("enable_grad=True requires leaving torch.inference_mode()")
        self._check_input(x)
        if x.shape[1] != 1:
            raise ValueError("step expects shape (B,1,D); use forward for full sequences")
        if not isinstance(cache, KVCache):
            raise TypeError("cache must be a KVCache created for this backbone")
        cache.validate(self.config, x.shape[0], x.device, self._projection_dtype(x))
        if episode_starts is not None:
            cache.validate_mask(episode_starts)

        with torch.set_grad_enabled(enable_grad):
            if episode_starts is not None:
                cache.reset(episode_starts)
            positions = cache.next_positions.clone().unsqueeze(1)
            cos, sin = self.rope(positions)
            indices = positions.remainder(self.config.context_length).view(x.shape[0], 1, 1, 1)
            indices = indices.expand(-1, self.config.num_heads, 1, self.config.head_dim)
            lengths = (cache.lengths + 1).clamp(max=self.config.context_length)
            allowed = (cache.slot_indices[None, :] < lengths[:, None])[:, None, None, :]
            for layer, block in enumerate(self.blocks):
                q, k, v = block.attention.project(block.attn_norm(x), cos, sin)
                if enable_grad:
                    # Out-of-place scatter snapshots history AND connects current
                    # K/V. Neither saved tensor aliases mutable persistent storage.
                    keys = cache.keys[layer].detach().scatter(2, indices, k)
                    values = cache.values[layer].detach().scatter(2, indices, v)
                else:
                    keys = cache.keys[layer].scatter_(2, indices, k)
                    values = cache.values[layer].scatter_(2, indices, v)
                x = block.finish(x, block.attention.attend(q, keys, values, allowed, False))
                if enable_grad:
                    with torch.no_grad():
                        cache.keys[layer].scatter_(2, indices, k.detach())
                        cache.values[layer].scatter_(2, indices, v.detach())
            with torch.no_grad():
                cache.next_positions.add_(1)
                cache.lengths.copy_(lengths)
            return self.output_norm(x)
