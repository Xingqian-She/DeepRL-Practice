"""Head-free aligned temporal encoder/decoder with explicit causal cross memory."""

import torch
from torch import nn

from .cache import EncoderDecoderCache
from .config import EncoderDecoderConfig
from .normalization import RMSNorm
from .position import apply_rotary
from .transformer import Attention, TransformerBackbone, TransformerBlock, cached_attention


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads, self.head_dim = config.num_heads, config.head_dim
        self.q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.kv = nn.Linear(config.hidden_dim, 2 * config.hidden_dim, bias=False)
        self.out = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_eps, affine=False) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(config.head_dim, config.rms_eps, affine=False) if config.qk_norm else nn.Identity()
        nn.init.xavier_uniform_(self.q.weight)
        for weight in self.kv.weight.chunk(2, dim=0):
            nn.init.xavier_uniform_(weight)
        nn.init.xavier_uniform_(self.out.weight)

    def project_query(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        return apply_rotary(self.q_norm(q), cos, sin)

    def project_memory(self, memory, cos, sin):
        b, t, _ = memory.shape
        k, v = self.kv(memory).view(b, t, 2, self.num_heads, self.head_dim).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        return apply_rotary(self.k_norm(k), cos, sin), v

    def attend(self, q, k, v, mask, is_causal):
        return Attention.attend(self, q, k, v, mask, is_causal)


class DecoderBlock(TransformerBlock):
    def __init__(self, config):
        super().__init__(config)
        self.cross_norm = RMSNorm(config.hidden_dim, config.rms_eps)
        self.memory_norm = RMSNorm(config.hidden_dim, config.rms_eps)
        self.cross_attention = CrossAttention(config)
        self.alpha_cross = nn.Parameter(torch.tensor(float(config.alpha_init)))

    def cross_finish(self, x, attention):
        x = x + self.alpha_cross.to(attention.dtype) * attention
        ffn = self.ffn(self.ffn_norm(x))
        return x + self.alpha_ffn.to(ffn.dtype) * ffn

    def forward(self, x, memory, cos, sin, mask, is_causal):
        q, k, v = self.attention.project(self.attn_norm(x), cos, sin)
        attention = self.attention.attend(q, k, v, mask, is_causal)
        x = x + self.alpha_attn.to(attention.dtype) * attention
        q = self.cross_attention.project_query(self.cross_norm(x), cos, sin)
        # Intentionally differentiable: actor gradients train encoder history.
        k, v = self.cross_attention.project_memory(self.memory_norm(memory), cos, sin)
        return self.cross_finish(x, self.cross_attention.attend(q, k, v, mask, is_causal))

    def step(self, x, memory, cache, layer, cos, sin, indices, allowed, enable_grad):
        q, k, v = self.attention.project(self.attn_norm(x), cos, sin)
        attention = cached_attention(self.attention, q, k, v, cache.decoder_keys[layer],
                                     cache.decoder_values[layer], indices, allowed, enable_grad)
        x = x + self.alpha_attn.to(attention.dtype) * attention
        q = self.cross_attention.project_query(self.cross_norm(x), cos, sin)
        # Only H_t is projected; past per-layer cross KV are already in the ring.
        k, v = self.cross_attention.project_memory(self.memory_norm(memory), cos, sin)
        cross = cached_attention(self.cross_attention, q, k, v, cache.cross_keys[layer],
                                 cache.cross_values[layer], indices, allowed, enable_grad)
        return self.cross_finish(x, cross)


class EncoderDecoderBackbone(nn.Module):
    """Two projected (B,T,D) streams -> (encoder history H, decoder features Z).

    Both streams describe the same timesteps and share all temporal metadata.
    ``forward`` retains the full history graph and never reads/writes a cache.
    ``step`` advances a fixed batch once, with detached historical state. With
    enable_grad=True, current H_t remains differentiable through cross-attention;
    the graph survives subsequent cache writes/reset, but does not span steps.
    """
    def __init__(self, config: EncoderDecoderConfig):
        super().__init__()
        if not isinstance(config, EncoderDecoderConfig):
            raise TypeError("config must be an EncoderDecoderConfig")
        self.config = config
        stack_config = config.encoder_config()
        self.encoder = TransformerBackbone(stack_config)
        self.decoder = nn.ModuleList([DecoderBlock(stack_config) for _ in range(config.decoder_layers)])
        self.output_norm = RMSNorm(config.hidden_dim, config.rms_eps) if config.final_norm else nn.Identity()

    def _check_inputs(self, encoder_inputs, decoder_inputs):
        self.encoder._check_input(encoder_inputs)
        self.encoder._check_input(decoder_inputs)
        if encoder_inputs.shape != decoder_inputs.shape or encoder_inputs.dtype != decoder_inputs.dtype:
            raise ValueError("encoder/decoder inputs must have the same (B,T,D) shape and dtype")

    def forward(self, encoder_inputs, decoder_inputs, *, episode_starts=None, valid_mask=None, position_ids=None):
        self._check_inputs(encoder_inputs, decoder_inputs)
        positions, mask, native_causal = self.encoder._sequence_metadata(
            encoder_inputs, episode_starts, valid_mask, position_ids, None, True,
        )
        cos, sin = self.encoder.rope(positions)
        memory = self.encoder._forward_with_metadata(
            encoder_inputs, cos, sin, mask, native_causal, valid_mask,
        )
        x = decoder_inputs if valid_mask is None else decoder_inputs.masked_fill(~valid_mask.unsqueeze(-1), 0)
        for block in self.decoder:
            x = block(x, memory, cos, sin, mask, native_causal)
        x = self.output_norm(x)
        if valid_mask is not None:
            x = x.masked_fill(~valid_mask.unsqueeze(-1), 0)
        return memory, x

    def create_cache(self, batch_size: int, *, device=None, dtype=None) -> EncoderDecoderCache:
        weight = self.encoder.blocks[0].attention.qkv.weight
        return EncoderDecoderCache(self.config, batch_size, device=weight.device if device is None else device,
                                   dtype=weight.dtype if dtype is None else dtype)

    def step(self, encoder_token, decoder_token, cache, *, episode_starts=None, enable_grad=False):
        if type(enable_grad) is not bool:
            raise TypeError("enable_grad must be bool")
        if enable_grad and torch.is_inference_mode_enabled():
            raise RuntimeError("enable_grad=True requires leaving torch.inference_mode()")
        self._check_inputs(encoder_token, decoder_token)
        if encoder_token.shape[1] != 1:
            raise ValueError("step expects shape (B,1,D); use forward for full sequences")
        if not isinstance(cache, EncoderDecoderCache):
            raise TypeError("cache must be an EncoderDecoderCache created for this backbone")
        cache.validate(self.config, encoder_token.shape[0], encoder_token.device,
                       self.encoder._projection_dtype(encoder_token))
        if episode_starts is not None:
            cache.validate_mask(episode_starts)
        with torch.set_grad_enabled(enable_grad):
            positions, indices, lengths, allowed = cache._prepare_step(episode_starts)
            cos, sin = self.encoder.rope(positions)
            memory = self.encoder._step_with_metadata(
                encoder_token, cache.encoder_keys, cache.encoder_values, cos, sin, indices, allowed, enable_grad,
            )
            x = decoder_token
            for layer, block in enumerate(self.decoder):
                x = block.step(x, memory, cache, layer, cos, sin, indices, allowed, enable_grad)
            cache._advance(lengths)
            return memory, self.output_norm(x)
