"""Fixed-shape ring storage. Metadata, not zero filling, invalidates old episodes."""

import torch

from .config import TransformerConfig


class KVCache:
    """Caller-owned state for one fixed batch; treat the exposed tensors as read-only.

    ``next_positions`` is episode-local and unbounded by the ring capacity.
    ``lengths`` counts valid slots, capped at context_length. After reset, valid
    slots are filled from slot zero again, so slot < length is sufficient even
    when environments reset at different times.
    """

    def __init__(self, config: TransformerConfig, batch_size: int, *, device, dtype) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if dtype not in (torch.float32, torch.float16, torch.bfloat16, torch.float64):
            raise TypeError("cache dtype must be a supported floating dtype")
        self.config = config
        self.batch_size = batch_size
        shape = (config.num_layers, batch_size, config.num_heads, config.context_length, config.head_dim)
        # Ordinary tensors also allow a later enable_grad=True call when the
        # cache was originally created by an inference_mode rollout.
        with torch.inference_mode(False):
            self.keys = torch.zeros(shape, device=device, dtype=dtype)
            self.values = torch.zeros_like(self.keys)
            self.lengths = torch.zeros(batch_size, device=device, dtype=torch.long)
            self.next_positions = torch.zeros_like(self.lengths)
            self.slot_indices = torch.arange(config.context_length, device=device)

    @property
    def device(self) -> torch.device:
        return self.keys.device

    @property
    def dtype(self) -> torch.dtype:
        return self.keys.dtype

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (
            self.keys, self.values, self.lengths, self.next_positions, self.slot_indices
        ))

    def validate_mask(self, mask: torch.Tensor) -> None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise TypeError("reset/episode_starts mask must be a bool Tensor")
        if mask.shape != (self.batch_size,) or mask.device != self.device:
            raise ValueError("reset/episode_starts mask must have shape (B,) on the cache device")

    def reset(self, done_mask: torch.Tensor | None = None) -> None:
        if done_mask is not None:
            self.validate_mask(done_mask)
        with torch.no_grad():
            if done_mask is None:
                self.lengths.zero_()
                self.next_positions.zero_()
            else:
                self.lengths.masked_fill_(done_mask, 0)
                self.next_positions.masked_fill_(done_mask, 0)

    def validate(self, config: TransformerConfig, batch_size: int, device, dtype) -> None:
        if self.config != config:
            raise ValueError("cache structure/config does not match this backbone")
        if self.batch_size != batch_size:
            raise ValueError("cache batch_size does not match the input")
        if self.device != device or self.dtype != dtype:
            raise ValueError(f"cache device/dtype must match projected K/V ({device}, {dtype})")
