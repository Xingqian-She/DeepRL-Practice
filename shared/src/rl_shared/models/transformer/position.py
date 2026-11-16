import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Interleaved 1D RoPE. Integer buffers survive model.half()/bfloat16()."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.register_buffer("pair_indices", torch.arange(0, head_dim, 2), persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Recompute the tiny frequency vector in FP32: casting a floating buffer
        # back from BF16 would not recover the original frequencies.
        inv_freq = self.theta ** (-self.pair_indices.float() / self.head_dim)
        phase = position_ids.float().unsqueeze(-1) * inv_freq
        return phase.cos().unsqueeze(1), phase.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
